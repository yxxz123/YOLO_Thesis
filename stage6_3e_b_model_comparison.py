#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 6.3E-B-2: nested sequence-grouped selector comparison.

Run from YOLO_Thesis repository root:
    python -u stage6_3e_b_model_comparison.py

Compares:
  M1 Rule-F0 (original 3 features)
  M2 Rule-F1 (expanded 8 features)
  M3 Logistic-F0
  M4 Logistic-F1
  M5 ShallowTree-F0
  M6 ShallowTree-F1

Outer evaluation: 5-fold GroupKFold by sequence_id.
Inner tuning: 4-fold GroupKFold, used for LogisticRegression/DecisionTree
hyperparameter and probability-threshold selection. The rule learner is a
fixed fitting algorithm; it is also evaluated in inner folds for diagnostics,
then fitted on the complete outer-training fold.

All OOF predictions use held-out sequences only. Labels are audit/training
targets, never model features. No YOLO training/inference.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from faster_coco_eval import COCO, COCOeval_faster

ROOT = Path(__file__).resolve().parent
SCENE_DIR = ROOT / "runs/detect/stage6_scene_selector"
ORACLE_DIR = ROOT / "runs/detect/stage6_selective_oracle"
C2_DIR = ROOT / "runs/detect/stage6_size_aware_fusion_iou07"
GT_JSON = ROOT / "runs/detect/source_validation/Stage5-Baseline-Unified/annotations_auto_detect_val.json"

FEATURE_CSV = SCENE_DIR / "stage6_3e_b_features/expanded_scene_features.csv"
FEATURE_AUDIT_JSON = SCENE_DIR / "stage6_3e_b_features/feature_audit.json"
UTILITY_CSV = SCENE_DIR / "coco_utility_analysis.csv"
A_JSON = ORACLE_DIR / "all_original_predictions.json"
C2_JSON = C2_DIR / "C2_area2048_predictions.json"
REFERENCE_JSON = C2_DIR / "faster_eval_metrics.json"
E3_SUMMARY_JSON = SCENE_DIR / "stage6_3e_budget_sweep/budget_sweep_summary.json"
E3_ASSIGNMENTS_CSV = SCENE_DIR / "stage6_3e_budget_sweep/budget_oof_assignments.csv"
E3_B80_PREDICTIONS = SCENE_DIR / "stage6_3e_budget_sweep/budget_80_predictions.json"

OUT_DIR = SCENE_DIR / "stage6_3e_b_model_comparison"

EXPECTED_TOTAL = 1567
EXPECTED_SOURCE_COUNTS = {"uavdt": 1320, "visdrone": 247}
EXPECTED_VIS_GROUPS = 94
EXPECTED_LABELS = {"beneficial": 158, "harmful": 89}
EXPECTED_NUMPY = "1.26.4"
EXPECTED_SKLEARN = "1.5.2"
EXPECTED_FASTER_COCO = "1.8.0"

F0 = [
    "n_det_conf025",
    "median_box_area_640",
    "tiny_ratio",
]
F1 = F0 + [
    "tiny_count",
    "det_density_per_mp",
    "mean_confidence_025",
    "box_area_cv_640",
    "grid_occupancy_4x4",
]
FEATURE_SETS = {"F0": F0, "F1": F1}
BUDGET = 0.80
MIN_TRIGGER_FRACTION = 0.10
MIN_TRIGGER_ABSOLUTE = 5
MAX_RULE_CONDITIONS = 2
QUANTILE_GRID = tuple(float(x) for x in np.arange(0.10, 0.91, 0.10))
OUTER_SPLITS = 5
INNER_SPLITS = 4
SEED = 42

EVAL_IOU_THRESHOLDS = np.linspace(0.50, 0.95, 10, dtype=np.float64)
EVAL_RECALL_THRESHOLDS = np.linspace(0.0, 1.0, 101, dtype=np.float64)
EVAL_MAX_DETS = [1, 10, 100]
EVAL_AREA_RANGES = [
    [0 ** 2, 1e5 ** 2],
    [0 ** 2, 32 ** 2],
    [32 ** 2, 96 ** 2],
    [96 ** 2, 1e5 ** 2],
]
EVAL_AREA_LABELS = ["all", "small", "medium", "large"]
METRIC_KEYS = ["AP50-95", "AP50", "APs", "APm", "APl"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json_atomic(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    tmp.replace(path)


def write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"拒绝写入空CSV：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def read_csv_by_id(path: Path) -> dict[int, dict[str, str]]:
    result: dict[int, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if "image_id" not in (reader.fieldnames or []):
            raise RuntimeError(f"{path}缺少image_id")
        for row in reader:
            image_id = int(row["image_id"])
            if image_id in result:
                raise RuntimeError(f"{path}重复image_id={image_id}")
            result[image_id] = row
    return result


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def check_versions() -> dict[str, str]:
    env = {
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "faster_coco_eval": version("faster-coco-eval"),
    }
    expected = {
        "numpy": EXPECTED_NUMPY,
        "scikit_learn": EXPECTED_SKLEARN,
        "faster_coco_eval": EXPECTED_FASTER_COCO,
    }
    if env != expected:
        raise RuntimeError(f"实验环境版本不匹配：actual={env}, expected={expected}")
    return env


def load_rows() -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]], dict[str, Any]]:
    feature_raw = read_csv_by_id(FEATURE_CSV)
    utility_raw = read_csv_by_id(UTILITY_CSV)
    if set(feature_raw) != set(utility_raw):
        raise RuntimeError("扩展特征CSV与utility CSV的image_id不一致")
    if len(feature_raw) != EXPECTED_TOTAL:
        raise RuntimeError(f"图片数={len(feature_raw)}，预期={EXPECTED_TOTAL}")

    all_rows: dict[int, dict[str, Any]] = {}
    source_counts = Counter()
    for image_id, f in feature_raw.items():
        u = utility_raw[image_id]
        for key in ("file_name", "source", "sequence_id"):
            if f[key] != u[key]:
                raise RuntimeError(f"image_id={image_id}的{key}在特征与标签文件中不一致")
        row: dict[str, Any] = {
            "image_id": image_id,
            "file_name": f["file_name"],
            "source": f["source"].strip().lower(),
            "sequence_id": f["sequence_id"],
            "width": int(f["width"]),
            "height": int(f["height"]),
            "selector_label": u["selector_label"].strip().lower(),
        }
        for feature in F1:
            value = f.get(feature, "")
            if value is None or value.strip() == "":
                row[feature] = None
            else:
                v = float(value)
                if not math.isfinite(v):
                    raise RuntimeError(f"非有限特征值 image_id={image_id} {feature}={value}")
                row[feature] = v
        if row["source"] not in EXPECTED_SOURCE_COUNTS:
            raise RuntimeError(f"未知source={row['source']}")
        source_counts[row["source"]] += 1
        all_rows[image_id] = row

    if dict(source_counts) != EXPECTED_SOURCE_COUNTS:
        raise RuntimeError(f"数据来源数量错误：{dict(source_counts)}")

    vis_rows = sorted(
        [r for r in all_rows.values() if r["source"] == "visdrone"],
        key=lambda r: r["image_id"],
    )
    label_counts = Counter(r["selector_label"] for r in vis_rows)
    if dict(label_counts) != EXPECTED_LABELS:
        raise RuntimeError(f"标签分布变化：{dict(label_counts)}，预期={EXPECTED_LABELS}")
    groups = {r["sequence_id"] for r in vis_rows}
    if len(groups) != EXPECTED_VIS_GROUPS:
        raise RuntimeError(f"VisDrone sequence_id数量={len(groups)}，预期={EXPECTED_VIS_GROUPS}")
    for r in vis_rows:
        if r["selector_label"] not in {"beneficial", "harmful"}:
            raise RuntimeError(f"未知标签：image_id={r['image_id']} label={r['selector_label']}")
    info = {
        "source_counts": dict(source_counts),
        "visdrone_label_counts": dict(label_counts),
        "visdrone_sequences": len(groups),
    }
    return vis_rows, all_rows, info


def y_utility(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([1 if r["selector_label"] == "beneficial" else -1 for r in rows],
                      dtype=np.int64)


def y_binary(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([1 if r["selector_label"] == "beneficial" else 0 for r in rows],
                      dtype=np.int64)


def feature_value(row: dict[str, Any], feature: str) -> float:
    value = row.get(feature)
    if value is None:
        if feature == "median_box_area_640":
            return float("inf")
        raise RuntimeError(f"特征缺失：image_id={row.get('image_id')} feature={feature}")
    return float(value)


def matrix_for(rows: list[dict[str, Any]], feature_names: list[str]) -> np.ndarray:
    values = []
    for row in rows:
        values.append([
            np.nan if row.get(name) is None else float(row[name])
            for name in feature_names
        ])
    return np.asarray(values, dtype=np.float64)


def rule_directions(feature: str, feature_set: str) -> list[str]:
    fixed = {
        "n_det_conf025": [">="],
        "median_box_area_640": ["<="],
        "tiny_ratio": [">="],
    }
    if feature in fixed:
        return fixed[feature]
    # Newly introduced features have no assumed direction; both directions
    # are searched inside the current training fold only.
    return [">=", "<="] if feature_set == "F1" else []


def condition_mask(rows: list[dict[str, Any]], condition: dict[str, Any]) -> np.ndarray:
    vals = np.asarray([feature_value(r, condition["feature"]) for r in rows], dtype=np.float64)
    threshold = float(condition["threshold"])
    if condition["op"] == ">=":
        return vals >= threshold
    if condition["op"] == "<=":
        return vals <= threshold
    raise RuntimeError(f"未知规则运算符：{condition['op']}")


def apply_rule(rows: list[dict[str, Any]], rule: dict[str, Any]) -> np.ndarray:
    if rule.get("rule_type") == "none":
        return np.zeros(len(rows), dtype=bool)
    conditions = rule.get("conditions", [])
    if not conditions:
        return np.zeros(len(rows), dtype=bool)
    mask = condition_mask(rows, conditions[0])
    for condition in conditions[1:]:
        mask &= condition_mask(rows, condition)
    return mask


def fit_rule(train_rows: list[dict[str, Any]], feature_set: str) -> dict[str, Any]:
    """Stage 6.3C-style 1- or 2-condition AND rule; training rows only."""
    features = FEATURE_SETS[feature_set]
    y = y_utility(train_rows)
    n_train = len(train_rows)
    min_selected = max(MIN_TRIGGER_ABSOLUTE,
                       int(math.ceil(MIN_TRIGGER_FRACTION * n_train)))
    max_selected = int(math.floor(BUDGET * n_train))
    if max_selected < min_selected:
        return {"rule_type": "none", "conditions": [], "reason": "budget_below_minimum"}

    thresholds: dict[str, list[float]] = {}
    directions: dict[str, list[str]] = {}
    for feature in features:
        vals = np.asarray([feature_value(r, feature) for r in train_rows], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        thresholds[feature] = (
            sorted({round(float(v), 8) for v in np.quantile(vals, QUANTILE_GRID)})
            if len(vals) else []
        )
        directions[feature] = rule_directions(feature, feature_set)

    best = None
    candidates = 0
    for n_conditions in (1, 2):
        for feature_combo in itertools.combinations(features, n_conditions):
            if any(not thresholds[f] or not directions[f] for f in feature_combo):
                continue
            option_lists = [
                [(op, threshold) for op in directions[f] for threshold in thresholds[f]]
                for f in feature_combo
            ]
            for combo in itertools.product(*option_lists):
                conditions = [
                    {"feature": feature, "op": op, "threshold": float(threshold)}
                    for feature, (op, threshold) in zip(feature_combo, combo)
                ]
                mask = apply_rule(train_rows, {"rule_type": "and", "conditions": conditions})
                n_selected = int(mask.sum())
                if n_selected < min_selected or n_selected > max_selected:
                    continue
                candidates += 1
                selected_y = y[mask]
                n_beneficial = int(np.sum(selected_y == 1))
                n_harmful = int(np.sum(selected_y == -1))
                precision = n_beneficial / n_selected if n_selected else 0.0
                net = float(np.sum(selected_y) / n_train)
                # Keep the original Stage 6.3C ranking/tie-break order.
                ranking = (net, precision, -n_conditions, n_selected)
                if best is None or ranking > best["ranking"]:
                    best = {
                        "ranking": ranking,
                        "rule_type": "and",
                        "conditions": conditions,
                        "train_images": n_train,
                        "train_selected": n_selected,
                        "train_beneficial_selected": n_beneficial,
                        "train_harmful_selected": n_harmful,
                        "train_trigger_rate": n_selected / n_train,
                        "train_net_label_utility_per_image": net,
                        "max_trigger_budget": BUDGET,
                        "candidate_rules_evaluated": candidates,
                    }

    if best is None or best["ranking"][0] <= 0:
        return {
            "rule_type": "none",
            "conditions": [],
            "reason": "no_positive_net_utility_rule_within_budget",
            "candidate_rules_evaluated": candidates,
            "max_trigger_budget": BUDGET,
        }
    best.pop("ranking")
    best["candidate_rules_evaluated"] = candidates
    return best


def selection_diagnostics(rows: list[dict[str, Any]], selected: np.ndarray) -> dict[str, Any]:
    labels = [r["selector_label"] for r in rows]
    n_sel = int(selected.sum())
    beneficial_total = sum(x == "beneficial" for x in labels)
    harmful_total = sum(x == "harmful" for x in labels)
    beneficial_sel = sum(bool(selected[i]) and labels[i] == "beneficial"
                         for i in range(len(rows)))
    harmful_sel = sum(bool(selected[i]) and labels[i] == "harmful"
                      for i in range(len(rows)))
    return {
        "images": len(rows),
        "selected_images": n_sel,
        "trigger_rate": n_sel / len(rows) if rows else 0.0,
        "beneficial_total": beneficial_total,
        "harmful_total": harmful_total,
        "beneficial_selected": beneficial_sel,
        "harmful_selected": harmful_sel,
        "beneficial_precision": beneficial_sel / n_sel if n_sel else 0.0,
        "beneficial_recall": beneficial_sel / beneficial_total if beneficial_total else 0.0,
        "harmful_capture_rate": harmful_sel / harmful_total if harmful_total else 0.0,
        "net_label_utility": beneficial_sel - harmful_sel,
        "net_label_utility_per_image": (
            (beneficial_sel - harmful_sel) / len(rows) if rows else 0.0
        ),
    }


def threshold_from_inner_oof(scores: np.ndarray, y: np.ndarray) -> tuple[float | None, np.ndarray, dict[str, Any]]:
    if len(scores) != len(y) or not np.all(np.isfinite(scores)):
        raise RuntimeError("inner OOF probability缺失或非有限")
    n = len(y)
    min_sel = max(MIN_TRIGGER_ABSOLUTE, int(math.ceil(MIN_TRIGGER_FRACTION * n)))
    max_sel = int(math.floor(BUDGET * n))
    best = None
    for threshold in sorted(set(float(x) for x in scores), reverse=True):
        selected = scores >= threshold
        count = int(selected.sum())
        if count < min_sel or count > max_sel:
            continue
        selected_y = y[selected]
        beneficial = int(np.sum(selected_y == 1))
        harmful = int(np.sum(selected_y == 0))
        precision = beneficial / count if count else 0.0
        net = float((beneficial - harmful) / n)
        ranking = (net, precision, -count)
        if best is None or ranking > best["ranking"]:
            best = {
                "ranking": ranking,
                "threshold": threshold,
                "selected": selected,
                "selected_count": count,
                "beneficial_selected": beneficial,
                "harmful_selected": harmful,
                "beneficial_precision": precision,
                "net_label_utility_per_image": net,
            }
    if best is None or best["ranking"][0] <= 0:
        selected = np.zeros(n, dtype=bool)
        return None, selected, {
            "selected_count": 0,
            "trigger_rate": 0.0,
            "beneficial_selected": 0,
            "harmful_selected": 0,
            "beneficial_precision": 0.0,
            "net_label_utility_per_image": 0.0,
            "reason": "no_positive_net_utility_threshold",
        }
    selected = best["selected"]
    details = {k: v for k, v in best.items() if k not in {"ranking", "selected"}}
    details["trigger_rate"] = details["selected_count"] / n
    return float(best["threshold"]), selected, details


def make_pipeline(family: str, config: dict[str, Any]) -> Pipeline:
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    if family == "logistic":
        return Pipeline([
            ("imputer", imputer),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                C=float(config["C"]),
                class_weight=config["class_weight"],
                penalty="l2",
                solver="lbfgs",
                max_iter=5000,
                random_state=SEED,
            )),
        ])
    if family == "tree":
        return Pipeline([
            ("imputer", imputer),
            ("classifier", DecisionTreeClassifier(
                max_depth=int(config["max_depth"]),
                min_samples_leaf=int(config["min_samples_leaf"]),
                class_weight=config["class_weight"],
                random_state=SEED,
            )),
        ])
    raise ValueError(f"未知模型family={family}")


def probability_for_positive(model: Pipeline, X: np.ndarray) -> np.ndarray:
    probs = model.predict_proba(X)
    classes = list(model.named_steps["classifier"].classes_)
    if 1 not in classes:
        raise RuntimeError(f"分类器缺少beneficial正类1，classes={classes}")
    return probs[:, classes.index(1)].astype(np.float64)


def logistic_configs() -> list[dict[str, Any]]:
    return [
        {"C": c, "class_weight": cw}
        for c in (0.01, 0.1, 1.0, 10.0)
        for cw in (None, "balanced")
    ]


def tree_configs() -> list[dict[str, Any]]:
    return [
        {"max_depth": depth, "min_samples_leaf": leaf, "class_weight": cw}
        for depth in (1, 2, 3)
        for leaf in (5, 10, 15)
        for cw in (None, "balanced")
    ]


def inner_rule_oof(
    outer_train_rows: list[dict[str, Any]],
    feature_set: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    groups = np.asarray([r["sequence_id"] for r in outer_train_rows], dtype=object)
    y = y_utility(outer_train_rows)
    dummy = np.zeros((len(outer_train_rows), 1), dtype=np.float32)
    scores = np.full(len(outer_train_rows), np.nan, dtype=np.float64)
    splitter = GroupKFold(n_splits=INNER_SPLITS)
    fold_details = []
    for inner_fold, (tr, va) in enumerate(splitter.split(dummy, y, groups), start=1):
        train_rows = [outer_train_rows[int(i)] for i in tr]
        val_rows = [outer_train_rows[int(i)] for i in va]
        if {r["sequence_id"] for r in train_rows} & {r["sequence_id"] for r in val_rows}:
            raise RuntimeError("inner GroupKFold出现sequence泄漏")
        rule = fit_rule(train_rows, feature_set)
        selected = apply_rule(val_rows, rule)
        scores[va] = selected.astype(np.float64)
        fold_details.append({
            "inner_fold": inner_fold,
            "train_images": len(tr),
            "validation_images": len(va),
            "rule": rule,
            "validation_diagnostics": selection_diagnostics(val_rows, selected),
        })
    if not np.all(np.isfinite(scores)):
        raise RuntimeError("inner rule OOF没有完整覆盖outer train")
    return scores.astype(bool), {
        "inner_oof_diagnostics": selection_diagnostics(outer_train_rows, scores.astype(bool)),
        "inner_folds": fold_details,
    }


def fit_classifier_nested(
    family: str,
    feature_set: str,
    outer_train_rows: list[dict[str, Any]],
    outer_test_rows: list[dict[str, Any]],
) -> tuple[np.ndarray, dict[str, Any]]:
    feature_names = FEATURE_SETS[feature_set]
    X_train = matrix_for(outer_train_rows, feature_names)
    X_test = matrix_for(outer_test_rows, feature_names)
    y_train = y_binary(outer_train_rows)
    groups_train = np.asarray([r["sequence_id"] for r in outer_train_rows], dtype=object)

    configs = logistic_configs() if family == "logistic" else tree_configs()
    inner_splitter = GroupKFold(n_splits=INNER_SPLITS)
    dummy = np.zeros((len(outer_train_rows), 1), dtype=np.float32)

    best = None
    config_records = []
    for config in configs:
        inner_scores = np.full(len(outer_train_rows), np.nan, dtype=np.float64)
        inner_fold_records = []
        for inner_fold, (tr, va) in enumerate(
            inner_splitter.split(dummy, y_train, groups_train), start=1
        ):
            if set(groups_train[tr]) & set(groups_train[va]):
                raise RuntimeError("inner classifier GroupKFold出现sequence泄漏")
            model = make_pipeline(family, config)
            model.fit(X_train[tr], y_train[tr])
            inner_scores[va] = probability_for_positive(model, X_train[va])
            inner_fold_records.append({
                "inner_fold": inner_fold,
                "train_images": len(tr),
                "validation_images": len(va),
            })
        if not np.all(np.isfinite(inner_scores)):
            raise RuntimeError("inner classifier OOF概率没有完整覆盖outer train")

        threshold, inner_selected, threshold_details = threshold_from_inner_oof(
            inner_scores, y_train
        )
        diagnostics = selection_diagnostics(outer_train_rows, inner_selected)
        ranking = (
            diagnostics["net_label_utility_per_image"],
            diagnostics["beneficial_precision"],
            -diagnostics["selected_images"],
        )
        record = {
            "config": config,
            "threshold": threshold,
            "inner_diagnostics": diagnostics,
            "threshold_selection": threshold_details,
            "inner_folds": inner_fold_records,
            "ranking": ranking,
        }
        config_records.append(record)
        if best is None or ranking > best["ranking"]:
            best = record

    if best is None:
        raise RuntimeError("分类器参数搜索没有产生候选")

    final_model = make_pipeline(family, best["config"])
    final_model.fit(X_train, y_train)
    test_scores = probability_for_positive(final_model, X_test)
    if best["threshold"] is None:
        selected = np.zeros(len(outer_test_rows), dtype=bool)
    else:
        selected = test_scores >= float(best["threshold"])

    details = {
        "family": family,
        "feature_set": feature_set,
        "selected_config": best["config"],
        "selected_probability_threshold": best["threshold"],
        "inner_training_diagnostics": best["inner_diagnostics"],
        "threshold_selection": best["threshold_selection"],
        "tested_config_count": len(configs),
        "all_config_inner_results": [
            {
                "config": x["config"],
                "threshold": x["threshold"],
                "inner_diagnostics": x["inner_diagnostics"],
                "ranking": list(x["ranking"]),
            }
            for x in config_records
        ],
    }
    return selected, {"details": details, "test_scores": test_scores}


def load_prediction_groups(path: Path, valid_ids: set[int]) -> dict[int, list[dict[str, Any]]]:
    raw = read_json(path)
    if not isinstance(raw, list):
        raise RuntimeError(f"预测文件不是COCO list：{path}")
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for pred in raw:
        image_id = int(pred["image_id"])
        if image_id not in valid_ids:
            raise RuntimeError(f"{path.name}包含未知image_id={image_id}")
        if len(pred["bbox"]) != 4:
            raise RuntimeError(f"bbox格式错误：{pred}")
        grouped[image_id].append(pred)
    return dict(grouped)


def prediction_signature(records: list[dict[str, Any]]) -> Counter:
    return Counter(
        (
            int(p["category_id"]),
            round(float(p["score"]), 8),
            tuple(round(float(v), 6) for v in p["bbox"]),
        )
        for p in records
    )


def check_uavdt_original_only(
    rows_by_id: dict[int, dict[str, Any]],
    original: dict[int, list[dict[str, Any]]],
    c2: dict[int, list[dict[str, Any]]],
) -> None:
    bad = []
    for image_id, row in rows_by_id.items():
        if row["source"] == "uavdt":
            if prediction_signature(original.get(image_id, [])) != prediction_signature(c2.get(image_id, [])):
                bad.append(image_id)
    if bad:
        raise RuntimeError(f"C2与Original在UAVDT上不一致，前20个ID：{bad[:20]}")


def write_combined_prediction(
    model_name: str,
    selected_ids: set[int],
    all_rows: dict[int, dict[str, Any]],
    original: dict[int, list[dict[str, Any]]],
    c2: dict[int, list[dict[str, Any]]],
    out_dir: Path,
) -> Path:
    records = []
    for image_id in sorted(all_rows):
        row = all_rows[image_id]
        use_c2 = row["source"] == "visdrone" and image_id in selected_ids
        records.extend(c2.get(image_id, []) if use_c2 else original.get(image_id, []))
    path = out_dir / "predictions" / f"{model_name.lower()}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, separators=(",", ":"))
    tmp.replace(path)
    return path


def evaluate_prediction(
    coco_gt: COCO,
    category_ids: list[int],
    prediction_path: Path,
    image_ids: list[int],
    scope: str,
) -> dict[str, float]:
    print(f"  COCO eval {prediction_path.name} | {scope} | images={len(image_ids)}", flush=True)
    coco_dt = coco_gt.loadRes(str(prediction_path))
    evaluator = COCOeval_faster(
        coco_gt, coco_dt, iouType="bbox", lvis_style=False, print_function=print
    )
    evaluator.params.imgIds = sorted(int(x) for x in image_ids)
    evaluator.params.catIds = list(category_ids)
    evaluator.params.iouThrs = EVAL_IOU_THRESHOLDS.copy()
    evaluator.params.recThrs = EVAL_RECALL_THRESHOLDS.copy()
    evaluator.params.maxDets = list(EVAL_MAX_DETS)
    evaluator.params.areaRng = [x.copy() for x in EVAL_AREA_RANGES]
    evaluator.params.areaRngLbl = list(EVAL_AREA_LABELS)
    evaluator.params.useCats = 1
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    stats = evaluator.stats_as_dict
    mapping = {
        "AP50-95": "AP_all",
        "AP50": "AP_50",
        "APs": "AP_small",
        "APm": "AP_medium",
        "APl": "AP_large",
    }
    result = {}
    for out_key, in_key in mapping.items():
        if in_key not in stats:
            raise RuntimeError(f"COCO evaluator缺少{in_key}; keys={list(stats.keys())}")
        result[out_key] = float(stats[in_key])
    del evaluator, coco_dt
    gc.collect()
    return result


def diagnostics_for_selected(
    rows: list[dict[str, Any]],
    selected: np.ndarray,
) -> dict[str, Any]:
    return selection_diagnostics(rows, selected)


def read_b80_reference() -> dict[int, dict[str, int]]:
    result = {}
    with E3_ASSIGNMENTS_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["budget"] != "B80":
                continue
            image_id = int(row["image_id"])
            if image_id in result:
                raise RuntimeError(f"Stage 6.3E-A B80重复image_id={image_id}")
            result[image_id] = {
                "fold": int(row["fold"]),
                "selected": int(row["selected_for_c2_2048"]),
            }
    if len(result) != 247:
        raise RuntimeError(f"Stage 6.3E-A B80 assignments={len(result)}，预期247")
    return result


def run_outer_cv(
    vis_rows: list[dict[str, Any]],
) -> tuple[dict[str, np.ndarray], dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    model_names = [
        "M1_Rule_F0",
        "M2_Rule_F1",
        "M3_Logistic_F0",
        "M4_Logistic_F1",
        "M5_Tree_F0",
        "M6_Tree_F1",
    ]
    selected_map = {name: np.zeros(len(vis_rows), dtype=bool) for name in model_names}
    score_map = {name: np.full(len(vis_rows), np.nan, dtype=np.float64) for name in model_names}
    fold_details = {name: [] for name in model_names}
    assignment_rows = []

    groups = np.asarray([r["sequence_id"] for r in vis_rows], dtype=object)
    dummy = np.zeros((len(vis_rows), 1), dtype=np.float32)
    outer_splitter = GroupKFold(n_splits=OUTER_SPLITS)

    for outer_fold, (train_idx, test_idx) in enumerate(
        outer_splitter.split(dummy, np.zeros(len(vis_rows)), groups), start=1
    ):
        train_rows = [vis_rows[int(i)] for i in train_idx]
        test_rows = [vis_rows[int(i)] for i in test_idx]
        train_groups = {r["sequence_id"] for r in train_rows}
        test_groups = {r["sequence_id"] for r in test_rows}
        if train_groups & test_groups:
            raise RuntimeError(f"Outer fold {outer_fold} sequence泄漏")
        print(
            f"\n[OUTER {outer_fold}/{OUTER_SPLITS}] "
            f"train={len(train_rows)} images/{len(train_groups)} seq; "
            f"test={len(test_rows)} images/{len(test_groups)} seq",
            flush=True,
        )

        # Fixed rule learners: inner OOF is diagnostic; the fixed rule is
        # finally fitted on the complete outer-training fold.
        for feature_set, model_name in (("F0", "M1_Rule_F0"), ("F1", "M2_Rule_F1")):
            inner_selected, inner_info = inner_rule_oof(train_rows, feature_set)
            rule = fit_rule(train_rows, feature_set)
            test_selected = apply_rule(test_rows, rule)
            for local_i, global_i in enumerate(test_idx):
                selected_map[model_name][int(global_i)] = bool(test_selected[local_i])
                score_map[model_name][int(global_i)] = float(bool(test_selected[local_i]))
            fold_details[model_name].append({
                "outer_fold": outer_fold,
                "train_groups": sorted(train_groups),
                "test_groups": sorted(test_groups),
                "inner_cv_diagnostic": inner_info,
                "outer_fitted_rule": rule,
                "outer_test_diagnostics": selection_diagnostics(test_rows, test_selected),
            })
            print(
                f"  {model_name}: selected={int(test_selected.sum())}/{len(test_rows)} "
                f"rule={rule.get('conditions', []) or 'NONE'}",
                flush=True,
            )

        for feature_set, family, model_name in (
            ("F0", "logistic", "M3_Logistic_F0"),
            ("F1", "logistic", "M4_Logistic_F1"),
            ("F0", "tree", "M5_Tree_F0"),
            ("F1", "tree", "M6_Tree_F1"),
        ):
            test_selected, info = fit_classifier_nested(
                family, feature_set, train_rows, test_rows
            )
            for local_i, global_i in enumerate(test_idx):
                selected_map[model_name][int(global_i)] = bool(test_selected[local_i])
                score_map[model_name][int(global_i)] = float(info["test_scores"][local_i])
            fold_details[model_name].append({
                "outer_fold": outer_fold,
                "train_groups": sorted(train_groups),
                "test_groups": sorted(test_groups),
                "model_selection": info["details"],
                "outer_test_diagnostics": selection_diagnostics(test_rows, test_selected),
            })
            print(
                f"  {model_name}: selected={int(test_selected.sum())}/{len(test_rows)} "
                f"config={info['details']['selected_config']} "
                f"threshold={info['details']['selected_probability_threshold']}",
                flush=True,
            )

        for local_i, global_i in enumerate(test_idx):
            row = vis_rows[int(global_i)]
            for model_name in model_names:
                assignment_rows.append({
                    "model": model_name,
                    "outer_fold": outer_fold,
                    "image_id": row["image_id"],
                    "file_name": row["file_name"],
                    "sequence_id": row["sequence_id"],
                    "selected_for_c2_2048": int(selected_map[model_name][int(global_i)]),
                    "selector_score": score_map[model_name][int(global_i)],
                    "selector_label_audit_only": row["selector_label"],
                    "n_det_conf025": row["n_det_conf025"],
                    "median_box_area_640": "" if row["median_box_area_640"] is None else row["median_box_area_640"],
                    "tiny_ratio": row["tiny_ratio"],
                    "tiny_count": row["tiny_count"],
                    "det_density_per_mp": row["det_density_per_mp"],
                    "mean_confidence_025": row["mean_confidence_025"],
                    "box_area_cv_640": row["box_area_cv_640"],
                    "grid_occupancy_4x4": row["grid_occupancy_4x4"],
                })

    for name, arr in selected_map.items():
        if len(arr) != 247 or not np.all(np.isfinite(score_map[name])):
            raise RuntimeError(f"{name}没有完整生成247张图片的OOF结果")
    if len(assignment_rows) != 247 * 6:
        raise RuntimeError(f"OOF assignments行数异常：{len(assignment_rows)}")
    return selected_map, fold_details, assignment_rows


def check_rule_f0_reproduces_b80(
    vis_rows: list[dict[str, Any]],
    selected_map: dict[str, np.ndarray],
    assignment_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    expected = read_b80_reference()
    actual = selected_map["M1_Rule_F0"]
    mismatches = []
    fold_mismatches = []
    actual_fold = {
        int(row["image_id"]): int(row["outer_fold"])
        for row in assignment_rows if row["model"] == "M1_Rule_F0"
    }
    for i, row in enumerate(vis_rows):
        image_id = int(row["image_id"])
        ref = expected[image_id]
        if int(actual[i]) != ref["selected"]:
            mismatches.append(image_id)
        if actual_fold.get(image_id) != ref["fold"]:
            fold_mismatches.append(image_id)
    if mismatches or fold_mismatches:
        raise RuntimeError(
            "M1_Rule_F0无法逐图片复现Stage 6.3E-A B80。"
            f"decision_mismatches={mismatches[:20]}, "
            f"fold_mismatches={fold_mismatches[:20]}。"
            "请勿继续评价，先排查GroupKFold划分/规则实现/排序/特征值。"
        )
    return {
        "status": "passed",
        "reference": "Stage 6.3E-A B80",
        "checked_images": len(vis_rows),
        "decision_mismatches": 0,
        "fold_mismatches": 0,
        "selected_images": int(actual.sum()),
        "trigger_rate": float(actual.mean()),
    }


def make_metrics_row(
    model_name: str,
    diag: dict[str, Any],
    full: dict[str, float],
    vis: dict[str, float],
    b80_full: dict[str, float],
    b80_vis: dict[str, float],
    c2_full: dict[str, float],
) -> dict[str, Any]:
    row = {
        "model": model_name,
        "selected_images": diag["selected_images"],
        "trigger_rate": diag["trigger_rate"],
        "beneficial_selected": diag["beneficial_selected"],
        "harmful_selected": diag["harmful_selected"],
        "beneficial_precision": diag["beneficial_precision"],
        "beneficial_recall": diag["beneficial_recall"],
        "harmful_capture_rate": diag["harmful_capture_rate"],
        "net_label_utility": diag["net_label_utility"],
    }
    for scope, metrics in (("full", full), ("visdrone", vis)):
        for key in METRIC_KEYS:
            row[f"{scope}_{key}"] = metrics[key]
    for key in ("AP50-95", "APs"):
        row[f"delta_full_{key}_vs_B80"] = full[key] - b80_full[key]
        row[f"delta_full_{key}_vs_C2"] = full[key] - c2_full[key]
        row[f"delta_visdrone_{key}_vs_B80"] = vis[key] - b80_vis[key]
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true",
                        help="仅覆盖Stage 6.3E-B模型对比独立目录")
    args = parser.parse_args()

    required = [
        FEATURE_CSV, FEATURE_AUDIT_JSON, UTILITY_CSV, A_JSON, C2_JSON,
        GT_JSON, REFERENCE_JSON, E3_SUMMARY_JSON, E3_ASSIGNMENTS_CSV,
        E3_B80_PREDICTIONS,
    ]
    missing = [p for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("缺少必要输入文件：\n" + "\n".join(map(str, missing)))
    if OUT_DIR.exists() and any(OUT_DIR.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"输出目录已有文件：{OUT_DIR}。检查后如需重跑，显式添加 --overwrite。"
        )

    env = check_versions()
    feature_audit = read_json(FEATURE_AUDIT_JSON)
    if feature_audit.get("status") != "completed":
        raise RuntimeError("特征提取审计状态不是completed")
    vis_rows, all_rows, data_info = load_rows()
    all_ids = set(all_rows)
    vis_ids = sorted(r["image_id"] for r in vis_rows)
    uavdt_ids = sorted(i for i, r in all_rows.items() if r["source"] == "uavdt")

    print("=" * 88)
    print("Stage 6.3E-B-2 | Nested GroupKFold model comparison")
    print("M1-M6 | Outer=5 GroupKFold | Inner=4 GroupKFold")
    print("No YOLO training. No YOLO inference. Development validation only.")
    print("=" * 88)
    print("Environment:", env)
    print("Data:", data_info)

    # Read only the existing prediction files.
    original = load_prediction_groups(A_JSON, all_ids)
    c2 = load_prediction_groups(C2_JSON, all_ids)
    check_uavdt_original_only(all_rows, original, c2)
    print("[PASS] UAVDT original-only prediction identity check.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    selected_map, fold_details, assignment_rows = run_outer_cv(vis_rows)

    # Strict audit: the original three-feature rule must reproduce E-A B80.
    reproduction = check_rule_f0_reproduces_b80(
        vis_rows, selected_map, assignment_rows
    )
    print("[PASS] M1 Rule-F0 reproduces E-A B80 image-by-image:", reproduction)

    write_csv_atomic(OUT_DIR / "nested_oof_assignments.csv", assignment_rows)
    write_json_atomic(OUT_DIR / "nested_fold_models.json", {
        "experiment": "Stage6.3E-B_Nested_GroupKFold",
        "outer_splitter": "GroupKFold(n_splits=5)",
        "inner_splitter": "GroupKFold(n_splits=4)",
        "group_key": "sequence_id",
        "budget": BUDGET,
        "fold_models": fold_details,
    })

    # Build OOF combined predictions for each selector.
    prediction_paths = {}
    for model_name, selected in selected_map.items():
        selected_ids = {
            vis_rows[i]["image_id"] for i in range(len(vis_rows)) if bool(selected[i])
        }
        prediction_paths[model_name] = write_combined_prediction(
            model_name, selected_ids, all_rows, original, c2, OUT_DIR
        )

    # Exact prediction identity check against E-A B80 for M1.
    m1_path = prediction_paths["M1_Rule_F0"]
    m1_raw = read_json(m1_path)
    e3_raw = read_json(E3_B80_PREDICTIONS)
    if prediction_signature(m1_raw) != prediction_signature(e3_raw):
        raise RuntimeError("M1 Rule-F0生成的完整预测与Stage 6.3E-A B80不一致。")
    print("[PASS] M1 Rule-F0 combined prediction exactly matches E-A B80.")

    # Read reference metrics for the original, C2, B70 and B80 baselines.
    reference = read_json(REFERENCE_JSON)
    e3_summary = read_json(E3_SUMMARY_JSON)
    ref_results = reference["results"]
    baseline_metrics = {
        "A_Original": ref_results["A_Original"],
        "C2_area2048": ref_results["C2_area2048"],
        "B70": e3_summary["budgets"]["B70"]["coco_metrics"],
        "B80": e3_summary["budgets"]["B80"]["coco_metrics"],
    }
    b80_full = baseline_metrics["B80"]["full"]
    b80_vis = baseline_metrics["B80"]["visdrone"]
    c2_full = baseline_metrics["C2_area2048"]["full"]

    print("\n[COCO] Loading GT and evaluating six OOF policies...")
    coco_gt = COCO(str(GT_JSON))
    gt_ids = sorted(int(x["id"]) for x in coco_gt.dataset["images"])
    if gt_ids != sorted(all_ids):
        raise RuntimeError("GT image IDs与1567张特征数据不一致")
    category_ids = sorted(int(x) for x in coco_gt.getCatIds())
    if not category_ids:
        raise RuntimeError("GT没有类别")
    metrics_by_model = {}
    csv_rows = []
    for model_name in selected_map:
        print(f"\n===== {model_name} =====", flush=True)
        path = prediction_paths[model_name]
        full_metrics = evaluate_prediction(coco_gt, category_ids, path, sorted(all_ids), "full_1567")
        vis_metrics = evaluate_prediction(coco_gt, category_ids, path, vis_ids, "visdrone_247")
        diag = selection_diagnostics(vis_rows, selected_map[model_name])
        metrics_by_model[model_name] = {
            "selection_diagnostics": diag,
            "full": full_metrics,
            "visdrone": vis_metrics,
            "delta_full_vs_B80": {
                k: full_metrics[k] - b80_full[k] for k in METRIC_KEYS
            },
            "delta_full_vs_C2": {
                k: full_metrics[k] - c2_full[k] for k in METRIC_KEYS
            },
            "prediction_file": str(path.relative_to(ROOT)),
            "prediction_sha256": sha256_file(path),
        }
        csv_rows.append(make_metrics_row(
            model_name, diag, full_metrics, vis_metrics, b80_full, b80_vis, c2_full
        ))
        print(
            f"[{model_name}] trigger={diag['selected_images']}/247 "
            f"({diag['trigger_rate']:.1%}); "
            f"Full AP50-95={full_metrics['AP50-95']:.6f}; "
            f"Full APs={full_metrics['APs']:.6f}; "
            f"VisDrone APs={vis_metrics['APs']:.6f}",
            flush=True,
        )

    write_csv_atomic(OUT_DIR / "model_comparison_metrics.csv", csv_rows)

    # Rank only for development-set interpretation; Stage 6.4 remains the
    # untouched independent test. Primary ranking: Full APs, then Full AP50-95,
    # then lower trigger rate.
    ranking = sorted(
        metrics_by_model.items(),
        key=lambda kv: (
            kv[1]["full"]["APs"],
            kv[1]["full"]["AP50-95"],
            -kv[1]["selection_diagnostics"]["trigger_rate"],
        ),
        reverse=True,
    )
    summary = {
        "experiment": "Stage6.3E-B_Expanded_Feature_Classifier_Comparison",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "completed",
        "interpretation_limit": (
            "Nested sequence-grouped OOF results on the 247 VisDrone development images. "
            "They are not an independent external test. The final choice must be confirmed "
            "on the untouched independent test set in Stage 6.4."
        ),
        "strict_scope": {
            "no_yolo_training": True,
            "no_yolo_inference": True,
            "no_new_dataset": True,
            "no_test_set_used": True,
            "stage6_3c_files_modified": False,
            "stage6_3d_files_modified": False,
            "stage6_3e_a_files_modified": False,
            "labels_used_only_for_training_and_posthoc_diagnostics": True,
            "all_features_available_before_tile_inference": True,
        },
        "protocol": {
            "total_images": EXPECTED_TOTAL,
            "source_counts": data_info["source_counts"],
            "visdrone_images": len(vis_rows),
            "visdrone_sequences": data_info["visdrone_sequences"],
            "visdrone_labels": data_info["visdrone_label_counts"],
            "outer_cv": "GroupKFold(n_splits=5), grouped by sequence_id",
            "inner_cv": "GroupKFold(n_splits=4), grouped by sequence_id",
            "budget": BUDGET,
            "minimum_trigger_fraction": MIN_TRIGGER_FRACTION,
            "feature_sets": {"F0": F0, "F1": F1},
            "models": [
                "M1_Rule_F0", "M2_Rule_F1",
                "M3_Logistic_F0", "M4_Logistic_F1",
                "M5_Tree_F0", "M6_Tree_F1",
            ],
            "logistic_grid": logistic_configs(),
            "tree_grid": tree_configs(),
            "rule_conditions": "1 or 2 conditions combined with AND; existing feature directions fixed; new feature directions searched within training folds",
            "prediction_policy": "UAVDT always A_Original; VisDrone uses C2_area2048 only when selected by outer-fold model",
            "evaluator": "faster_coco_eval",
            "evaluator_version": EXPECTED_FASTER_COCO,
            "imgsz": 640,
            "conf": 0.001,
            "nms_iou": 0.7,
            "max_det": 300,
        },
        "environment": env,
        "input_sha256": {
            "expanded_scene_features.csv": sha256_file(FEATURE_CSV),
            "feature_audit.json": sha256_file(FEATURE_AUDIT_JSON),
            "utility_csv": sha256_file(UTILITY_CSV),
            "original_predictions": sha256_file(A_JSON),
            "c2_predictions": sha256_file(C2_JSON),
            "ground_truth": sha256_file(GT_JSON),
            "stage6_3e_a_summary": sha256_file(E3_SUMMARY_JSON),
            "stage6_3e_a_assignments": sha256_file(E3_ASSIGNMENTS_CSV),
        },
        "baseline_metrics": baseline_metrics,
        "m1_rule_f0_reproduction_of_stage6_3e_a_b80": reproduction,
        "models": metrics_by_model,
        "development_ranking": [
            {"rank": i + 1, "model": name,
             "full_APs": values["full"]["APs"],
             "full_AP50_95": values["full"]["AP50-95"],
             "trigger_rate": values["selection_diagnostics"]["trigger_rate"]}
            for i, (name, values) in enumerate(ranking)
        ],
        "outputs": {
            "summary": str((OUT_DIR / "model_comparison_summary.json").relative_to(ROOT)),
            "metrics": str((OUT_DIR / "model_comparison_metrics.csv").relative_to(ROOT)),
            "assignments": str((OUT_DIR / "nested_oof_assignments.csv").relative_to(ROOT)),
            "fold_models": str((OUT_DIR / "nested_fold_models.json").relative_to(ROOT)),
            "predictions": {
                name: str(path.relative_to(ROOT)) for name, path in prediction_paths.items()
            },
        },
    }
    write_json_atomic(OUT_DIR / "model_comparison_summary.json", summary)

    print("\n" + "=" * 88)
    print("Stage 6.3E-B completed.")
    print("Best development candidate:", ranking[0][0])
    print("Do not treat this as independent test performance.")
    print("Outputs:", OUT_DIR)
    print("=" * 88)


if __name__ == "__main__":
    main()
