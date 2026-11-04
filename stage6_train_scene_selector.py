#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 6.3C: GroupKFold out-of-fold scene selector for adaptive C2-2048 inference.

Run from the repository root:
    python -u stage6_train_scene_selector.py

This script DOES NOT train YOLO and DOES NOT run model inference.
It learns simple, interpretable threshold rules from Stage 6.3B2 per-image
utility labels, applies them out-of-fold by video sequence, creates three
OOF prediction files, and evaluates them with the fixed COCO protocol.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import itertools
import json
import math
import platform
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import GroupKFold
from importlib.metadata import version
from faster_coco_eval import COCO, COCOeval_faster


# =====================================================================
# 1. Fixed paths and protocol
# =====================================================================
ROOT = Path(__file__).resolve().parent
SCENE_DIR = ROOT / "runs/detect/stage6_scene_selector"
ORACLE_DIR = ROOT / "runs/detect/stage6_selective_oracle"
C2_DIR = ROOT / "runs/detect/stage6_size_aware_fusion_iou07"
GT_JSON = ROOT / "runs/detect/source_validation/Stage5-Baseline-Unified/annotations_auto_detect_val.json"

FEATURE_CSV = SCENE_DIR / "corrected_scene_features.csv"
UTILITY_CSV = SCENE_DIR / "coco_utility_analysis.csv"
MAPPING_CSV = ORACLE_DIR / "image_source_mapping.csv"
A_JSON = ORACLE_DIR / "all_original_predictions.json"
C2_JSON = C2_DIR / "C2_area2048_predictions.json"
REFERENCE_JSON = C2_DIR / "faster_eval_metrics.json"

OUT_ASSIGNMENTS = SCENE_DIR / "selector_oof_assignments.csv"
OUT_RULES = SCENE_DIR / "selector_fold_rules.json"
OUT_SUMMARY = SCENE_DIR / "selector_oof_summary.json"

EXPECTED_COUNTS = {"uavdt": 1320, "visdrone": 247}
EXPECTED_TOTAL = 1567
N_SPLITS = 5

# These are predeclared maximum trigger budgets on each training fold.
# They are NOT selected using the held-out fold.
POLICY_BUDGETS = {
    "Loose": 0.70,
    "Medium": 0.50,
    "Strict": 0.30,
}
MIN_TRIGGER_FRACTION = 0.10
MIN_TRIGGER_ABSOLUTE = 5
MAX_RULE_CONDITIONS = 2
QUANTILE_GRID = tuple(float(x) for x in np.arange(0.10, 0.91, 0.10))

# Feature directions are fixed from the Stage 6.3B2 observed relationships.
# n_det_conf025: more detections -> more likely useful to tile
# median_box_area_640: smaller median boxes -> more likely useful to tile
# tiny_ratio: higher tiny-object ratio -> more likely useful to tile
FEATURE_SPECS = {
    "n_det_conf025": {"op": ">="},
    "median_box_area_640": {"op": "<="},
    "tiny_ratio": {"op": ">="},
}
FEATURE_COLUMNS = list(FEATURE_SPECS)

EXPECTED_SHA256 = {
    "features": "db33bd0a0cd47b32298d36fab9fd99e7f8835def5a03a28bed8c7cadc566106d",
    "original": "4979c79d6966539b51f67592f8bdf10efc21d71f126065d13c730dc20a79fc23",
    "c2": "7e018f981f2cbcf39b512226892b58a067655c5e49c4ace8a6c07282ca3ef5b6",
    "gt": "75098d09c3a02e8f8e32eab99df6c977474748dd177a97b1692f766bbdcf0de4",
}

# Fixed COCO protocol copied from stage6_eval_faster.py.
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


# =====================================================================
# 2. Utility functions
# =====================================================================
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)


def finite_float(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"字段 {field} 不是数值：{value!r}") from exc
    if not math.isfinite(result):
        raise RuntimeError(f"字段 {field} 包含非有限值：{value!r}")
    return result


def feature_value(row: dict, name: str) -> float:
    value = row[name]
    if value is None or value == "":
        # Missing median area means no detections at feature confidence.
        # +inf will never satisfy median_box_area_640 <= threshold.
        if name == "median_box_area_640":
            return float("inf")
        raise RuntimeError(f"特征 {name} 缺失，image_id={row.get('image_id')}")
    return float(value)


def prediction_signature(records: list[dict]) -> Counter:
    return Counter(
        (
            int(p["category_id"]),
            round(float(p["score"]), 8),
            tuple(round(float(v), 6) for v in p["bbox"]),
        )
        for p in records
    )


def check_required_files() -> None:
    required = [FEATURE_CSV, UTILITY_CSV, MAPPING_CSV, A_JSON, C2_JSON, GT_JSON, REFERENCE_JSON]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("缺少以下输入文件：\n" + "\n".join(missing))
    SCENE_DIR.mkdir(parents=True, exist_ok=True)


def check_hashes_and_reference() -> dict:
    actual = {
        "features": sha256_file(FEATURE_CSV),
        "original": sha256_file(A_JSON),
        "c2": sha256_file(C2_JSON),
        "gt": sha256_file(GT_JSON),
    }
    for key, expected in EXPECTED_SHA256.items():
        if actual[key] != expected:
            raise RuntimeError(
                f"SHA256不匹配：{key}\nexpected={expected}\nactual={actual[key]}\n"
                "当前文件不是已经审阅的Stage 6.3B2版本；停止运行，先核对仓库提交。"
            )

    ref = load_json(REFERENCE_JSON)
    if ref.get("prediction_sha256", {}).get("A_Original") != actual["original"]:
        raise RuntimeError("faster_eval_metrics.json中的A_Original哈希不匹配。")
    if ref.get("prediction_sha256", {}).get("C2_area2048") != actual["c2"]:
        raise RuntimeError("faster_eval_metrics.json中的C2_area2048哈希不匹配。")
    if ref.get("ground_truth", {}).get("sha256") != actual["gt"]:
        raise RuntimeError("faster_eval_metrics.json中的GT哈希不匹配。")

    protocol = ref.get("protocol", {})
    if protocol.get("evaluator_version") != "1.8.0":
        raise RuntimeError("参考评价文件不是faster-coco-eval==1.8.0。")
    if int(protocol.get("imgsz", -1)) != 640 or float(protocol.get("nms_iou", -1)) != 0.7:
        raise RuntimeError("参考评价文件的imgsz/NMS IoU与当前固定协议不一致。")
    if version("faster-coco-eval") != "1.8.0":
        raise RuntimeError(
            f"当前faster-coco-eval版本为{version('faster-coco-eval')}，要求1.8.0。"
        )
    return {"sha256": actual, "reference": ref}


# =====================================================================
# 3. Load features, labels and source mapping
# =====================================================================
def load_csv_by_id(path: Path) -> dict[int, dict]:
    rows = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            image_id = int(raw["image_id"])
            if image_id in rows:
                raise RuntimeError(f"{path.name}存在重复image_id={image_id}")
            rows[image_id] = raw
    return rows


def load_feature_and_utility_rows() -> tuple[dict[int, dict], dict]:
    feature_raw = load_csv_by_id(FEATURE_CSV)
    utility_raw = load_csv_by_id(UTILITY_CSV)
    if len(feature_raw) != EXPECTED_TOTAL or len(utility_raw) != EXPECTED_TOTAL:
        raise RuntimeError(
            f"CSV行数错误：features={len(feature_raw)}, utility={len(utility_raw)}，"
            f"预期均为{EXPECTED_TOTAL}。"
        )
    if set(feature_raw) != set(utility_raw):
        raise RuntimeError("corrected_scene_features.csv与coco_utility_analysis.csv的image_id不一致。")

    feature_by_id = {}
    for image_id, frow in feature_raw.items():
        urow = utility_raw[image_id]
        for key in ("file_name", "source", "sequence_id"):
            if frow[key] != urow[key]:
                raise RuntimeError(f"image_id={image_id}的{key}在两个CSV中不一致。")
        row = {
            "image_id": image_id,
            "file_name": frow["file_name"],
            "source": frow["source"].strip().lower(),
            "sequence_id": frow["sequence_id"],
            "n_det_conf025": int(frow["n_det_conf025"]),
            "median_box_area_640": (
                None if frow["median_box_area_640"] in ("", None)
                else finite_float(frow["median_box_area_640"], "median_box_area_640")
            ),
            "tiny_ratio": finite_float(frow["tiny_ratio"], "tiny_ratio"),
            "tiny_count": int(frow["tiny_count"]),
            # Labels and utility fields are audit/target fields only.
            # They are never passed to apply_rule().
            "selector_label": urow["selector_label"].strip().lower(),
            "selector_label_reason": urow.get("selector_label_reason", ""),
            "small_gt_count": int(float(urow["A_small_gt_count"])),
            "delta_small_mean_f1_50_95": finite_float(
                urow["delta_small_mean_f1_50_95"], "delta_small_mean_f1_50_95"
            ),
            "delta_small_mean_fp_50_95": finite_float(
                urow["delta_small_mean_fp_50_95"], "delta_small_mean_fp_50_95"
            ),
        }
        if row["source"] not in EXPECTED_COUNTS:
            raise RuntimeError(f"未知source={row['source']}，image_id={image_id}")
        feature_by_id[image_id] = row

    source_counts = Counter(row["source"] for row in feature_by_id.values())
    if dict(source_counts) != EXPECTED_COUNTS:
        raise RuntimeError(f"数据来源数量不匹配：{dict(source_counts)}")

    vis_rows = [r for r in feature_by_id.values() if r["source"] == "visdrone"]
    label_counts = Counter(r["selector_label"] for r in vis_rows)
    if label_counts != Counter({"beneficial": 158, "harmful": 89}):
        raise RuntimeError(
            f"Stage 6.3B2标签分布与已审阅版本不一致：{dict(label_counts)}。"
        )
    for row in vis_rows:
        if row["selector_label"] not in {"beneficial", "harmful", "neutral"}:
            raise RuntimeError(
                f"VisDrone标签非法：image_id={row['image_id']}, label={row['selector_label']}"
            )
    return feature_by_id, {"source_counts": dict(source_counts), "label_counts": dict(label_counts)}


def load_and_check_mapping(feature_by_id: dict[int, dict]) -> dict[int, str]:
    mapping = {}
    with MAPPING_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            image_id = int(row["image_id"])
            if image_id in mapping:
                raise RuntimeError(f"image_source_mapping.csv重复image_id={image_id}")
            mapping[image_id] = row["source"].strip().lower()
            if image_id not in feature_by_id:
                raise RuntimeError(f"mapping出现未知image_id={image_id}")
            if mapping[image_id] != feature_by_id[image_id]["source"]:
                raise RuntimeError(f"mapping与feature来源不一致，image_id={image_id}")
            if row["file_name"] != feature_by_id[image_id]["file_name"]:
                raise RuntimeError(f"mapping与feature文件名不一致，image_id={image_id}")
    if set(mapping) != set(feature_by_id):
        raise RuntimeError("mapping与feature的image_id集合不一致。")
    return mapping


# =====================================================================
# 4. Load predictions and verify UAVDT is original-only
# =====================================================================
def load_prediction_groups(path: Path, valid_ids: set[int]) -> tuple[dict[int, list[dict]], int]:
    raw = load_json(path)
    if not isinstance(raw, list):
        raise RuntimeError(f"预测文件不是JSON list：{path}")
    grouped = defaultdict(list)
    for p in raw:
        image_id = int(p["image_id"])
        if image_id not in valid_ids:
            raise RuntimeError(f"预测文件包含未知image_id={image_id}：{path}")
        for key in ("category_id", "bbox", "score"):
            if key not in p:
                raise RuntimeError(f"预测记录缺少{key}：{path}")
        if len(p["bbox"]) != 4:
            raise RuntimeError(f"bbox格式错误：{p}")
        vals = [float(x) for x in p["bbox"]]
        if not all(math.isfinite(x) for x in vals) or vals[2] < 0 or vals[3] < 0:
            raise RuntimeError(f"bbox数值非法：{p}")
        if not math.isfinite(float(p["score"])):
            raise RuntimeError(f"score数值非法：{p}")
        grouped[image_id].append(p)
    return dict(grouped), len(raw)


def check_uavdt_original_only(
    feature_by_id: dict[int, dict],
    a_by_id: dict[int, list[dict]],
    c2_by_id: dict[int, list[dict]],
) -> dict:
    mismatches = []
    for image_id, row in feature_by_id.items():
        if row["source"] != "uavdt":
            continue
        if prediction_signature(a_by_id.get(image_id, [])) != prediction_signature(
            c2_by_id.get(image_id, [])
        ):
            mismatches.append(image_id)
    if mismatches:
        raise RuntimeError(f"C2在UAVDT上不是原图预测，前20个不一致ID：{mismatches[:20]}")
    return {"checked_images": 1320, "mismatch_images": 0, "passed": True}


# =====================================================================
# 5. Rule learner: train-fold-only quantile threshold search
# =====================================================================
def utility_target(row: dict) -> int:
    """Use Stage 6.3B2's precomputed direction label, not GT values as features."""
    label = row["selector_label"]
    if label == "beneficial":
        return 1
    if label == "harmful":
        return -1
    return 0


def get_thresholds(train_rows: list[dict], feature: str) -> list[float]:
    values = np.asarray([feature_value(r, feature) for r in train_rows], dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return []
    thresholds = np.quantile(values, QUANTILE_GRID)
    # Rounded unique values avoid duplicate candidates when feature values tie.
    return sorted({round(float(x), 8) for x in thresholds})


def condition_mask(rows: list[dict], condition: dict) -> np.ndarray:
    feature = condition["feature"]
    threshold = float(condition["threshold"])
    values = np.asarray([feature_value(r, feature) for r in rows], dtype=np.float64)
    if condition["op"] == ">=":
        return values >= threshold
    if condition["op"] == "<=":
        return values <= threshold
    raise RuntimeError(f"未知规则运算符：{condition['op']}")


def apply_rule(rows: list[dict], rule: dict) -> np.ndarray:
    if rule.get("rule_type") == "none":
        return np.zeros(len(rows), dtype=bool)
    masks = [condition_mask(rows, c) for c in rule["conditions"]]
    if not masks:
        return np.zeros(len(rows), dtype=bool)
    selected = masks[0].copy()
    for mask in masks[1:]:
        selected &= mask
    return selected


def fit_rule(train_rows: list[dict], max_trigger_fraction: float) -> dict:
    """Fit a 1- or 2-condition AND rule using only this fold's training rows."""
    if len(train_rows) == 0:
        raise RuntimeError("规则训练集为空。")

    y = np.asarray([utility_target(r) for r in train_rows], dtype=np.int64)
    n_train = len(train_rows)
    min_selected = max(MIN_TRIGGER_ABSOLUTE, int(math.ceil(MIN_TRIGGER_FRACTION * n_train)))
    max_selected = int(math.floor(max_trigger_fraction * n_train))
    if max_selected < min_selected:
        return {"rule_type": "none", "conditions": [], "reason": "trigger_budget_below_minimum"}

    thresholds_by_feature = {
        feature: get_thresholds(train_rows, feature)
        for feature in FEATURE_COLUMNS
    }

    best = None
    candidate_count = 0

    for n_conditions in range(1, MAX_RULE_CONDITIONS + 1):
        for feature_combo in itertools.combinations(FEATURE_COLUMNS, n_conditions):
            threshold_lists = [thresholds_by_feature[f] for f in feature_combo]
            if any(not x for x in threshold_lists):
                continue
            for threshold_combo in itertools.product(*threshold_lists):
                conditions = [
                    {
                        "feature": feature,
                        "op": FEATURE_SPECS[feature]["op"],
                        "threshold": float(threshold),
                    }
                    for feature, threshold in zip(feature_combo, threshold_combo)
                ]
                mask = apply_rule(train_rows, {"rule_type": "and", "conditions": conditions})
                n_selected = int(mask.sum())
                if n_selected < min_selected or n_selected > max_selected:
                    continue
                candidate_count += 1

                selected_y = y[mask]
                n_beneficial = int(np.sum(selected_y == 1))
                n_harmful = int(np.sum(selected_y == -1))
                n_neutral = int(np.sum(selected_y == 0))
                net_utility = float(np.sum(selected_y) / n_train)
                beneficial_precision = float(n_beneficial / n_selected) if n_selected else 0.0
                total_beneficial = int(np.sum(y == 1))
                beneficial_recall = float(n_beneficial / total_beneficial) if total_beneficial else 0.0

                # Prefer higher net utility, then higher beneficial precision,
                # then simpler rules, then broader coverage when tied.
                ranking = (
                    net_utility,
                    beneficial_precision,
                    -len(conditions),
                    n_selected,
                )
                candidate = {
                    "ranking": ranking,
                    "rule_type": "and",
                    "conditions": conditions,
                    "train_images": n_train,
                    "train_selected": n_selected,
                    "train_trigger_rate": n_selected / n_train,
                    "train_beneficial_selected": n_beneficial,
                    "train_harmful_selected": n_harmful,
                    "train_neutral_selected": n_neutral,
                    "train_beneficial_precision": beneficial_precision,
                    "train_beneficial_recall": beneficial_recall,
                    "train_net_label_utility_per_image": net_utility,
                    "max_trigger_fraction": max_trigger_fraction,
                    "min_selected": min_selected,
                    "max_selected": max_selected,
                }
                if best is None or ranking > best["ranking"]:
                    best = candidate

    if best is None or best["ranking"][0] <= 0:
        return {
            "rule_type": "none",
            "conditions": [],
            "reason": "no_positive_net_utility_rule_within_trigger_budget",
            "train_images": n_train,
            "max_trigger_fraction": max_trigger_fraction,
            "min_selected": min_selected,
            "max_selected": max_selected,
            "candidate_rules_evaluated": candidate_count,
        }

    best.pop("ranking", None)
    best["candidate_rules_evaluated"] = candidate_count
    return best


def evaluate_rule_diagnostics(rows: list[dict], selected: np.ndarray) -> dict:
    labels = [r["selector_label"] for r in rows]
    n_selected = int(selected.sum())
    n_beneficial_total = sum(x == "beneficial" for x in labels)
    n_harmful_total = sum(x == "harmful" for x in labels)
    n_beneficial_selected = sum(bool(selected[i]) and labels[i] == "beneficial" for i in range(len(rows)))
    n_harmful_selected = sum(bool(selected[i]) and labels[i] == "harmful" for i in range(len(rows)))
    n_neutral_selected = sum(bool(selected[i]) and labels[i] == "neutral" for i in range(len(rows)))

    gt_positive = [
        float(rows[i]["delta_small_mean_f1_50_95"])
        for i in range(len(rows))
        if selected[i] and rows[i]["small_gt_count"] > 0
    ]
    gt_zero_fp_delta = [
        float(rows[i]["delta_small_mean_fp_50_95"])
        for i in range(len(rows))
        if selected[i] and rows[i]["small_gt_count"] == 0
    ]

    return {
        "images": len(rows),
        "selected_images": n_selected,
        "trigger_rate": n_selected / len(rows) if rows else 0.0,
        "beneficial_selected": n_beneficial_selected,
        "harmful_selected": n_harmful_selected,
        "neutral_selected": n_neutral_selected,
        "beneficial_precision_among_selected": (
            n_beneficial_selected / n_selected if n_selected else 0.0
        ),
        "beneficial_recall": (
            n_beneficial_selected / n_beneficial_total if n_beneficial_total else 0.0
        ),
        "harmful_capture_rate": (
            n_harmful_selected / n_harmful_total if n_harmful_total else 0.0
        ),
        "net_label_utility": n_beneficial_selected - n_harmful_selected,
        "mean_selected_delta_f1_when_small_gt_exists": (
            float(np.mean(gt_positive)) if gt_positive else None
        ),
        "mean_selected_delta_small_fp_when_no_small_gt": (
            float(np.mean(gt_zero_fp_delta)) if gt_zero_fp_delta else None
        ),
    }


# =====================================================================
# 6. Build GroupKFold out-of-fold assignments
# =====================================================================
def run_groupkfold(feature_by_id: dict[int, dict]) -> tuple[dict, dict, list[dict]]:
    vis_rows = sorted(
        [r for r in feature_by_id.values() if r["source"] == "visdrone"],
        key=lambda r: r["image_id"],
    )
    groups = np.asarray([r["sequence_id"] for r in vis_rows], dtype=object)
    unique_groups = sorted(set(groups.tolist()))
    if len(unique_groups) < N_SPLITS:
        raise RuntimeError(f"视频序列数不足{N_SPLITS}折：{len(unique_groups)}")

    splitter = GroupKFold(n_splits=N_SPLITS)
    selected_by_policy = {name: set() for name in POLICY_BUDGETS}
    fold_rule_output = {name: [] for name in POLICY_BUDGETS}
    assignment_rows = []
    fold_test_counts = Counter()

    X_dummy = np.zeros((len(vis_rows), 1), dtype=np.float32)
    y_dummy = np.zeros(len(vis_rows), dtype=np.int64)

    for fold_id, (train_idx, test_idx) in enumerate(
        splitter.split(X_dummy, y_dummy, groups), start=1
    ):
        train_rows = [vis_rows[int(i)] for i in train_idx]
        test_rows = [vis_rows[int(i)] for i in test_idx]
        train_groups = set(r["sequence_id"] for r in train_rows)
        test_groups = set(r["sequence_id"] for r in test_rows)
        if train_groups & test_groups:
            raise RuntimeError(f"Fold {fold_id}发生视频序列泄漏。")
        fold_test_counts[fold_id] = len(test_rows)

        print(
            f"\n[FOLD {fold_id}/{N_SPLITS}] train={len(train_rows)} images/"
            f"{len(train_groups)} groups; test={len(test_rows)} images/"
            f"{len(test_groups)} groups"
        )

        for policy_name, budget in POLICY_BUDGETS.items():
            rule = fit_rule(train_rows, budget)
            test_selected = apply_rule(test_rows, rule)
            diagnostics = evaluate_rule_diagnostics(test_rows, test_selected)

            for i, row in enumerate(test_rows):
                if bool(test_selected[i]):
                    selected_by_policy[policy_name].add(int(row["image_id"]))
                assignment_rows.append({
                    "policy": policy_name,
                    "fold": fold_id,
                    "image_id": int(row["image_id"]),
                    "file_name": row["file_name"],
                    "sequence_id": row["sequence_id"],
                    "selected_for_c2_2048": int(bool(test_selected[i])),
                    "selector_label_audit_only": row["selector_label"],
                    "n_det_conf025": row["n_det_conf025"],
                    "median_box_area_640": (
                        "" if row["median_box_area_640"] is None else row["median_box_area_640"]
                    ),
                    "tiny_ratio": row["tiny_ratio"],
                    "rule_type": rule.get("rule_type"),
                    "rule_conditions": json.dumps(rule.get("conditions", []), ensure_ascii=False),
                })

            fold_rule_output[policy_name].append({
                "fold": fold_id,
                "train_groups": sorted(train_groups),
                "test_groups": sorted(test_groups),
                "train_images": len(train_rows),
                "test_images": len(test_rows),
                "rule": rule,
                "test_diagnostics": diagnostics,
            })
            print(
                f"  {policy_name:>6}: rule={rule.get('conditions', []) or 'NONE'}, "
                f"test_trigger={diagnostics['selected_images']}/{diagnostics['images']} "
                f"({diagnostics['trigger_rate']:.1%}), "
                f"beneficial/harmful={diagnostics['beneficial_selected']}/"
                f"{diagnostics['harmful_selected']}"
            )

    # Every VisDrone image must receive exactly one OOF decision per policy.
    expected_vis_ids = {int(r["image_id"]) for r in vis_rows}
    for policy_name, selected_ids in selected_by_policy.items():
        if not selected_ids.issubset(expected_vis_ids):
            raise RuntimeError(f"{policy_name}选择器输出了非VisDrone image_id。")
        policy_rows = [r for r in assignment_rows if r["policy"] == policy_name]
        assigned_ids = [int(r["image_id"]) for r in policy_rows]
        if len(assigned_ids) != len(expected_vis_ids) or set(assigned_ids) != expected_vis_ids:
            raise RuntimeError(f"{policy_name}的OOF决策未覆盖全部247张VisDrone图片。")

    group_fold = {}
    for policy_name, folds in fold_rule_output.items():
        for item in folds:
            for group in item["test_groups"]:
                if group in group_fold and group_fold[group] != item["fold"]:
                    raise RuntimeError(f"视频序列被分配到多个测试折：{group}")
                group_fold[group] = item["fold"]
    if set(group_fold) != set(unique_groups):
        raise RuntimeError("GroupKFold没有覆盖全部VisDrone视频序列。")

    return selected_by_policy, fold_rule_output, assignment_rows


# =====================================================================
# 7. Create OOF prediction JSONs
# =====================================================================
def write_oof_prediction(
    policy_name: str,
    selected_vis_ids: set[int],
    feature_by_id: dict[int, dict],
    a_by_id: dict[int, list[dict]],
    c2_by_id: dict[int, list[dict]],
) -> tuple[Path, dict]:
    output_path = SCENE_DIR / f"selector_oof_predictions_{policy_name.lower()}.json"
    output = []
    source_counts = Counter()
    selected_detection_count = 0
    original_detection_count = 0

    for image_id in sorted(feature_by_id):
        row = feature_by_id[image_id]
        use_c2 = row["source"] == "visdrone" and image_id in selected_vis_ids
        records = c2_by_id.get(image_id, []) if use_c2 else a_by_id.get(image_id, [])
        output.extend(records)
        source_counts[row["source"]] += 1
        if use_c2:
            selected_detection_count += len(records)
        else:
            original_detection_count += len(records)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, separators=(",", ":"), allow_nan=False)

    # Verify every UAVDT image still uses the original prediction records.
    for image_id, row in feature_by_id.items():
        if row["source"] == "uavdt":
            if prediction_signature(a_by_id.get(image_id, [])) != prediction_signature(
                c2_by_id.get(image_id, [])
            ):
                raise RuntimeError("UAVDT原图路径检查失败。")

    info = {
        "path": str(output_path.relative_to(ROOT)),
        "sha256": sha256_file(output_path),
        "prediction_count": len(output),
        "selected_visdrone_images": len(selected_vis_ids),
        "visdrone_images": EXPECTED_COUNTS["visdrone"],
        "visdrone_trigger_rate": len(selected_vis_ids) / EXPECTED_COUNTS["visdrone"],
        "uavdt_images_using_original": EXPECTED_COUNTS["uavdt"],
        "detections_from_selected_c2_images": selected_detection_count,
        "detections_from_original_path_images": original_detection_count,
    }
    print(f"\n[{policy_name}] OOF预测文件已写入：{output_path}")
    print(
        f"  触发C2图片：{len(selected_vis_ids)}/{EXPECTED_COUNTS['visdrone']} "
        f"({info['visdrone_trigger_rate']:.1%}); predictions={len(output)}"
    )
    del output
    gc.collect()
    return output_path, info


# =====================================================================
# 8. Official COCO evaluation
# =====================================================================
def evaluate_prediction_file(
    coco_gt,
    category_ids: list[int],
    prediction_path: Path,
    image_ids: list[int],
    scope_name: str,
) -> dict:
    print(f"\nCOCO评价：{prediction_path.name} | {scope_name} | images={len(image_ids)}")
    coco_dt = coco_gt.loadRes(str(prediction_path))
    evaluator = COCOeval_faster(
        coco_gt,
        coco_dt,
        iouType="bbox",
        lvis_style=False,
        print_function=print,
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
    key_map = {
        "AP50-95": "AP_all",
        "AP50": "AP_50",
        "APs": "AP_small",
        "APm": "AP_medium",
        "APl": "AP_large",
    }
    result = {}
    for out_key, in_key in key_map.items():
        if in_key not in stats:
            raise RuntimeError(f"评价器缺少指标{in_key}，实际字段：{list(stats.keys())}")
        result[out_key] = float(stats[in_key])

    del evaluator, coco_dt
    gc.collect()
    return result


def metric_delta(current: dict, baseline: dict) -> dict:
    return {key: float(current[key] - baseline[key]) for key in METRIC_KEYS}


# =====================================================================
# 9. Main
# =====================================================================
def main() -> None:
    print("=" * 78)
    print("Stage 6.3C | 5-fold GroupKFold OOF Scene Selector")
    print("No YOLO training. No model inference. Reuses committed predictions.")
    print("=" * 78)

    check_required_files()
    hash_info = check_hashes_and_reference()
    feature_by_id, data_info = load_feature_and_utility_rows()
    source_by_id = load_and_check_mapping(feature_by_id)
    valid_ids = set(feature_by_id)

    print("\n[1/6] 输入检查通过")
    print("  images:", len(feature_by_id))
    print("  source counts:", data_info["source_counts"])
    print("  VisDrone labels:", data_info["label_counts"])
    print("  sha256:", hash_info["sha256"])

    print("\n[2/6] 读取原图和C2-2048预测")
    a_by_id, a_count = load_prediction_groups(A_JSON, valid_ids)
    c2_by_id, c2_count = load_prediction_groups(C2_JSON, valid_ids)
    print(f"  A_Original detections: {a_count}")
    print(f"  C2_area2048 detections: {c2_count}")
    uavdt_check = check_uavdt_original_only(feature_by_id, a_by_id, c2_by_id)
    print("  UAVDT original-only check:", uavdt_check)

    print("\n[3/6] 执行5折GroupKFold，按sequence_id隔离视频序列")
    selected_by_policy, fold_rules, assignment_rows = run_groupkfold(feature_by_id)

    print("\n[4/6] 生成三套OOF预测文件")
    prediction_paths = {}
    prediction_info = {}
    for policy_name in POLICY_BUDGETS:
        path, info = write_oof_prediction(
            policy_name,
            selected_by_policy[policy_name],
            feature_by_id,
            a_by_id,
            c2_by_id,
        )
        prediction_paths[policy_name] = path
        prediction_info[policy_name] = info

    # Write the OOF decision audit table.
    assignment_fields = [
        "policy", "fold", "image_id", "file_name", "sequence_id",
        "selected_for_c2_2048", "selector_label_audit_only",
        "n_det_conf025", "median_box_area_640", "tiny_ratio",
        "rule_type", "rule_conditions",
    ]
    with OUT_ASSIGNMENTS.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=assignment_fields)
        writer.writeheader()
        writer.writerows(assignment_rows)

    write_json(OUT_RULES, {
        "experiment": "Stage6.3C_5Fold_GroupKFold_OOF_Rule_Selector",
        "protocol": {
            "splitter": "sklearn.model_selection.GroupKFold",
            "n_splits": N_SPLITS,
            "group_key": "sequence_id",
            "feature_columns": FEATURE_COLUMNS,
            "label_is_training_target_only": True,
            "label_is_never_used_as_inference_feature": True,
            "threshold_quantiles": list(QUANTILE_GRID),
            "max_rule_conditions": MAX_RULE_CONDITIONS,
            "rule_logic": "one condition or two conditions combined with AND",
            "feature_directions": FEATURE_SPECS,
            "policy_max_trigger_budgets": POLICY_BUDGETS,
            "minimum_trigger_fraction": MIN_TRIGGER_FRACTION,
            "minimum_trigger_absolute": MIN_TRIGGER_ABSOLUTE,
            "utility": "beneficial=+1, harmful=-1, neutral=0; maximize net label utility under trigger budget",
        },
        "fold_rules": fold_rules,
    })

    print("\n[5/6] 使用固定faster_coco_eval协议评价OOF预测")
    if version("faster-coco-eval") != "1.8.0":
        raise RuntimeError("评价器版本发生变化，终止。")
    coco_gt = COCO(str(GT_JSON))
    gt_image_ids = sorted(int(x["id"]) for x in coco_gt.dataset["images"])
    if gt_image_ids != sorted(valid_ids):
        raise RuntimeError("COCO GT image IDs与特征文件不一致。")
    category_ids = sorted(int(x) for x in coco_gt.getCatIds())
    if not category_ids:
        raise RuntimeError("COCO GT没有类别。")

    full_ids = sorted(valid_ids)
    uavdt_ids = sorted(i for i, s in source_by_id.items() if s == "uavdt")
    visdrone_ids = sorted(i for i, s in source_by_id.items() if s == "visdrone")

    reference = hash_info["reference"]["results"]
    baseline_full = reference["A_Original"]["full"]
    baseline_vis = reference["A_Original"]["visdrone"]
    c2_full = reference["C2_area2048"]["full"]
    c2_vis = reference["C2_area2048"]["visdrone"]
    uavdt_reference = reference["A_Original"]["uavdt"]

    policy_results = {}
    for policy_name in POLICY_BUDGETS:
        pred_path = prediction_paths[policy_name]
        full_metrics = evaluate_prediction_file(coco_gt, category_ids, pred_path, full_ids, "full")
        vis_metrics = evaluate_prediction_file(coco_gt, category_ids, pred_path, visdrone_ids, "visdrone")

        oof_rows = [r for r in assignment_rows if r["policy"] == policy_name]
        oof_rows.sort(key=lambda r: int(r["image_id"]))
        selected_mask = np.asarray([bool(int(r["selected_for_c2_2048"])) for r in oof_rows])
        diagnostic_rows = [feature_by_id[int(r["image_id"])] for r in oof_rows]
        diagnostics = evaluate_rule_diagnostics(diagnostic_rows, selected_mask)

        policy_results[policy_name] = {
            "prediction": prediction_info[policy_name],
            "oof_selection_diagnostics": diagnostics,
            "coco_metrics": {
                "full": full_metrics,
                "visdrone": vis_metrics,
                # UAVDT is unchanged by construction and inherits the validated A reference.
                "uavdt_reference_unchanged": uavdt_reference,
            },
            "delta_vs_A_Original": {
                "full": metric_delta(full_metrics, baseline_full),
                "visdrone": metric_delta(vis_metrics, baseline_vis),
                "uavdt": {key: 0.0 for key in METRIC_KEYS},
            },
            "delta_vs_C2_area2048": {
                "full": metric_delta(full_metrics, c2_full),
                "visdrone": metric_delta(vis_metrics, c2_vis),
            },
        }
        print(
            f"\n[{policy_name}] FULL AP50-95={full_metrics['AP50-95']:.6f}, "
            f"APs={full_metrics['APs']:.6f}; "
            f"VisDrone AP50-95={vis_metrics['AP50-95']:.6f}, "
            f"APs={vis_metrics['APs']:.6f}; "
            f"trigger={diagnostics['selected_images']}/247 "
            f"({diagnostics['trigger_rate']:.1%})"
        )

    print("\n[6/6] 写入汇总文件")
    summary = {
        "experiment": "Stage6.3C_5Fold_GroupKFold_OOF_Rule_Selector",
        "status": "completed",
        "purpose": "Per-image adaptive choice between A_Original and C2_area2048 using out-of-fold scene rules",
        "important_interpretation": (
            "These are internal sequence-grouped OOF estimates on the 247 VisDrone development images. "
            "They are not an independent external test; final deployment/generalization requires a separate holdout or new source data."
        ),
        "protocol": {
            "n_images": EXPECTED_TOTAL,
            "source_counts": data_info["source_counts"],
            "visdrone_label_counts": data_info["label_counts"],
            "splitter": "GroupKFold",
            "n_splits": N_SPLITS,
            "group_key": "sequence_id",
            "feature_columns": FEATURE_COLUMNS,
            "policy_budgets": POLICY_BUDGETS,
            "quantile_grid": list(QUANTILE_GRID),
            "max_rule_conditions": MAX_RULE_CONDITIONS,
            "rule_logic": "AND",
            "target": "Stage6.3B2 selector_label; beneficial=+1, harmful=-1, neutral=0",
            "prediction_policy": "UAVDT always A_Original; VisDrone uses C2_area2048 only when selected by its held-out fold rule",
            "evaluator": "faster_coco_eval",
            "evaluator_version": version("faster-coco-eval"),
            "imgsz": 640,
            "conf": 0.001,
            "nms_iou": 0.7,
            "nms_max_det": 300,
            "iou_thresholds": [float(x) for x in EVAL_IOU_THRESHOLDS],
            "recall_thresholds": [float(x) for x in EVAL_RECALL_THRESHOLDS],
            "max_dets": EVAL_MAX_DETS,
            "area_ranges": EVAL_AREA_RANGES,
            "area_labels": EVAL_AREA_LABELS,
        },
        "input_sha256": hash_info["sha256"],
        "uavdt_original_only_check": uavdt_check,
        "reference_metrics": {
            "A_Original": reference["A_Original"],
            "C2_area2048": reference["C2_area2048"],
        },
        "policies": policy_results,
        "outputs": {
            "assignments_csv": str(OUT_ASSIGNMENTS.relative_to(ROOT)),
            "fold_rules_json": str(OUT_RULES.relative_to(ROOT)),
            "summary_json": str(OUT_SUMMARY.relative_to(ROOT)),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": __import__("sklearn").__version__,
            "faster_coco_eval": version("faster-coco-eval"),
        },
    }
    write_json(OUT_SUMMARY, summary)

    print("\n" + "=" * 78)
    print("Stage 6.3C完成。没有训练YOLO，没有重新运行模型推理。")
    print("Outputs:")
    print(" -", OUT_ASSIGNMENTS)
    print(" -", OUT_RULES)
    print(" -", OUT_SUMMARY)
    for policy_name in POLICY_BUDGETS:
        print(" -", prediction_paths[policy_name])
    print("\n请先检查selector_oof_summary.json，再决定是否进入Stage 6.3D。")
    print("=" * 78)


if __name__ == "__main__":
    main()
