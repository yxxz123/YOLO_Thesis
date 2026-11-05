#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 6.3F-C: finalize M3/M4 scene selectors and benchmark live runtime.

Run from YOLO_Thesis repository root:
    python -u stage6_3f_c_runtime_benchmark.py

This stage does NOT train or fine-tune YOLO. It:
1) fits final lightweight M3/F0 and M4/F1 LogisticRegression selectors on
   the 247-image VisDrone development subset using 4-fold SGKF OOF scores;
2) benchmarks A_Original, C2_area2048, M3 and M4 using the same component-sum
   runtime protocol as Stage 6.3D-B;
3) verifies live original/C2 predictions and live feature parity.

Important:
- The source-specific policy is retained: UAVDT always uses original inference;
  VisDrone is eligible for selective C2_area2048 tiling.
- Runtime results are not accuracy results. The final selectors are trained
  using the VisDrone development labels, so do NOT report COCO accuracy on
  these same 247 images as independent performance.
- The untouched independent test remains reserved for Stage 6.4.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import shutil
import time
from collections import Counter, defaultdict
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
SCENE_DIR = ROOT / "runs/detect/stage6_scene_selector"
FEATURE_CSV = SCENE_DIR / "stage6_3e_b_features/expanded_scene_features.csv"
FEATURE_AUDIT_JSON = SCENE_DIR / "stage6_3e_b_features/feature_audit.json"
UTILITY_CSV = SCENE_DIR / "coco_utility_analysis.csv"
F_B_SUMMARY = SCENE_DIR / "stage6_3f_repeated_sgkf/stage6_3f_b_summary.json"
D_RUNTIME_SUMMARY = SCENE_DIR / "stage6_3d_runtime_summary.json"

ORIGINAL_JSON = ROOT / "runs/detect/stage6_selective_oracle/all_original_predictions.json"
C2_JSON = ROOT / "runs/detect/stage6_size_aware_fusion_iou07/C2_area2048_predictions.json"
GT_JSON = ROOT / "runs/detect/source_validation/Stage5-Baseline-Unified/annotations_auto_detect_val.json"

WEIGHT_DEFAULT = ROOT / "runs/detect/stage2_screening/YOLO11s-DySample/weights/best.pt"
IMAGE_DIR_DEFAULT = ROOT / "datasets/HighwayVehicle-v1.0/images/val"
OUT_DIR = SCENE_DIR / "stage6_3f_c_runtime"

EXPECTED_NUMPY = "1.26.4"
EXPECTED_SKLEARN = "1.5.2"
EXPECTED_WEIGHT_SHA256 = "1a63af9a02c3970a71a2971890b84773851875971620c6eb9336becb090030bb"
EXPECTED_SOURCE_COUNTS = {"uavdt": 1320, "visdrone": 247}
EXPECTED_LABEL_COUNTS = {"beneficial": 158, "harmful": 89}
EXPECTED_TOTAL = 1567
EXPECTED_VIS_GROUPS = 94

F0 = ["n_det_conf025", "median_box_area_640", "tiny_ratio"]
F1 = F0 + [
    "tiny_count",
    "det_density_per_mp",
    "mean_confidence_025",
    "box_area_cv_640",
    "grid_occupancy_4x4",
]
FEATURE_SETS = {"F0": F0, "F1": F1}
MODELS = {
    "M3_Logistic_F0": "F0",
    "M4_Logistic_F1": "F1",
}

# Keep identical to Stage 6.1 / Stage 6.3E-B / Stage 6.3D-B.
IMGSZ = 640
CONF = 0.001
NMS_IOU = 0.7
MAX_DET = 300
OVERLAP = 0.15
HALF = False
AUGMENT = False
FEATURE_CONF = 0.25
FEATURE_IMGSZ = 640
TINY_AREA_THRESHOLD = 32 ** 2
TILE_AREA_THRESHOLD = 2048.0

BUDGET = 0.80
MIN_TRIGGER_FRACTION = 0.10
MIN_TRIGGER_ABSOLUTE = 5
FINAL_CV_SPLITS = 4
FINAL_CV_SEED = 42
MODEL_RANDOM_STATE = 42

LOGISTIC_CONFIGS = [
    {"C": c, "class_weight": cw}
    for c in (0.01, 0.1, 1.0, 10.0)
    for cw in (None, "balanced")
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def check_environment() -> dict[str, str]:
    env = {
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }
    if env["numpy"] != EXPECTED_NUMPY or env["scikit_learn"] != EXPECTED_SKLEARN:
        raise RuntimeError(
            f"环境版本不匹配：actual={env}, "
            f"expected numpy={EXPECTED_NUMPY}, sklearn={EXPECTED_SKLEARN}"
        )
    return env


def normalize_name(value: str) -> str:
    return str(value).replace("\\", "/").lstrip("./")


def sequence_id_from_name(filename: str) -> str:
    stem = Path(str(filename)).stem
    parts = stem.rsplit("_", 1)
    return parts[0] if len(parts) == 2 else stem


def load_training_rows() -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]], dict[str, Any]]:
    feature_raw = read_csv(FEATURE_CSV)
    utility_raw = read_csv(UTILITY_CSV)

    features_by_id: dict[int, dict[str, str]] = {}
    utility_by_id: dict[int, dict[str, str]] = {}

    for row in feature_raw:
        image_id = int(row["image_id"])
        if image_id in features_by_id:
            raise RuntimeError(f"expanded_scene_features.csv重复image_id={image_id}")
        features_by_id[image_id] = row

    for row in utility_raw:
        image_id = int(row["image_id"])
        if image_id in utility_by_id:
            raise RuntimeError(f"coco_utility_analysis.csv重复image_id={image_id}")
        utility_by_id[image_id] = row

    if set(features_by_id) != set(utility_by_id):
        raise RuntimeError("扩展特征CSV与utility CSV的image_id集合不一致")
    if len(features_by_id) != EXPECTED_TOTAL:
        raise RuntimeError(f"特征图片数={len(features_by_id)}，预期={EXPECTED_TOTAL}")

    all_rows: dict[int, dict[str, Any]] = {}
    source_counts: Counter = Counter()

    for image_id, raw in features_by_id.items():
        utility = utility_by_id[image_id]
        for key in ("file_name", "source", "sequence_id", "width", "height"):
            if str(raw[key]).strip() != str(utility[key]).strip():
                raise RuntimeError(f"image_id={image_id}的{key}在两个CSV中不一致")

        item: dict[str, Any] = {
            "image_id": image_id,
            "file_name": raw["file_name"],
            "source": raw["source"].strip().lower(),
            "sequence_id": raw["sequence_id"].strip(),
            "width": int(raw["width"]),
            "height": int(raw["height"]),
            "selector_label": utility["selector_label"].strip().lower(),
        }

        for feature in F1:
            value = raw.get(feature, "")
            if value is None or value.strip() == "":
                item[feature] = None
            else:
                number = float(value)
                if not math.isfinite(number):
                    raise RuntimeError(f"非有限特征值：image_id={image_id}, {feature}={value}")
                item[feature] = number

        if item["source"] not in EXPECTED_SOURCE_COUNTS:
            raise RuntimeError(f"未知source={item['source']}")
        source_counts[item["source"]] += 1
        all_rows[image_id] = item

    if dict(source_counts) != EXPECTED_SOURCE_COUNTS:
        raise RuntimeError(f"数据来源数量错误：{dict(source_counts)}")

    vis_rows = sorted(
        [r for r in all_rows.values() if r["source"] == "visdrone"],
        key=lambda r: r["image_id"],
    )
    label_counts = Counter(r["selector_label"] for r in vis_rows)
    if dict(label_counts) != EXPECTED_LABEL_COUNTS:
        raise RuntimeError(f"VisDrone标签分布异常：{dict(label_counts)}")
    if len({r["sequence_id"] for r in vis_rows}) != EXPECTED_VIS_GROUPS:
        raise RuntimeError("VisDrone sequence_id数量不是94")
    if any(r["selector_label"] not in {"beneficial", "harmful"} for r in vis_rows):
        raise RuntimeError("VisDrone存在未知selector_label")

    data_info = {
        "total_images": len(all_rows),
        "source_counts": dict(source_counts),
        "visdrone_images": len(vis_rows),
        "visdrone_sequences": EXPECTED_VIS_GROUPS,
        "visdrone_labels": dict(label_counts),
    }
    return vis_rows, all_rows, data_info


def matrix_for(rows: list[dict[str, Any]], feature_names: list[str]) -> np.ndarray:
    return np.asarray(
        [
            [np.nan if row.get(name) is None else float(row[name]) for name in feature_names]
            for row in rows
        ],
        dtype=np.float64,
    )


def y_binary(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [1 if r["selector_label"] == "beneficial" else 0 for r in rows],
        dtype=np.int64,
    )


def make_pipeline(config: dict[str, Any]) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=float(config["C"]),
            class_weight=config["class_weight"],
            penalty="l2",
            solver="lbfgs",
            max_iter=5000,
            random_state=MODEL_RANDOM_STATE,
        )),
    ])


def positive_probability(model: Pipeline, X: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(X)
    classes = list(model.named_steps["classifier"].classes_)
    if 1 not in classes:
        raise RuntimeError(f"模型缺少beneficial正类1：classes={classes}")
    return probabilities[:, classes.index(1)].astype(np.float64)


def selection_diagnostics(rows: list[dict[str, Any]], selected: np.ndarray) -> dict[str, Any]:
    labels = [r["selector_label"] for r in rows]
    n_selected = int(selected.sum())
    beneficial_total = sum(x == "beneficial" for x in labels)
    harmful_total = sum(x == "harmful" for x in labels)
    beneficial_selected = sum(
        bool(selected[i]) and labels[i] == "beneficial" for i in range(len(rows))
    )
    harmful_selected = sum(
        bool(selected[i]) and labels[i] == "harmful" for i in range(len(rows))
    )
    return {
        "images": len(rows),
        "selected_images": n_selected,
        "trigger_rate": n_selected / len(rows),
        "beneficial_total": beneficial_total,
        "harmful_total": harmful_total,
        "beneficial_selected": beneficial_selected,
        "harmful_selected": harmful_selected,
        "beneficial_precision": beneficial_selected / n_selected if n_selected else 0.0,
        "beneficial_recall": beneficial_selected / beneficial_total if beneficial_total else 0.0,
        "harmful_capture_rate": harmful_selected / harmful_total if harmful_total else 0.0,
        "net_label_utility": beneficial_selected - harmful_selected,
        "net_label_utility_per_image": (
            (beneficial_selected - harmful_selected) / len(rows)
        ),
    }


def threshold_from_oof(
    scores: np.ndarray,
    y: np.ndarray,
) -> tuple[float | None, np.ndarray, dict[str, Any]]:
    if len(scores) != len(y) or not np.all(np.isfinite(scores)):
        raise RuntimeError("OOF概率缺失或包含NaN/Inf")

    n = len(y)
    min_selected = max(
        MIN_TRIGGER_ABSOLUTE,
        int(math.ceil(MIN_TRIGGER_FRACTION * n)),
    )
    max_selected = int(math.floor(BUDGET * n))
    best = None

    for threshold in sorted(set(float(x) for x in scores), reverse=True):
        selected = scores >= threshold
        count = int(selected.sum())
        if count < min_selected or count > max_selected:
            continue

        beneficial = int(np.sum(y[selected] == 1))
        harmful = int(np.sum(y[selected] == 0))
        precision = beneficial / count if count else 0.0
        net = float((beneficial - harmful) / n)
        ranking = (net, precision, -count)

        if best is None or ranking > best["ranking"]:
            best = {
                "ranking": ranking,
                "threshold": float(threshold),
                "selected": selected,
                "selected_count": count,
                "beneficial_selected": beneficial,
                "harmful_selected": harmful,
                "beneficial_precision": precision,
                "net_label_utility_per_image": net,
            }

    if best is None or best["ranking"][0] <= 0:
        return None, np.zeros(n, dtype=bool), {
            "selected_count": 0,
            "trigger_rate": 0.0,
            "beneficial_selected": 0,
            "harmful_selected": 0,
            "beneficial_precision": 0.0,
            "net_label_utility_per_image": 0.0,
            "reason": "no_positive_net_utility_threshold",
        }

    selected = best["selected"]
    details = {
        k: v for k, v in best.items()
        if k not in {"ranking", "selected"}
    }
    details["trigger_rate"] = details["selected_count"] / n
    return best["threshold"], selected, details


def fit_final_selector(
    model_name: str,
    feature_set: str,
    vis_rows: list[dict[str, Any]],
    input_hashes: dict[str, str],
    output_dir: Path,
) -> tuple[Pipeline, dict[str, Any]]:
    feature_names = FEATURE_SETS[feature_set]
    X = matrix_for(vis_rows, feature_names)
    y = y_binary(vis_rows)
    groups = np.asarray([r["sequence_id"] for r in vis_rows], dtype=object)
    dummy = np.zeros((len(vis_rows), 1), dtype=np.float32)

    splitter = StratifiedGroupKFold(
        n_splits=FINAL_CV_SPLITS,
        shuffle=True,
        random_state=FINAL_CV_SEED,
    )
    splits = list(splitter.split(dummy, y, groups))
    if len(splits) != FINAL_CV_SPLITS:
        raise RuntimeError("最终选择器OOF折数错误")

    split_audit = []
    for fold, (train_idx, val_idx) in enumerate(splits, start=1):
        train_groups = set(groups[train_idx])
        val_groups = set(groups[val_idx])
        if train_groups & val_groups:
            raise RuntimeError(f"{model_name}最终拟合CV出现sequence泄漏")
        if len(np.unique(y[train_idx])) != 2 or len(np.unique(y[val_idx])) != 2:
            raise RuntimeError(f"{model_name}最终拟合CV某折缺少类别")
        split_audit.append({
            "fold": fold,
            "train_images": int(len(train_idx)),
            "validation_images": int(len(val_idx)),
            "train_groups": len(train_groups),
            "validation_groups": len(val_groups),
            "train_beneficial": int(np.sum(y[train_idx] == 1)),
            "train_harmful": int(np.sum(y[train_idx] == 0)),
            "validation_beneficial": int(np.sum(y[val_idx] == 1)),
            "validation_harmful": int(np.sum(y[val_idx] == 0)),
            "group_overlap": 0,
        })

    best = None
    config_results = []

    for config in LOGISTIC_CONFIGS:
        oof_scores = np.full(len(vis_rows), np.nan, dtype=np.float64)

        for train_idx, val_idx in splits:
            model = make_pipeline(config)
            model.fit(X[train_idx], y[train_idx])
            oof_scores[val_idx] = positive_probability(model, X[val_idx])

        if not np.all(np.isfinite(oof_scores)):
            raise RuntimeError(f"{model_name} OOF概率未完整覆盖247张图片")

        threshold, oof_selected, threshold_info = threshold_from_oof(oof_scores, y)
        diagnostics = selection_diagnostics(vis_rows, oof_selected)
        ranking = (
            diagnostics["net_label_utility_per_image"],
            diagnostics["beneficial_precision"],
            -diagnostics["selected_images"],
        )
        record = {
            "config": config,
            "threshold": threshold,
            "threshold_selection": threshold_info,
            "oof_diagnostics": diagnostics,
            "ranking": ranking,
        }
        config_results.append(record)
        if best is None or ranking > best["ranking"]:
            best = record

    if best is None or best["threshold"] is None:
        raise RuntimeError(f"{model_name}没有找到正向净效用阈值")

    final_model = make_pipeline(best["config"])
    final_model.fit(X, y)

    model_path = output_dir / f"final_selector_{model_name.lower()}.joblib"
    joblib.dump(final_model, model_path)

    metadata = {
        "model": model_name,
        "feature_set": feature_set,
        "feature_names": feature_names,
        "training_images": len(vis_rows),
        "training_sequences": len(set(groups)),
        "label_counts": {
            "beneficial": int(np.sum(y == 1)),
            "harmful": int(np.sum(y == 0)),
        },
        "final_config": best["config"],
        "probability_threshold": float(best["threshold"]),
        "budget": BUDGET,
        "minimum_trigger_fraction": MIN_TRIGGER_FRACTION,
        "minimum_trigger_absolute": MIN_TRIGGER_ABSOLUTE,
        "finalization_cv": {
            "splitter": "StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)",
            "group_key": "sequence_id",
            "split_audit": split_audit,
            "objective": (
                "maximize (beneficial_selected - harmful_selected)/n on OOF scores; "
                "tie-break by beneficial precision, then fewer selected images"
            ),
        },
        "oof_diagnostics_for_finalization_only": best["oof_diagnostics"],
        "tested_config_count": len(config_results),
        "all_config_oof_results": [
            {
                "config": x["config"],
                "threshold": x["threshold"],
                "oof_diagnostics": x["oof_diagnostics"],
                "ranking": list(x["ranking"]),
            }
            for x in config_results
        ],
        "input_sha256": input_hashes,
        "model_file": str(model_path.relative_to(ROOT)),
        "model_sha256": sha256_file(model_path),
        "interpretation_limit": (
            "The 247 VisDrone development labels are used to finalize this selector. "
            "OOF diagnostics are for threshold/configuration selection only; they are "
            "not an independent test result. Do not report accuracy on these same images."
        ),
    }
    return final_model, metadata


def load_or_fit_selectors(
    vis_rows: list[dict[str, Any]],
    input_hashes: dict[str, str],
    output_dir: Path,
    resume: bool,
) -> tuple[dict[str, Pipeline], dict[str, Any]]:
    metadata_path = output_dir / "final_selector_metadata.json"
    model_paths = {
        name: output_dir / f"final_selector_{name.lower()}.joblib"
        for name in MODELS
    }

    if resume and metadata_path.is_file() and all(p.is_file() for p in model_paths.values()):
        metadata = load_json(metadata_path)
        if metadata.get("input_sha256") != input_hashes:
            raise RuntimeError("已保存的最终选择器与当前输入SHA256不一致")
        models = {}
        for name, path in model_paths.items():
            expected = metadata["models"][name]["model_sha256"]
            if sha256_file(path) != expected:
                raise RuntimeError(f"选择器模型文件SHA256不一致：{path}")
            models[name] = joblib.load(path)
        print("[RESUME] Loaded frozen final selectors from output directory.", flush=True)
        return models, metadata

    models: dict[str, Pipeline] = {}
    metadata = {
        "experiment": "Stage6.3F_C_Final_Selector_Fit",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "completed",
        "models": {},
        "input_sha256": input_hashes,
    }

    for model_name, feature_set in MODELS.items():
        print(f"[FINALIZE] {model_name} using {feature_set}...", flush=True)
        model, model_meta = fit_final_selector(
            model_name, feature_set, vis_rows, input_hashes, output_dir
        )
        models[model_name] = model
        metadata["models"][model_name] = model_meta
        print(
            f"  config={model_meta['final_config']} "
            f"threshold={model_meta['probability_threshold']:.6f} "
            f"OOF_trigger={model_meta['oof_diagnostics_for_finalization_only']['trigger_rate']:.3f}",
            flush=True,
        )

    write_json_atomic(metadata_path, metadata)
    return models, metadata


def read_prediction_groups(path: Path, valid_ids: set[int]) -> dict[int, list[dict[str, Any]]]:
    raw = load_json(path)
    if not isinstance(raw, list):
        raise RuntimeError(f"预测文件不是COCO detection list：{path}")
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, pred in enumerate(raw):
        image_id = int(pred["image_id"])
        if image_id not in valid_ids:
            raise RuntimeError(f"{path.name}包含未知image_id={image_id}")
        if len(pred.get("bbox", [])) != 4:
            raise RuntimeError(f"{path.name}第{index}条bbox格式错误")
        grouped[image_id].append(pred)
    return dict(grouped)


def prediction_signature(records: list[dict[str, Any]]) -> Counter:
    return Counter(
        (
            int(p["category_id"]),
            round(float(p["score"]), 7),
            tuple(round(float(v), 5) for v in p["bbox"]),
        )
        for p in records
    )


def check_uavdt_original_only(
    all_rows: dict[int, dict[str, Any]],
    original: dict[int, list[dict[str, Any]]],
    c2: dict[int, list[dict[str, Any]]],
) -> None:
    bad = []
    for image_id, row in all_rows.items():
        if row["source"] == "uavdt":
            if prediction_signature(original.get(image_id, [])) != prediction_signature(c2.get(image_id, [])):
                bad.append(image_id)
    if bad:
        raise RuntimeError(f"C2与Original在UAVDT上不一致，前20个ID：{bad[:20]}")


def build_image_index(image_dir: Path):
    files = [
        p for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    ]
    if not files:
        raise RuntimeError(f"验证图片目录为空：{image_dir}")
    by_relative: dict[str, Path] = {}
    by_basename: dict[str, list[Path]] = defaultdict(list)
    for path in files:
        by_relative[normalize_name(path.relative_to(image_dir))] = path
        by_basename[path.name].append(path)
    return by_relative, by_basename


def resolve_image_path(
    filename: str,
    image_dir: Path,
    by_relative: dict[str, Path],
    by_basename: dict[str, list[Path]],
) -> Path:
    raw = Path(str(filename))
    if raw.is_absolute() and raw.is_file():
        return raw
    direct = image_dir / raw
    if direct.is_file():
        return direct
    key = normalize_name(filename)
    if key in by_relative:
        return by_relative[key]
    matches = by_basename.get(Path(key).name, [])
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(f"图片文件名重复，无法唯一定位：{filename}")
    raise FileNotFoundError(f"找不到图片：{filename}")


def empty_result():
    return (
        np.zeros((0, 4), dtype=np.float32),
        np.zeros((0,), dtype=np.float32),
        np.zeros((0,), dtype=np.int64),
    )


def extract_result(result: Any, offset_x: int = 0, offset_y: int = 0):
    if result.boxes is None or len(result.boxes) == 0:
        return empty_result()
    boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    scores = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
    classes = result.boxes.cls.detach().cpu().numpy().astype(np.int64)
    boxes[:, [0, 2]] += offset_x
    boxes[:, [1, 3]] += offset_y
    return boxes, scores, classes


def merge_nms(boxes, scores, classes, width, height, torch, batched_nms):
    if len(boxes) == 0:
        return empty_result()

    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    classes = np.asarray(classes, dtype=np.int64).reshape(-1)

    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)
    valid = (
        (boxes[:, 2] > boxes[:, 0])
        & (boxes[:, 3] > boxes[:, 1])
        & np.isfinite(scores)
    )
    boxes, scores, classes = boxes[valid], scores[valid], classes[valid]
    if len(boxes) == 0:
        return empty_result()

    keep = batched_nms(
        torch.from_numpy(boxes),
        torch.from_numpy(scores),
        torch.from_numpy(classes),
        NMS_IOU,
    )[:MAX_DET].cpu().numpy()

    return boxes[keep], scores[keep], classes[keep]


def make_tiles(image: np.ndarray):
    height, width = image.shape[:2]
    tile_w = min(math.ceil(width / (2 - OVERLAP)), width)
    tile_h = min(math.ceil(height / (2 - OVERLAP)), height)
    xs = sorted(set([0, max(0, width - tile_w)]))
    ys = sorted(set([0, max(0, height - tile_h)]))
    tiles, offsets = [], []
    for y in ys:
        for x in xs:
            tiles.append(image[y:y + tile_h, x:x + tile_w].copy())
            offsets.append((x, y))
    return tiles, offsets


def prediction_args(device: str) -> dict[str, Any]:
    return {
        "imgsz": IMGSZ,
        "conf": CONF,
        "iou": NMS_IOU,
        "max_det": MAX_DET,
        "device": device,
        "half": HALF,
        "rect": False,
        "augment": AUGMENT,
        "agnostic_nms": False,
        "verbose": False,
        "save": False,
    }


def synchronize(torch, device: str) -> None:
    if torch.cuda.is_available() and str(device).lower() not in {"cpu", "-1"}:
        torch.cuda.synchronize()


def predict_original(model, image, pred_args, torch, batched_nms):
    results = model.predict(source=image, batch=1, **pred_args)
    boxes, scores, classes = extract_result(results[0])
    height, width = image.shape[:2]
    return merge_nms(boxes, scores, classes, width, height, torch, batched_nms)


def predict_tiled(model, image, pred_args, torch, batched_nms):
    tiles, offsets = make_tiles(image)
    results = model.predict(source=tiles, batch=len(tiles), **pred_args)
    if len(results) != len(tiles):
        raise RuntimeError(f"切片数{len(tiles)}与推理结果数{len(results)}不一致")

    all_boxes, all_scores, all_classes = [], [], []
    for result, (offset_x, offset_y) in zip(results, offsets):
        boxes, scores, classes = extract_result(result, offset_x, offset_y)
        if len(boxes):
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_classes.append(classes)

    if not all_boxes:
        return empty_result(), len(tiles)

    height, width = image.shape[:2]
    merged = merge_nms(
        np.concatenate(all_boxes),
        np.concatenate(all_scores),
        np.concatenate(all_classes),
        width,
        height,
        torch,
        batched_nms,
    )
    return merged, len(tiles)


def fuse_c2_area2048(original, tiled, width, height, torch, batched_nms):
    b0, s0, c0 = original
    b1, s1, c1 = tiled

    if len(b1):
        tile_areas = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
        keep = tile_areas < TILE_AREA_THRESHOLD
        b1, s1, c1 = b1[keep], s1[keep], c1[keep]

    if len(b1) == 0:
        return original

    if len(b0) == 0:
        boxes, scores, classes = b1, s1, c1
    else:
        boxes = np.concatenate([b0, b1])
        scores = np.concatenate([s0, s1])
        classes = np.concatenate([c0, c1])

    return merge_nms(boxes, scores, classes, width, height, torch, batched_nms)


def compute_scene_features(
    result: tuple[np.ndarray, np.ndarray, np.ndarray],
    width: int,
    height: int,
    feature_set: str,
) -> dict[str, Any]:
    boxes, scores, _classes = result
    keep = np.asarray(scores, dtype=np.float32) >= FEATURE_CONF
    boxes = np.asarray(boxes, dtype=np.float32)[keep]
    scores = np.asarray(scores, dtype=np.float32)[keep]
    n_det = int(len(boxes))

    if n_det == 0:
        base = {
            "n_det_conf025": 0,
            "median_box_area_640": None,
            "tiny_ratio": 0.0,
        }
        if feature_set == "F1":
            base.update({
                "tiny_count": 0,
                "det_density_per_mp": 0.0,
                "mean_confidence_025": 0.0,
                "box_area_cv_640": 0.0,
                "grid_occupancy_4x4": 0.0,
            })
        return base

    scale = min(FEATURE_IMGSZ / float(width), FEATURE_IMGSZ / float(height))
    box_widths = boxes[:, 2] - boxes[:, 0]
    box_heights = boxes[:, 3] - boxes[:, 1]
    if np.any(box_widths <= 0) or np.any(box_heights <= 0):
        raise RuntimeError("原图预测框存在非正宽高")
    areas = (box_widths * scale) * (box_heights * scale)
    if not np.all(np.isfinite(areas)) or np.any(areas <= 0):
        raise RuntimeError("场景特征面积出现非法值")

    tiny_count = int(np.sum(areas < TINY_AREA_THRESHOLD))
    base = {
        "n_det_conf025": n_det,
        "median_box_area_640": round(float(np.median(areas)), 4),
        "tiny_ratio": round(float(tiny_count / n_det), 6),
    }

    if feature_set == "F0":
        return base

    mean_area = float(np.mean(areas))
    std_area = float(np.std(areas, ddof=0))
    megapixels = (width * height) / 1_000_000.0

    centers_x = (boxes[:, 0] + boxes[:, 2]) / (2.0 * float(width))
    centers_y = (boxes[:, 1] + boxes[:, 3]) / (2.0 * float(height))
    grid_cells = set()
    for cx, cy in zip(centers_x, centers_y):
        gx = min(3, max(0, int(math.floor(float(cx) * 4.0))))
        gy = min(3, max(0, int(math.floor(float(cy) * 4.0))))
        grid_cells.add((gx, gy))

    base.update({
        "tiny_count": tiny_count,
        "det_density_per_mp": round(float(n_det / megapixels), 8),
        "mean_confidence_025": round(float(np.mean(scores)), 8),
        "box_area_cv_640": round(float(std_area / mean_area), 8) if mean_area > 0 else 0.0,
        "grid_occupancy_4x4": round(float(len(grid_cells) / 16.0), 8),
    })
    return base


def verify_feature_parity(
    image_id: int,
    live: dict[str, Any],
    saved: dict[str, Any],
    feature_names: list[str],
) -> None:
    for name in feature_names:
        live_value = live.get(name)
        saved_value = saved.get(name)
        if live_value is None or saved_value is None:
            if live_value is not None or saved_value is not None:
                raise RuntimeError(
                    f"特征空值状态不一致：image_id={image_id}, {name}, "
                    f"live={live_value}, saved={saved_value}"
                )
            continue
        tolerance = 1e-3 if name == "median_box_area_640" else 1e-6
        if not math.isclose(
            float(live_value), float(saved_value),
            rel_tol=0.0, abs_tol=tolerance
        ):
            raise RuntimeError(
                f"实时特征与训练特征不一致：image_id={image_id}, {name}, "
                f"live={live_value}, saved={saved_value}"
            )


def predict_selector(
    model: Pipeline,
    feature_row: dict[str, Any],
    feature_names: list[str],
    threshold: float,
) -> tuple[float, bool]:
    X = np.asarray(
        [[np.nan if feature_row.get(name) is None else float(feature_row[name])
          for name in feature_names]],
        dtype=np.float64,
    )
    score = float(positive_probability(model, X)[0])
    return score, bool(score >= threshold)


def prediction_to_coco(image_id: int, result, category_id: int) -> list[dict[str, Any]]:
    boxes, scores, classes = result
    output = []
    for box, score, cls in zip(boxes, scores, classes):
        if int(cls) != 0:
            raise RuntimeError(f"单类别模型出现非0类别：image_id={image_id}, cls={cls}")
        x1, y1, x2, y2 = map(float, box)
        if x2 <= x1 or y2 <= y1:
            continue
        output.append({
            "image_id": int(image_id),
            "category_id": int(category_id),
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "score": float(score),
        })
    return output


def check_prediction_identity(
    image_id: int,
    live_result,
    stored_grouped: dict[int, list[dict[str, Any]]],
    category_id: int,
    name: str,
) -> None:
    live = prediction_to_coco(image_id, live_result, category_id)
    expected = stored_grouped.get(image_id, [])
    if prediction_signature(live) != prediction_signature(expected):
        raise RuntimeError(
            f"{name}预测复现失败：image_id={image_id}。"
            "检查权重、推理参数、Ultralytics版本、切片坐标和NMS。"
        )


def validate_inputs(weights: Path) -> tuple[
    list[dict[str, Any]], dict[int, dict[str, Any]], dict[str, Any],
    dict[int, dict[str, Any]], int, dict[str, str]
]:
    required = [
        FEATURE_CSV, FEATURE_AUDIT_JSON, UTILITY_CSV, F_B_SUMMARY,
        D_RUNTIME_SUMMARY, ORIGINAL_JSON, C2_JSON, GT_JSON, weights,
    ]
    missing = [p for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("缺少必要输入文件：\n" + "\n".join(map(str, missing)))

    feature_audit = load_json(FEATURE_AUDIT_JSON)
    f_b = load_json(F_B_SUMMARY)
    d_runtime = load_json(D_RUNTIME_SUMMARY)

    if feature_audit.get("status") != "completed":
        raise RuntimeError("Stage 6.3E-B特征审计状态不是completed")
    if f_b.get("status") != "completed":
        raise RuntimeError("Stage 6.3F-B summary不是completed")
    if d_runtime.get("status") != "completed":
        raise RuntimeError("Stage 6.3D-B runtime summary不是completed")

    current_weight_hash = sha256_file(weights)
    expected_weight_hash = d_runtime.get("weights_sha256")
    if expected_weight_hash and current_weight_hash != expected_weight_hash:
        raise RuntimeError(
            "当前YOLO权重与Stage 6.3D-B运行时测试权重不一致："
            f"expected={expected_weight_hash}, actual={current_weight_hash}"
        )
    if current_weight_hash != EXPECTED_WEIGHT_SHA256:
        raise RuntimeError(
            "当前YOLO权重不是已验证的YOLO11s-DySample best.pt："
            f"{current_weight_hash}"
        )

    input_paths = {
        "expanded_scene_features.csv": FEATURE_CSV,
        "feature_audit.json": FEATURE_AUDIT_JSON,
        "utility_csv": UTILITY_CSV,
        "original_predictions": ORIGINAL_JSON,
        "c2_predictions": C2_JSON,
        "ground_truth": GT_JSON,
        "stage6_3f_b_summary": F_B_SUMMARY,
        "stage6_3d_runtime_summary": D_RUNTIME_SUMMARY,
        "yolo_weights": weights,
    }
    input_hashes = {name: sha256_file(path) for name, path in input_paths.items()}

    f_b_hashes = f_b.get("input_sha256", {})
    for key in (
        "expanded_scene_features.csv",
        "feature_audit.json",
        "utility_csv",
        "original_predictions",
        "c2_predictions",
        "ground_truth",
    ):
        expected = f_b_hashes.get(key)
        actual = input_hashes[key]
        if expected and expected != actual:
            raise RuntimeError(f"Stage 6.3F-B输入SHA256不一致：{key}")

    d_hashes = d_runtime.get("input_sha256", {})
    for d_key, current_key in (
        ("original", "original_predictions"),
        ("c2", "c2_predictions"),
        ("gt", "ground_truth"),
    ):
        expected = d_hashes.get(d_key)
        if expected and expected != input_hashes[current_key]:
            raise RuntimeError(f"Stage 6.3D-B输入SHA256不一致：{d_key}")

    vis_rows, all_rows, data_info = load_training_rows()

    gt = load_json(GT_JSON)
    gt_images = {int(x["id"]): x for x in gt["images"]}
    if set(gt_images) != set(all_rows):
        raise RuntimeError("GT image_id集合与扩展特征CSV不一致")
    categories = gt.get("categories", [])
    if len(categories) != 1:
        raise RuntimeError(f"GT类别数应为1，实际={len(categories)}")
    category_id = int(categories[0]["id"])

    for image_id, row in all_rows.items():
        gt_row = gt_images[image_id]
        if normalize_name(row["file_name"]) != normalize_name(gt_row["file_name"]):
            raise RuntimeError(f"image_id={image_id} file_name与GT不一致")
        if row["width"] != int(gt_row["width"]) or row["height"] != int(gt_row["height"]):
            raise RuntimeError(f"image_id={image_id}尺寸与GT不一致")
        if row["sequence_id"] != sequence_id_from_name(gt_row["file_name"]):
            raise RuntimeError(f"image_id={image_id} sequence_id与文件名解析不一致")

    return vis_rows, all_rows, data_info, gt_images, category_id, input_hashes


def run_one_repeat(
    repeat: int,
    records: list[dict[str, Any]],
    vis_ids: set[int],
    vis_rows_by_id: dict[int, dict[str, Any]],
    models: dict[str, Pipeline],
    selector_metadata: dict[str, Any],
    original_grouped: dict[int, list[dict[str, Any]]],
    c2_grouped: dict[int, list[dict[str, Any]]],
    category_id: int,
    model,
    cv2,
    torch,
    batched_nms,
    device: str,
    verify_samples: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pred_args = prediction_args(device)
    uavdt_records = [r for r in records if r["source"] == "uavdt"]
    vis_records = [r for r in records if r["source"] == "visdrone"]

    verify_a_ids = set(
        r["image_id"] for r in uavdt_records[:verify_samples]
    ) | set(
        r["image_id"] for r in vis_records[:verify_samples]
    )
    verify_c2_ids = set(r["image_id"] for r in vis_records[:verify_samples])

    read_t: dict[int, float] = {}
    original_t: dict[int, float] = {}
    feature_t = {name: {} for name in MODELS}
    selector_t = {name: {} for name in MODELS}
    selector_scores = {name: {} for name in MODELS}
    selected_ids = {name: set() for name in MODELS}
    tile_t: dict[int, float] = {}
    fusion_t: dict[int, float] = {}

    verified_a: set[int] = set()
    verified_c2: set[int] = set()
    feature_checks = {name: 0 for name in MODELS}

    pass_start = time.perf_counter()

    for index, rec in enumerate(records, start=1):
        image_id = rec["image_id"]

        t0 = time.perf_counter()
        image = cv2.imread(str(rec["image_path"]))
        read_t[image_id] = time.perf_counter() - t0
        if image is None:
            raise RuntimeError(f"图片读取失败：{rec['image_path']}")

        height, width = image.shape[:2]
        if (width, height) != (rec["width"], rec["height"]):
            raise RuntimeError(
                f"图片尺寸与GT不一致：{rec['file_name']} "
                f"actual={width}x{height}, expected={rec['width']}x{rec['height']}"
            )

        synchronize(torch, device)
        t0 = time.perf_counter()
        original = predict_original(model, image, pred_args, torch, batched_nms)
        synchronize(torch, device)
        original_t[image_id] = time.perf_counter() - t0

        if image_id in verify_a_ids and image_id not in verified_a:
            check_prediction_identity(
                image_id, original, original_grouped, category_id, "A_Original"
            )
            verified_a.add(image_id)

        if rec["source"] == "visdrone":
            saved_features = vis_rows_by_id[image_id]

            for model_name, feature_set in MODELS.items():
                feature_names = FEATURE_SETS[feature_set]

                t0 = time.perf_counter()
                live_features = compute_scene_features(
                    original, width, height, feature_set
                )
                feature_t[model_name][image_id] = time.perf_counter() - t0

                verify_feature_parity(
                    image_id, live_features, saved_features, feature_names
                )
                feature_checks[model_name] += 1

                threshold = float(
                    selector_metadata["models"][model_name]["probability_threshold"]
                )
                t0 = time.perf_counter()
                score, selected = predict_selector(
                    models[model_name], live_features, feature_names, threshold
                )
                selector_t[model_name][image_id] = time.perf_counter() - t0
                selector_scores[model_name][image_id] = score
                if selected:
                    selected_ids[model_name].add(image_id)

            # Tile cost is measured for every VisDrone image once; policy totals
            # include it only for the images selected by that policy.
            synchronize(torch, device)
            t0 = time.perf_counter()
            tiled, tile_count = predict_tiled(
                model, image, pred_args, torch, batched_nms
            )
            synchronize(torch, device)
            tile_t[image_id] = time.perf_counter() - t0
            if tile_count != 4:
                raise RuntimeError(f"image_id={image_id}应生成4个切片，实际={tile_count}")

            t0 = time.perf_counter()
            c2_result = fuse_c2_area2048(
                original, tiled, width, height, torch, batched_nms
            )
            fusion_t[image_id] = time.perf_counter() - t0

            if image_id in verify_c2_ids and image_id not in verified_c2:
                check_prediction_identity(
                    image_id, c2_result, c2_grouped, category_id, "C2_area2048"
                )
                verified_c2.add(image_id)

        if index % 100 == 0 or index == len(records):
            print(
                f"[repeat {repeat}] processed {index}/{len(records)} | "
                f"M3 selected={len(selected_ids['M3_Logistic_F0'])} | "
                f"M4 selected={len(selected_ids['M4_Logistic_F1'])}",
                flush=True,
            )

    synchronize(torch, device)
    measurement_pass_wall = time.perf_counter() - pass_start

    if len(verified_a) != len(verify_a_ids):
        raise RuntimeError(f"A_Original预测复核不完整：{len(verified_a)}/{len(verify_a_ids)}")
    if len(verified_c2) != len(verify_c2_ids):
        raise RuntimeError(f"C2_area2048预测复核不完整：{len(verified_c2)}/{len(verify_c2_ids)}")
    for model_name in MODELS:
        if feature_checks[model_name] != len(vis_ids):
            raise RuntimeError(
                f"{model_name}特征复核不完整：{feature_checks[model_name]}/{len(vis_ids)}"
            )

    policies = {
        "A_Original": {
            "model_name": None,
            "selected": set(),
        },
        "C2_area2048": {
            "model_name": None,
            "selected": set(vis_ids),
        },
        "M3_Logistic_F0": {
            "model_name": "M3_Logistic_F0",
            "selected": selected_ids["M3_Logistic_F0"],
        },
        "M4_Logistic_F1": {
            "model_name": "M4_Logistic_F1",
            "selected": selected_ids["M4_Logistic_F1"],
        },
    }

    repeat_metric_rows = []
    per_image_rows = []

    for policy_name, policy_info in policies.items():
        model_name = policy_info["model_name"]
        chosen = policy_info["selected"]

        total_read = float(sum(read_t.values()))
        total_original = float(sum(original_t.values()))
        total_feature = (
            float(sum(feature_t[model_name].values())) if model_name else 0.0
        )
        total_selector = (
            float(sum(selector_t[model_name].values())) if model_name else 0.0
        )
        total_tile = float(sum(tile_t[i] for i in chosen if i in tile_t))
        total_fusion = float(sum(fusion_t[i] for i in chosen if i in fusion_t))

        latencies = []
        for rec in records:
            image_id = rec["image_id"]
            is_visdrone = rec["source"] == "visdrone"
            selected = image_id in chosen

            latency = read_t[image_id] + original_t[image_id]
            feature_ms = selector_ms = tile_ms = fusion_ms = score = 0.0

            if model_name and is_visdrone:
                feature_ms = feature_t[model_name][image_id] * 1000.0
                selector_ms = selector_t[model_name][image_id] * 1000.0
                score = selector_scores[model_name][image_id]
                latency += (
                    feature_t[model_name][image_id]
                    + selector_t[model_name][image_id]
                )

            if selected:
                tile_ms = tile_t[image_id] * 1000.0
                fusion_ms = fusion_t[image_id] * 1000.0
                latency += tile_t[image_id] + fusion_t[image_id]

            total_image_ms = latency * 1000.0
            latencies.append(latency)

            per_image_rows.append({
                "repeat": repeat,
                "policy": policy_name,
                "image_id": image_id,
                "source": rec["source"],
                "sequence_id": rec["sequence_id"],
                "selected_for_tiling": int(selected),
                "selector_score": score if model_name and is_visdrone else "",
                "read_ms": read_t[image_id] * 1000.0,
                "original_ms": original_t[image_id] * 1000.0,
                "feature_ms": feature_ms,
                "selector_ms": selector_ms,
                "tile_ms": tile_ms,
                "fusion_ms": fusion_ms,
                "total_image_ms": total_image_ms,
            })

        component_sum = (
            total_read + total_original + total_feature
            + total_selector + total_tile + total_fusion
        )
        if not math.isclose(
            component_sum, float(sum(latencies)),
            rel_tol=1e-8, abs_tol=1e-6
        ):
            raise RuntimeError(f"{policy_name}组件耗时与逐图耗时之和不一致")

        selected_count = len(chosen)
        repeat_metric_rows.append({
            "repeat": repeat,
            "policy": policy_name,
            "images": len(records),
            "selected_visdrone_images": selected_count,
            "trigger_rate_visdrone": selected_count / len(vis_ids),
            "tile_crops": selected_count * 4,
            "total_seconds": component_sum,
            "images_per_second": len(records) / component_sum if component_sum else 0.0,
            "mean_image_ms": float(np.mean(latencies) * 1000.0),
            "p95_image_ms": float(np.percentile(latencies, 95) * 1000.0),
            "read_seconds": total_read,
            "original_seconds": total_original,
            "feature_seconds": total_feature,
            "selector_seconds": total_selector,
            "tile_seconds": total_tile,
            "fusion_seconds": total_fusion,
            "measurement_pass_wall_seconds": measurement_pass_wall,
            "verified_original_images": len(verified_a),
            "verified_c2_visdrone_images": len(verified_c2),
            "feature_parity_M3": feature_checks["M3_Logistic_F0"],
            "feature_parity_M4": feature_checks["M4_Logistic_F1"],
        })

    repeat_summary = {
        "repeat": repeat,
        "status": "completed",
        "measurement_pass_wall_seconds": measurement_pass_wall,
        "feature_parity": {
            "M3_Logistic_F0": feature_checks["M3_Logistic_F0"],
            "M4_Logistic_F1": feature_checks["M4_Logistic_F1"],
        },
        "prediction_verification": {
            "original_images": len(verified_a),
            "c2_visdrone_images": len(verified_c2),
        },
        "selected_images": {
            name: len(selected_ids[name]) for name in MODELS
        },
        "repeat_metrics": repeat_metric_rows,
    }
    return repeat_summary, per_image_rows


def summarize_values(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0 or not np.all(np.isfinite(array)):
        raise RuntimeError("运行时汇总数据为空或含NaN/Inf")
    return {
        "mean": float(np.mean(array)),
        "std_between_repeats": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 6.3F-C final selector and live runtime benchmark"
    )
    parser.add_argument("--weights", type=str, default=str(WEIGHT_DEFAULT))
    parser.add_argument("--image-dir", type=str, default=str(IMAGE_DIR_DEFAULT))
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--verify-samples", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.resume and args.overwrite:
        parser.error("--resume与--overwrite不能同时使用")
    if args.repeats < 1 or args.warmup < 0 or args.verify_samples < 0:
        parser.error("repeats必须>=1，warmup和verify-samples必须>=0")

    env = check_environment()
    weights = Path(args.weights)
    image_dir = Path(args.image_dir)
    if not weights.is_absolute():
        weights = ROOT / weights
    if not image_dir.is_absolute():
        image_dir = ROOT / image_dir
    if not image_dir.is_dir():
        raise FileNotFoundError(image_dir)

    if OUT_DIR.exists() and any(OUT_DIR.iterdir()):
        if args.overwrite:
            shutil.rmtree(OUT_DIR)
        elif not args.resume:
            raise FileExistsError(
                f"{OUT_DIR}已有内容；中断续跑使用--resume，确认重做才使用--overwrite"
            )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPEAT_DIR = OUT_DIR / "repeats"
    REPEAT_DIR.mkdir(parents=True, exist_ok=True)

    vis_rows, all_rows, data_info, gt_images, category_id, input_hashes = validate_inputs(weights)
    vis_ids = {r["image_id"] for r in vis_rows}
    vis_rows_by_id = {r["image_id"]: r for r in vis_rows}
    all_ids = set(all_rows)

    models, selector_metadata = load_or_fit_selectors(
        vis_rows, input_hashes, OUT_DIR, args.resume
    )

    original_grouped = read_prediction_groups(ORIGINAL_JSON, all_ids)
    c2_grouped = read_prediction_groups(C2_JSON, all_ids)
    check_uavdt_original_only(all_rows, original_grouped, c2_grouped)

    by_relative, by_basename = build_image_index(image_dir)
    records = []
    for image_id in sorted(all_rows):
        row = all_rows[image_id]
        gt_row = gt_images[image_id]
        image_path = resolve_image_path(
            gt_row["file_name"], image_dir, by_relative, by_basename
        )
        records.append({
            "image_id": image_id,
            "source": row["source"],
            "file_name": gt_row["file_name"],
            "sequence_id": row["sequence_id"],
            "width": row["width"],
            "height": row["height"],
            "image_path": image_path,
        })
    if len(records) != EXPECTED_TOTAL:
        raise RuntimeError(f"待测图片数={len(records)}，预期={EXPECTED_TOTAL}")
    for rec in records:
        if not rec["image_path"].is_file():
            raise FileNotFoundError(rec["image_path"])

    try:
        import cv2
        import torch
        from torchvision.ops import batched_nms
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError(
            "需要opencv-python、torch、torchvision、ultralytics"
        ) from exc

    print("=" * 88)
    print("Stage 6.3F-C | Final M3/M4 selectors + live runtime benchmark")
    print("Data:", data_info)
    print("Weights:", weights)
    print("Weight SHA256:", input_hashes["yolo_weights"])
    print("Image directory:", image_dir)
    print("Device:", args.device, "Repeats:", args.repeats, "Warmup:", args.warmup)
    print("Policies: A_Original / C2_area2048 / M3_F0 / M4_F1")
    print("UAVDT=original only; VisDrone=selector-gated C2_area2048")
    print("No YOLO training. No COCO accuracy evaluation. No independent test.")
    print("Environment:", env)
    print("=" * 88, flush=True)

    model = YOLO(str(weights))
    if len(model.names) != 1:
        raise RuntimeError(f"模型类别数应为1，当前为{len(model.names)}")

    device = str(args.device)
    pred_args = prediction_args(device)

    # Warm-up is excluded from timing.
    warm_image = cv2.imread(str(records[0]["image_path"]))
    warm_vis_record = next(r for r in records if r["source"] == "visdrone")
    warm_vis = cv2.imread(str(warm_vis_record["image_path"]))
    if warm_image is None or warm_vis is None:
        raise RuntimeError("Warm-up图片读取失败")
    warm_tiles, _ = make_tiles(warm_vis)
    for _ in range(args.warmup):
        synchronize(torch, device)
        model.predict(source=warm_image, batch=1, **pred_args)
        synchronize(torch, device)
        model.predict(source=warm_tiles, batch=len(warm_tiles), **pred_args)
        synchronize(torch, device)
    del warm_image, warm_vis, warm_tiles
    gc.collect()

    repeat_summaries = []
    all_per_image_rows = []

    for repeat in range(1, args.repeats + 1):
        checkpoint = REPEAT_DIR / f"repeat_{repeat}.json"
        per_image_checkpoint = REPEAT_DIR / f"repeat_{repeat}_per_image.csv"

        if args.resume and checkpoint.is_file() and per_image_checkpoint.is_file():
            cached = load_json(checkpoint)
            if cached.get("status") != "completed":
                raise RuntimeError(f"重复实验检查点未完成：{checkpoint}")
            if cached.get("input_sha256") != input_hashes:
                raise RuntimeError(f"重复实验检查点输入SHA256不一致：{checkpoint}")
            cached_rows = read_csv(per_image_checkpoint)
            repeat_summaries.append(cached["result"])
            all_per_image_rows.extend(cached_rows)
            print(f"[RESUME] repeat={repeat} loaded completed checkpoint", flush=True)
            continue

        print(f"\n[REPEAT {repeat}/{args.repeats}]", flush=True)
        repeat_summary, per_image_rows = run_one_repeat(
            repeat,
            records,
            vis_ids,
            vis_rows_by_id,
            models,
            selector_metadata,
            original_grouped,
            c2_grouped,
            category_id,
            model,
            cv2,
            torch,
            batched_nms,
            device,
            args.verify_samples,
        )

        write_json_atomic(checkpoint, {
            "status": "completed",
            "repeat": repeat,
            "input_sha256": input_hashes,
            "result": repeat_summary,
        })
        write_csv_atomic(per_image_checkpoint, per_image_rows)

        repeat_summaries.append(repeat_summary)
        all_per_image_rows.extend(per_image_rows)

        for row in repeat_summary["repeat_metrics"]:
            print(
                f"  {row['policy']}: {row['total_seconds']:.3f}s | "
                f"{row['images_per_second']:.2f} img/s | "
                f"selected={row['selected_visdrone_images']}/247",
                flush=True,
            )

    if len(repeat_summaries) != args.repeats:
        raise RuntimeError("完成的重复实验数量与--repeats不一致")

    repeat_metric_rows = [
        row
        for result in repeat_summaries
        for row in result["repeat_metrics"]
    ]
    write_csv_atomic(OUT_DIR / "runtime_repeat_metrics.csv", repeat_metric_rows)
    write_csv_atomic(OUT_DIR / "runtime_per_image.csv", all_per_image_rows)

    policy_names = [
        "A_Original",
        "C2_area2048",
        "M3_Logistic_F0",
        "M4_Logistic_F1",
    ]
    summary = {
        "experiment": "Stage6.3F_C_Final_Selector_Live_Runtime_Benchmark",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "completed",
        "protocol": {
            "images": EXPECTED_TOTAL,
            "source_counts": EXPECTED_SOURCE_COUNTS,
            "imgsz": IMGSZ,
            "conf": CONF,
            "nms_iou": NMS_IOU,
            "max_det": MAX_DET,
            "feature_conf": FEATURE_CONF,
            "feature_imgsz": FEATURE_IMGSZ,
            "tiny_area_threshold": TINY_AREA_THRESHOLD,
            "overlap": OVERLAP,
            "tile_layout": "2x2",
            "tile_area_filter": "tile box area < 2048 in original-image pixels",
            "half": HALF,
            "augment": AUGMENT,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "source_policy": (
                "UAVDT always original-only; VisDrone uses C2_area2048 only "
                "when the final selector triggers"
            ),
            "timing_method": (
                "same component-sum method as Stage 6.3D-B: read/original inference "
                "are measured once per image; F0/F1 feature and selector costs are "
                "measured online; tile/fusion costs are measured for each VisDrone "
                "image and included in each policy total only when that policy selects it"
            ),
            "runtime_only": True,
            "no_yolo_training": True,
            "no_accuracy_evaluation": True,
            "independent_test_used": False,
        },
        "environment": env,
        "hardware": {
            "torch": str(torch.__version__),
            "cuda": getattr(torch.version, "cuda", None),
            "gpu_name": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available() and device.lower() not in {"cpu", "-1"}
                else "CPU"
            ),
        },
        "input_sha256": input_hashes,
        "final_selector_metadata": str(
            (OUT_DIR / "final_selector_metadata.json").relative_to(ROOT)
        ),
        "final_selectors": selector_metadata["models"],
        "repeat_results": repeat_summaries,
        "policies": {},
        "outputs": {
            "summary": str((OUT_DIR / "runtime_summary.json").relative_to(ROOT)),
            "repeat_metrics": str((OUT_DIR / "runtime_repeat_metrics.csv").relative_to(ROOT)),
            "per_image": str((OUT_DIR / "runtime_per_image.csv").relative_to(ROOT)),
        },
        "interpretation_limit": (
            "Runtime is measured on the 1567-image development validation set using "
            "the existing source-specific policy. It is not an accuracy estimate. "
            "The final selector was trained on the 247 VisDrone development labels; "
            "Stage 6.4 must evaluate accuracy once on the untouched independent test."
        ),
    }

    for policy in policy_names:
        rows = [r for r in repeat_metric_rows if r["policy"] == policy]
        summary["policies"][policy] = {
            "selected_visdrone_images_mean": float(np.mean([
                r["selected_visdrone_images"] for r in rows
            ])),
            "trigger_rate_visdrone_mean": float(np.mean([
                r["trigger_rate_visdrone"] for r in rows
            ])),
            "tile_crops_mean": float(np.mean([r["tile_crops"] for r in rows])),
            "total_seconds": summarize_values([float(r["total_seconds"]) for r in rows]),
            "images_per_second": summarize_values([float(r["images_per_second"]) for r in rows]),
            "mean_image_ms": summarize_values([float(r["mean_image_ms"]) for r in rows]),
            "p95_image_ms": summarize_values([float(r["p95_image_ms"]) for r in rows]),
            "component_seconds_mean": {
                key: float(np.mean([r[f"{key}_seconds"] for r in rows]))
                for key in (
                    "read", "original", "feature", "selector", "tile", "fusion"
                )
            },
        }

    c2_seconds = summary["policies"]["C2_area2048"]["total_seconds"]["mean"]
    original_seconds = summary["policies"]["A_Original"]["total_seconds"]["mean"]
    for policy in ("A_Original", "M3_Logistic_F0", "M4_Logistic_F1"):
        policy_seconds = summary["policies"][policy]["total_seconds"]["mean"]
        summary["policies"][policy]["speedup_vs_C2"] = (
            c2_seconds / policy_seconds if policy_seconds > 0 else None
        )
        summary["policies"][policy]["time_reduction_vs_C2_percent"] = (
            (c2_seconds - policy_seconds) / c2_seconds * 100.0
            if c2_seconds > 0 else None
        )
        summary["policies"][policy]["time_increase_vs_original_percent"] = (
            (policy_seconds - original_seconds) / original_seconds * 100.0
            if original_seconds > 0 else None
        )

    write_json_atomic(OUT_DIR / "runtime_summary.json", summary)

    print("\n" + "=" * 88)
    print("Stage 6.3F-C completed.")
    for policy in policy_names:
        p = summary["policies"][policy]
        print(
            f"{policy:20s} "
            f"{p['total_seconds']['mean']:.3f}s | "
            f"{p['images_per_second']['mean']:.2f} img/s | "
            f"P95={p['p95_image_ms']['mean']:.2f} ms | "
            f"tiles={p['tile_crops_mean']:.1f}"
        )
    print("Summary:", OUT_DIR / "runtime_summary.json")
    print("Per-image:", OUT_DIR / "runtime_per_image.csv")
    print("=" * 88)


if __name__ == "__main__":
    main()
