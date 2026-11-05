#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 6.3E: sequence-grouped OOF selector budget sensitivity sweep.

Run from the YOLO_Thesis repository root:
    python -u stage6_3e_budget_sweep.py

This experiment:
- sweeps maximum training-fold trigger budgets B30 ... B90;
- keeps the existing three features, rule family, thresholds and labels fixed;
- uses 5-fold GroupKFold by sequence_id;
- reuses committed original/C2 predictions and evaluates them with the fixed
  faster-coco-eval protocol;
- estimates runtime from the completed Stage 6.3D-B component benchmark.

It DOES NOT train YOLO, run YOLO inference, change the dataset, or write to
Stage 6.3C / Stage 6.3D files. All new outputs go to:
    runs/detect/stage6_scene_selector/stage6_3e_budget_sweep/

Runtime values for B30-B90 are estimates, not new live measurements. The
Stage 6.3D-B measured runtime summary is used as a read-only calibration source.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import itertools
import json
import math
import os
import platform
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import GroupKFold
from faster_coco_eval import COCO, COCOeval_faster


# ============================================================================
# 1. Fixed repository paths and experiment protocol
# ============================================================================
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

# Stage 6.3C read-only audit inputs.
C3_ASSIGNMENTS_CSV = SCENE_DIR / "selector_oof_assignments.csv"
C3_RULES_JSON = SCENE_DIR / "selector_fold_rules.json"
C3_SUMMARY_JSON = SCENE_DIR / "selector_oof_summary.json"
C3_PREDICTIONS = {
    "Loose": SCENE_DIR / "selector_oof_predictions_loose.json",
    "Medium": SCENE_DIR / "selector_oof_predictions_medium.json",
    "Strict": SCENE_DIR / "selector_oof_predictions_strict.json",
}

# Stage 6.3D read-only reference inputs.
D3_BOOTSTRAP_JSON = SCENE_DIR / "stage6_3d_bootstrap_summary.json"
D3_RUNTIME_JSON = SCENE_DIR / "stage6_3d_runtime_summary.json"
D3_RUNTIME_PER_IMAGE_CSV = SCENE_DIR / "stage6_3d_runtime_per_image.csv"

# Stage 6.3E outputs: no output path below points to a 6.3C / 6.3D file.
OUT_DIR = SCENE_DIR / "stage6_3e_budget_sweep"
OUT_SUMMARY = OUT_DIR / "budget_sweep_summary.json"
OUT_METRICS = OUT_DIR / "budget_sweep_metrics.csv"
OUT_ASSIGNMENTS = OUT_DIR / "budget_oof_assignments.csv"

EXPECTED_COUNTS = {"uavdt": 1320, "visdrone": 247}
EXPECTED_TOTAL = 1567
N_SPLITS = 5

BUDGETS = {f"B{x}": x / 100.0 for x in (30, 40, 50, 60, 70, 80, 90)}
# Existing Stage 6.3C policies must be exactly reproduced at these budgets.
C3_POLICY_FOR_BUDGET = {"B30": "Strict", "B50": "Medium", "B70": "Loose"}

MIN_TRIGGER_FRACTION = 0.10
MIN_TRIGGER_ABSOLUTE = 5
MAX_RULE_CONDITIONS = 2
QUANTILE_GRID = tuple(float(x) for x in np.arange(0.10, 0.91, 0.10))

# Fixed feature set and directions from Stage 6.3B2 / Stage 6.3C.
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

EXPECTED_EVALUATOR_VERSION = "1.8.0"
EXPECTED_NUMPY_VERSION = "1.26.4"
EXPECTED_SKLEARN_VERSION = "1.5.2"

# Fixed COCO evaluation protocol copied from the reviewed Stage 6.3C script.
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


# ============================================================================
# 2. Basic utilities
# ============================================================================
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json_atomic(path: Path, obj: Any, *, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        if compact:
            json.dump(obj, f, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        else:
            json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(tmp, path)


def write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"拒绝写入空CSV：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def finite_float(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"字段{field}不是有效数字：{value!r}") from exc
    if not math.isfinite(result):
        raise RuntimeError(f"字段{field}不是有限数值：{value!r}")
    return result


def load_csv_by_id(path: Path) -> dict[int, dict[str, str]]:
    rows: dict[int, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "image_id" not in reader.fieldnames:
            raise RuntimeError(f"CSV缺少image_id列：{path}")
        for raw in reader:
            image_id = int(raw["image_id"])
            if image_id in rows:
                raise RuntimeError(f"{path.name}存在重复image_id={image_id}")
            rows[image_id] = raw
    return rows


def feature_value(row: dict[str, Any], name: str) -> float:
    value = row.get(name)
    if value is None or value == "":
        # Same treatment as Stage 6.3C: no detection -> median area is +inf,
        # so the condition median_box_area_640 <= threshold is false.
        if name == "median_box_area_640":
            return float("inf")
        raise RuntimeError(f"特征{name}缺失，image_id={row.get('image_id')}")
    return float(value)


def prediction_signature(records: list[dict[str, Any]]) -> Counter:
    return Counter(
        (
            int(p["category_id"]),
            round(float(p["score"]), 8),
            tuple(round(float(v), 6) for v in p["bbox"]),
        )
        for p in records
    )


def require_files(paths: list[Path]) -> None:
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("缺少必要输入文件：\n" + "\n".join(missing))


def check_package_versions(c3_summary: dict[str, Any]) -> dict[str, str]:
    current = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit_learn": __import__("sklearn").__version__,
        "faster_coco_eval": version("faster-coco-eval"),
    }
    expected_env = c3_summary.get("environment", {})
    if current["numpy"] != EXPECTED_NUMPY_VERSION:
        raise RuntimeError(
            f"NumPy版本不匹配：当前{current['numpy']}，Stage 6.3C为{EXPECTED_NUMPY_VERSION}。"
        )
    if current["scikit_learn"] != EXPECTED_SKLEARN_VERSION:
        raise RuntimeError(
            f"scikit-learn版本不匹配：当前{current['scikit_learn']}，Stage 6.3C为{EXPECTED_SKLEARN_VERSION}。"
        )
    if current["faster_coco_eval"] != EXPECTED_EVALUATOR_VERSION:
        raise RuntimeError(
            f"faster-coco-eval版本不匹配：当前{current['faster_coco_eval']}，"
            f"要求{EXPECTED_EVALUATOR_VERSION}。"
        )
    for key in ("numpy", "scikit_learn", "faster_coco_eval"):
        if expected_env.get(key) and expected_env[key] != current[key]:
            raise RuntimeError(
                f"当前环境与Stage 6.3C环境不一致：{key}={current[key]}，"
                f"Stage 6.3C={expected_env[key]}。"
            )
    return current


# ============================================================================
# 3. Validate Stage 6.3C / Stage 6.3D read-only inputs
# ============================================================================
def check_common_inputs() -> dict[str, Any]:
    required = [
        FEATURE_CSV, UTILITY_CSV, MAPPING_CSV, A_JSON, C2_JSON, GT_JSON,
        REFERENCE_JSON, C3_ASSIGNMENTS_CSV, C3_RULES_JSON, C3_SUMMARY_JSON,
        D3_BOOTSTRAP_JSON, D3_RUNTIME_JSON, D3_RUNTIME_PER_IMAGE_CSV,
        *C3_PREDICTIONS.values(),
    ]
    require_files(required)

    c3_summary = load_json(C3_SUMMARY_JSON)
    c3_rules = load_json(C3_RULES_JSON)
    d3_bootstrap = load_json(D3_BOOTSTRAP_JSON)
    d3_runtime = load_json(D3_RUNTIME_JSON)
    reference = load_json(REFERENCE_JSON)

    if c3_summary.get("status") != "completed":
        raise RuntimeError("Stage 6.3C summary状态不是completed。")
    if c3_summary.get("experiment") != "Stage6.3C_5Fold_GroupKFold_OOF_Rule_Selector":
        raise RuntimeError("Stage 6.3C summary不是预期版本。")
    if d3_bootstrap.get("status") != "completed" or d3_runtime.get("status") != "completed":
        raise RuntimeError("Stage 6.3D-A或Stage 6.3D-B尚未完成，不能开展预算扫描。")

    protocol = c3_summary.get("protocol", {})
    expected_c3_protocol = {
        "n_images": EXPECTED_TOTAL,
        "source_counts": EXPECTED_COUNTS,
        "splitter": "GroupKFold",
        "n_splits": N_SPLITS,
        "group_key": "sequence_id",
        "feature_columns": FEATURE_COLUMNS,
        "max_rule_conditions": MAX_RULE_CONDITIONS,
        "rule_logic": "AND",
        "evaluator_version": EXPECTED_EVALUATOR_VERSION,
    }
    for key, expected in expected_c3_protocol.items():
        if protocol.get(key) != expected:
            raise RuntimeError(
                f"Stage 6.3C协议字段不匹配：{key}; expected={expected!r}; actual={protocol.get(key)!r}"
            )
    expected_old_budgets = {"Loose": 0.70, "Medium": 0.50, "Strict": 0.30}
    if protocol.get("policy_budgets") != expected_old_budgets:
        raise RuntimeError("Stage 6.3C的Loose/Medium/Strict预算与已审阅版本不一致。")

    rules_protocol = c3_rules.get("protocol", {})
    if rules_protocol.get("n_splits") != N_SPLITS:
        raise RuntimeError("selector_fold_rules.json的折数不是5。")
    if rules_protocol.get("group_key") != "sequence_id":
        raise RuntimeError("selector_fold_rules.json没有按sequence_id分组。")
    if rules_protocol.get("feature_columns") != FEATURE_COLUMNS:
        raise RuntimeError("selector_fold_rules.json的特征集合发生变化。")
    if rules_protocol.get("threshold_quantiles") != list(QUANTILE_GRID):
        raise RuntimeError("selector_fold_rules.json的分位数网格发生变化。")
    if rules_protocol.get("feature_directions") != FEATURE_SPECS:
        raise RuntimeError("selector_fold_rules.json的特征方向发生变化。")
    if rules_protocol.get("max_rule_conditions") != MAX_RULE_CONDITIONS:
        raise RuntimeError("selector_fold_rules.json的规则条件数量发生变化。")
    if float(rules_protocol.get("minimum_trigger_fraction", -1)) != MIN_TRIGGER_FRACTION:
        raise RuntimeError("selector_fold_rules.json的最小触发比例发生变化。")
    if int(rules_protocol.get("minimum_trigger_absolute", -1)) != MIN_TRIGGER_ABSOLUTE:
        raise RuntimeError("selector_fold_rules.json的最小触发图片数发生变化。")
    if rules_protocol.get("policy_max_trigger_budgets") != {"Loose": 0.70, "Medium": 0.50, "Strict": 0.30}:
        raise RuntimeError("selector_fold_rules.json中的原始预算设置发生变化。")

    # Stage 6.3D-A point estimates must agree with the completed Stage 6.3C
    # official COCO results before the new budget sweep is allowed to proceed.
    bootstrap_points = d3_bootstrap.get("stage6_3c_point_estimates", {})
    for policy in ("Loose", "Medium", "Strict"):
        if policy not in bootstrap_points:
            raise RuntimeError(f"Stage 6.3D-A缺少{policy}的Stage 6.3C点估计。")
        for scope in ("full", "visdrone"):
            old_metrics = c3_summary["policies"][policy]["coco_metrics"][scope]
            boot_metrics = bootstrap_points[policy][scope]
            for metric_key in METRIC_KEYS:
                if abs(float(old_metrics[metric_key]) - float(boot_metrics[metric_key])) > 1e-9:
                    raise RuntimeError(
                        f"Stage 6.3D-A与Stage 6.3C点估计不一致：{policy}/{scope}/{metric_key}"
                    )

    bootstrap = d3_bootstrap.get("bootstrap", {})
    if bootstrap.get("replicates") != 1000 or bootstrap.get("seed") != 42:
        raise RuntimeError("Stage 6.3D-A不是已审阅的1000次、seed=42 Bootstrap结果。")
    if bootstrap.get("number_of_sequence_groups") != 94:
        raise RuntimeError("Stage 6.3D-A的VisDrone序列组数量不是94。")

    expected_runtime_csv = str(D3_RUNTIME_PER_IMAGE_CSV.relative_to(ROOT))
    if d3_runtime.get("per_image_csv") != expected_runtime_csv:
        raise RuntimeError(
            "Stage 6.3D-B summary引用的逐图耗时CSV路径与当前文件不一致："
            f"{d3_runtime.get('per_image_csv')} vs {expected_runtime_csv}"
        )

    runtime_protocol = d3_runtime.get("protocol", {})
    runtime_expected = {
        "images": EXPECTED_TOTAL,
        "source_counts": EXPECTED_COUNTS,
        "imgsz": 640,
        "conf": 0.001,
        "nms_iou": 0.7,
        "max_det": 300,
        "overlap": 0.15,
        "tile_layout": "2x2",
        "evaluator_version": EXPECTED_EVALUATOR_VERSION,
        "repeats": 3,
    }
    for key, expected in runtime_expected.items():
        if runtime_protocol.get(key) != expected:
            raise RuntimeError(
                f"Stage 6.3D-B运行时协议不匹配：{key}; expected={expected!r}; actual={runtime_protocol.get(key)!r}"
            )
    if d3_runtime.get("selector_verification", {}).get("result") != "passed":
        raise RuntimeError("Stage 6.3D-B的实时选择器核验未通过。")
    if d3_runtime.get("prediction_verification", {}).get("result") != "passed":
        raise RuntimeError("Stage 6.3D-B的预测核验未通过。")
    expected_runtime_counts = {"C2_area2048": 247, "Loose": 154, "Medium": 120}
    for policy, expected_count in expected_runtime_counts.items():
        actual_count = int(d3_runtime.get("policies", {}).get(policy, {}).get("selected_visdrone_images", -1))
        if actual_count != expected_count:
            raise RuntimeError(
                f"Stage 6.3D-B {policy}触发数量异常：{actual_count}，预期{expected_count}。"
            )

    actual_hashes = {
        "features": sha256_file(FEATURE_CSV),
        "original": sha256_file(A_JSON),
        "c2": sha256_file(C2_JSON),
        "gt": sha256_file(GT_JSON),
    }
    for key, expected in EXPECTED_SHA256.items():
        if actual_hashes[key] != expected:
            raise RuntimeError(
                f"输入SHA256与已审阅版本不匹配：{key}\n"
                f"expected={expected}\nactual={actual_hashes[key]}"
            )

    # All prior stages must refer to the exact same frozen prediction/GT files.
    for stage_name, stage_obj in (
        ("Stage 6.3C", c3_summary),
        ("Stage 6.3D-A", d3_bootstrap),
        ("Stage 6.3D-B", d3_runtime),
    ):
        stage_hashes = stage_obj.get("input_sha256", {})
        for key, actual in actual_hashes.items():
            if stage_hashes.get(key) != actual:
                raise RuntimeError(
                    f"{stage_name}的{key}哈希与当前输入不一致："
                    f"stage={stage_hashes.get(key)}, current={actual}"
                )

    if reference.get("prediction_sha256", {}).get("A_Original") != actual_hashes["original"]:
        raise RuntimeError("faster_eval_metrics.json中的A_Original哈希不匹配。")
    if reference.get("prediction_sha256", {}).get("C2_area2048") != actual_hashes["c2"]:
        raise RuntimeError("faster_eval_metrics.json中的C2_area2048哈希不匹配。")
    if reference.get("ground_truth", {}).get("sha256") != actual_hashes["gt"]:
        raise RuntimeError("faster_eval_metrics.json中的GT哈希不匹配。")
    ref_protocol = reference.get("protocol", {})
    if ref_protocol.get("evaluator_version") != EXPECTED_EVALUATOR_VERSION:
        raise RuntimeError("faster_eval_metrics.json的评价器版本不是1.8.0。")
    if int(ref_protocol.get("imgsz", -1)) != 640 or float(ref_protocol.get("nms_iou", -1)) != 0.7:
        raise RuntimeError("faster_eval_metrics.json的输入尺寸或NMS IoU不匹配。")

    environment = check_package_versions(c3_summary)

    # Check that the C3 summary's three reference prediction files are intact.
    for policy, path in C3_PREDICTIONS.items():
        expected_sha = c3_summary["policies"][policy]["prediction"]["sha256"]
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"Stage 6.3C {policy} OOF预测文件SHA256不匹配："
                f"expected={expected_sha}, actual={actual_sha}"
            )

    return {
        "c3_summary": c3_summary,
        "c3_rules": c3_rules,
        "d3_bootstrap": d3_bootstrap,
        "d3_runtime": d3_runtime,
        "reference": reference,
        "input_sha256": actual_hashes,
        "additional_input_sha256": {
            "utility_csv": sha256_file(UTILITY_CSV),
            "mapping_csv": sha256_file(MAPPING_CSV),
            "c3_assignments_csv": sha256_file(C3_ASSIGNMENTS_CSV),
            "c3_rules_json": sha256_file(C3_RULES_JSON),
            "c3_summary_json": sha256_file(C3_SUMMARY_JSON),
            "d3_bootstrap_json": sha256_file(D3_BOOTSTRAP_JSON),
            "d3_runtime_json": sha256_file(D3_RUNTIME_JSON),
            "d3_runtime_per_image_csv": sha256_file(D3_RUNTIME_PER_IMAGE_CSV),
        },
        "environment": environment,
    }


# ============================================================================
# 4. Load and validate features, labels, source mapping and predictions
# ============================================================================
def load_feature_and_utility_rows() -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    feature_raw = load_csv_by_id(FEATURE_CSV)
    utility_raw = load_csv_by_id(UTILITY_CSV)
    if len(feature_raw) != EXPECTED_TOTAL or len(utility_raw) != EXPECTED_TOTAL:
        raise RuntimeError(
            f"CSV行数错误：features={len(feature_raw)}, utility={len(utility_raw)}，"
            f"预期均为{EXPECTED_TOTAL}。"
        )
    if set(feature_raw) != set(utility_raw):
        raise RuntimeError("corrected_scene_features.csv与coco_utility_analysis.csv的image_id不一致。")

    feature_by_id: dict[int, dict[str, Any]] = {}
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
            # The label is a fold-training target and a held-out diagnostic only.
            # It is NEVER passed to apply_rule() or used as an inference feature.
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
        raise RuntimeError(f"Stage 6.3B2标签分布发生变化：{dict(label_counts)}")
    for row in vis_rows:
        if row["selector_label"] not in {"beneficial", "harmful", "neutral"}:
            raise RuntimeError(
                f"VisDrone标签非法：image_id={row['image_id']}, label={row['selector_label']}"
            )
    return feature_by_id, {
        "source_counts": dict(source_counts),
        "label_counts": dict(label_counts),
    }


def load_and_check_mapping(feature_by_id: dict[int, dict[str, Any]]) -> dict[int, str]:
    mapping: dict[int, str] = {}
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


def load_prediction_groups(path: Path, valid_ids: set[int]) -> tuple[dict[int, list[dict[str, Any]]], int]:
    raw = load_json(path)
    if not isinstance(raw, list):
        raise RuntimeError(f"预测文件不是JSON list：{path}")
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for pred in raw:
        image_id = int(pred["image_id"])
        if image_id not in valid_ids:
            raise RuntimeError(f"预测文件包含未知image_id={image_id}：{path}")
        for key in ("category_id", "bbox", "score"):
            if key not in pred:
                raise RuntimeError(f"预测记录缺少{key}：{path}")
        if len(pred["bbox"]) != 4:
            raise RuntimeError(f"bbox格式错误：{pred}")
        box = [float(x) for x in pred["bbox"]]
        if not all(math.isfinite(x) for x in box) or box[2] < 0 or box[3] < 0:
            raise RuntimeError(f"bbox数值非法：{pred}")
        if not math.isfinite(float(pred["score"])):
            raise RuntimeError(f"score数值非法：{pred}")
        grouped[image_id].append(pred)
    return dict(grouped), len(raw)


def check_uavdt_original_only(
    feature_by_id: dict[int, dict[str, Any]],
    a_by_id: dict[int, list[dict[str, Any]]],
    c2_by_id: dict[int, list[dict[str, Any]]],
) -> dict[str, Any]:
    mismatches = []
    for image_id, row in feature_by_id.items():
        if row["source"] != "uavdt":
            continue
        if prediction_signature(a_by_id.get(image_id, [])) != prediction_signature(c2_by_id.get(image_id, [])):
            mismatches.append(image_id)
    if mismatches:
        raise RuntimeError(f"C2在UAVDT上不是原图预测，前20个不一致ID：{mismatches[:20]}")
    return {"checked_images": EXPECTED_COUNTS["uavdt"], "mismatch_images": 0, "passed": True}


# ============================================================================
# 5. Fixed rule learner: same as Stage 6.3C, only the budget changes
# ============================================================================
def utility_target(row: dict[str, Any]) -> int:
    label = row["selector_label"]
    if label == "beneficial":
        return 1
    if label == "harmful":
        return -1
    return 0


def get_thresholds(train_rows: list[dict[str, Any]], feature: str) -> list[float]:
    values = np.asarray([feature_value(row, feature) for row in train_rows], dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return []
    thresholds = np.quantile(values, QUANTILE_GRID)
    return sorted({round(float(x), 8) for x in thresholds})


def condition_mask(rows: list[dict[str, Any]], condition: dict[str, Any]) -> np.ndarray:
    feature = condition["feature"]
    threshold = float(condition["threshold"])
    values = np.asarray([feature_value(row, feature) for row in rows], dtype=np.float64)
    if condition["op"] == ">=":
        return values >= threshold
    if condition["op"] == "<=":
        return values <= threshold
    raise RuntimeError(f"未知规则运算符：{condition['op']}")


def apply_rule(rows: list[dict[str, Any]], rule: dict[str, Any]) -> np.ndarray:
    if rule.get("rule_type") == "none":
        return np.zeros(len(rows), dtype=bool)
    masks = [condition_mask(rows, condition) for condition in rule.get("conditions", [])]
    if not masks:
        return np.zeros(len(rows), dtype=bool)
    selected = masks[0].copy()
    for mask in masks[1:]:
        selected &= mask
    return selected


def fit_rule(train_rows: list[dict[str, Any]], max_trigger_fraction: float) -> dict[str, Any]:
    """Fit a 1- or 2-condition AND rule using training-fold rows only."""
    if not train_rows:
        raise RuntimeError("规则训练集为空。")

    y = np.asarray([utility_target(row) for row in train_rows], dtype=np.int64)
    n_train = len(train_rows)
    min_selected = max(MIN_TRIGGER_ABSOLUTE, int(math.ceil(MIN_TRIGGER_FRACTION * n_train)))
    max_selected = int(math.floor(max_trigger_fraction * n_train))
    if max_selected < min_selected:
        return {"rule_type": "none", "conditions": [], "reason": "trigger_budget_below_minimum"}

    thresholds_by_feature = {
        feature: get_thresholds(train_rows, feature) for feature in FEATURE_COLUMNS
    }
    best = None
    candidate_count = 0

    for n_conditions in range(1, MAX_RULE_CONDITIONS + 1):
        for feature_combo in itertools.combinations(FEATURE_COLUMNS, n_conditions):
            threshold_lists = [thresholds_by_feature[feature] for feature in feature_combo]
            if any(not values for values in threshold_lists):
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

                # Identical Stage 6.3C ranking: utility, precision, simplicity, coverage.
                ranking = (net_utility, beneficial_precision, -len(conditions), n_selected)
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


def evaluate_rule_diagnostics(rows: list[dict[str, Any]], selected: np.ndarray) -> dict[str, Any]:
    labels = [row["selector_label"] for row in rows]
    n_selected = int(selected.sum())
    n_beneficial_total = sum(label == "beneficial" for label in labels)
    n_harmful_total = sum(label == "harmful" for label in labels)
    n_beneficial_selected = sum(bool(selected[i]) and labels[i] == "beneficial" for i in range(len(rows)))
    n_harmful_selected = sum(bool(selected[i]) and labels[i] == "harmful" for i in range(len(rows)))
    n_neutral_selected = sum(bool(selected[i]) and labels[i] == "neutral" for i in range(len(rows)))

    selected_delta_f1 = [
        float(rows[i]["delta_small_mean_f1_50_95"])
        for i in range(len(rows)) if selected[i] and rows[i]["small_gt_count"] > 0
    ]
    selected_delta_fp = [
        float(rows[i]["delta_small_mean_fp_50_95"])
        for i in range(len(rows)) if selected[i] and rows[i]["small_gt_count"] == 0
    ]
    return {
        "images": len(rows),
        "selected_images": n_selected,
        "trigger_rate": n_selected / len(rows) if rows else 0.0,
        "beneficial_selected": n_beneficial_selected,
        "harmful_selected": n_harmful_selected,
        "neutral_selected": n_neutral_selected,
        "beneficial_precision_among_selected": n_beneficial_selected / n_selected if n_selected else 0.0,
        "beneficial_recall": n_beneficial_selected / n_beneficial_total if n_beneficial_total else 0.0,
        "harmful_capture_rate": n_harmful_selected / n_harmful_total if n_harmful_total else 0.0,
        "net_label_utility": n_beneficial_selected - n_harmful_selected,
        "mean_selected_delta_f1_when_small_gt_exists": float(np.mean(selected_delta_f1)) if selected_delta_f1 else None,
        "mean_selected_delta_small_fp_when_no_small_gt": float(np.mean(selected_delta_fp)) if selected_delta_fp else None,
    }


# ============================================================================
# 6. Read Stage 6.3C OOF assignments and enforce exact overlap checks
# ============================================================================
def read_c3_assignments() -> dict[str, dict[int, dict[str, int]]]:
    result: dict[str, dict[int, dict[str, int]]] = {name: {} for name in ("Loose", "Medium", "Strict")}
    with C3_ASSIGNMENTS_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"policy", "fold", "image_id", "selected_for_c2_2048"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"Stage 6.3C assignments缺少字段：{sorted(missing)}")
        for row in reader:
            policy = row["policy"]
            if policy not in result:
                raise RuntimeError(f"Stage 6.3C assignments出现未知policy={policy}")
            image_id = int(row["image_id"])
            if image_id in result[policy]:
                raise RuntimeError(f"Stage 6.3C assignments重复：{policy}, image_id={image_id}")
            result[policy][image_id] = {
                "fold": int(row["fold"]),
                "selected": int(row["selected_for_c2_2048"]),
            }
    for policy, mapping in result.items():
        if len(mapping) != EXPECTED_COUNTS["visdrone"]:
            raise RuntimeError(
                f"Stage 6.3C {policy} OOF覆盖数量错误：{len(mapping)}，"
                f"预期{EXPECTED_COUNTS['visdrone']}。"
            )
    return result


def load_stage6_3d_per_image_timings(
    source_by_id: dict[int, str],
    feature_by_id: dict[int, dict[str, Any]],
    c3_assignments: dict[str, dict[int, dict[str, int]]],
    runtime_summary: dict[str, Any],
) -> dict[int, dict[str, dict[int, dict[str, Any]]]]:
    """Load Stage 6.3D-B's measured per-image components; no inference is run."""
    required_columns = {
        "repeat", "policy", "image_id", "source", "sequence_id",
        "selected_for_tiling", "read_ms", "original_ms", "feature_ms",
        "rule_ms", "tile_ms", "fusion_ms", "total_image_ms",
    }
    timings: dict[int, dict[str, dict[int, dict[str, Any]]]] = defaultdict(
        lambda: {name: {} for name in ("A_Original", "C2_area2048", "Loose", "Medium")}
    )
    with D3_RUNTIME_PER_IMAGE_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = required_columns - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"Stage 6.3D-B逐图耗时CSV缺少字段：{sorted(missing)}")
        for raw in reader:
            repeat = int(raw["repeat"])
            policy = raw["policy"]
            image_id = int(raw["image_id"])
            if policy not in ("A_Original", "C2_area2048", "Loose", "Medium"):
                raise RuntimeError(f"Stage 6.3D-B逐图CSV出现未知策略：{policy}")
            if image_id not in source_by_id:
                raise RuntimeError(f"Stage 6.3D-B逐图CSV出现未知image_id={image_id}")
            if image_id in timings[repeat][policy]:
                raise RuntimeError(f"Stage 6.3D-B逐图CSV重复：repeat={repeat}, policy={policy}, image_id={image_id}")
            if raw["source"].strip().lower() != source_by_id[image_id]:
                raise RuntimeError(f"Stage 6.3D-B逐图CSV来源不一致：image_id={image_id}")
            if raw["sequence_id"] != feature_by_id[image_id]["sequence_id"]:
                raise RuntimeError(f"Stage 6.3D-B逐图CSV sequence_id不一致：image_id={image_id}")

            row = {
                "image_id": image_id,
                "source": raw["source"].strip().lower(),
                "sequence_id": raw["sequence_id"],
                "selected_for_tiling": int(raw["selected_for_tiling"]),
                "read_ms": finite_float(raw["read_ms"], "read_ms"),
                "original_ms": finite_float(raw["original_ms"], "original_ms"),
                "feature_ms": finite_float(raw["feature_ms"], "feature_ms"),
                "rule_ms": finite_float(raw["rule_ms"], "rule_ms"),
                "tile_ms": finite_float(raw["tile_ms"], "tile_ms"),
                "fusion_ms": finite_float(raw["fusion_ms"], "fusion_ms"),
                "total_image_ms": finite_float(raw["total_image_ms"], "total_image_ms"),
            }
            for key in ("read_ms", "original_ms", "feature_ms", "rule_ms", "tile_ms", "fusion_ms", "total_image_ms"):
                if row[key] < 0:
                    raise RuntimeError(f"Stage 6.3D-B逐图耗时出现负值：{key}, image_id={image_id}")
            component_sum = sum(row[key] for key in ("read_ms", "original_ms", "feature_ms", "rule_ms", "tile_ms", "fusion_ms"))
            if abs(component_sum - row["total_image_ms"]) > 1e-5:
                raise RuntimeError(
                    f"Stage 6.3D-B逐图耗时分解不一致：repeat={repeat}, policy={policy}, "
                    f"image_id={image_id}, components={component_sum}, total={row['total_image_ms']}"
                )
            timings[repeat][policy][image_id] = row

    expected_repeats = int(runtime_summary["protocol"]["repeats"])
    expected_ids = set(source_by_id)
    if set(timings) != set(range(1, expected_repeats + 1)):
        raise RuntimeError(f"Stage 6.3D-B逐图CSV repeat编号错误：{sorted(timings)}")

    vis_ids = {image_id for image_id, source in source_by_id.items() if source == "visdrone"}
    for repeat in range(1, expected_repeats + 1):
        for policy in ("A_Original", "C2_area2048", "Loose", "Medium"):
            rows = timings[repeat][policy]
            if set(rows) != expected_ids or len(rows) != EXPECTED_TOTAL:
                raise RuntimeError(
                    f"Stage 6.3D-B逐图CSV覆盖错误：repeat={repeat}, policy={policy}, "
                    f"rows={len(rows)}, expected={EXPECTED_TOTAL}"
                )

        # Validate that the measured selector decisions are exactly the C3 OOF decisions.
        for policy in ("Loose", "Medium"):
            for image_id in vis_ids:
                measured = timings[repeat][policy][image_id]["selected_for_tiling"]
                expected = c3_assignments[policy][image_id]["selected"]
                if measured != expected:
                    raise RuntimeError(
                        f"Stage 6.3D-B逐图选择与Stage 6.3C不一致：{policy}, "
                        f"repeat={repeat}, image_id={image_id}, measured={measured}, expected={expected}"
                    )
        for image_id in vis_ids:
            if timings[repeat]["C2_area2048"][image_id]["selected_for_tiling"] != 1:
                raise RuntimeError(f"C2逐图耗时记录未对VisDrone全量切片：repeat={repeat}, image_id={image_id}")
        for image_id, source in source_by_id.items():
            if source == "uavdt":
                for policy in ("A_Original", "C2_area2048", "Loose", "Medium"):
                    if timings[repeat][policy][image_id]["selected_for_tiling"] != 0:
                        raise RuntimeError(f"UAVDT逐图耗时记录不应切片：{policy}, image_id={image_id}")
            if timings[repeat]["A_Original"][image_id]["selected_for_tiling"] != 0:
                raise RuntimeError(f"A_Original逐图耗时记录不应切片：image_id={image_id}")

    # Read/original inference is performed once per image and reused across
    # all policy reconstructions in Stage 6.3D-B. Verify those components.
    for repeat in range(1, expected_repeats + 1):
        for image_id in expected_ids:
            base_row = timings[repeat]["A_Original"][image_id]
            for policy in ("C2_area2048", "Loose", "Medium"):
                other = timings[repeat][policy][image_id]
                if (abs(base_row["read_ms"] - other["read_ms"]) > 1e-8
                        or abs(base_row["original_ms"] - other["original_ms"]) > 1e-8):
                    raise RuntimeError(
                        f"Stage 6.3D-B read/original耗时在策略间不一致："
                        f"repeat={repeat}, policy={policy}, image_id={image_id}"
                    )

    # The feature pass is computed once per VisDrone image and copied into both
    # adaptive policy rows by Stage 6.3D-B. Verify that identity before averaging.
    for repeat in range(1, expected_repeats + 1):
        for image_id in vis_ids:
            f_loose = timings[repeat]["Loose"][image_id]["feature_ms"]
            f_medium = timings[repeat]["Medium"][image_id]["feature_ms"]
            if abs(f_loose - f_medium) > 1e-8:
                raise RuntimeError(
                    f"Stage 6.3D-B两种策略的feature_ms不一致：repeat={repeat}, image_id={image_id}"
                )

    print(
        f"[RUNTIME CSV VERIFIED] {len(timings)} repeats x 4 policies x "
        f"{EXPECTED_TOTAL} images = {len(timings) * 4 * EXPECTED_TOTAL} rows"
    )
    return dict(timings)


# ============================================================================
# 7. GroupKFold OOF sweep: only the maximum trigger budget changes
# ============================================================================
def run_groupkfold_budget_sweep(
    feature_by_id: dict[int, dict[str, Any]],
    c3_assignments: dict[str, dict[int, dict[str, int]]],
) -> tuple[dict[str, set[int]], dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    vis_rows = sorted(
        [row for row in feature_by_id.values() if row["source"] == "visdrone"],
        key=lambda row: row["image_id"],
    )
    groups = np.asarray([row["sequence_id"] for row in vis_rows], dtype=object)
    unique_groups = sorted(set(groups.tolist()))
    if len(unique_groups) != 94:
        raise RuntimeError(f"VisDrone sequence_id组数不是94：{len(unique_groups)}")

    splitter = GroupKFold(n_splits=N_SPLITS)
    selected_by_budget = {name: set() for name in BUDGETS}
    fold_rules_by_budget = {name: [] for name in BUDGETS}
    assignment_rows: list[dict[str, Any]] = []
    fold_by_image: dict[int, int] = {}

    x_dummy = np.zeros((len(vis_rows), 1), dtype=np.float32)
    y_dummy = np.zeros(len(vis_rows), dtype=np.int64)

    for fold_id, (train_idx, test_idx) in enumerate(
        splitter.split(x_dummy, y_dummy, groups), start=1
    ):
        train_rows = [vis_rows[int(i)] for i in train_idx]
        test_rows = [vis_rows[int(i)] for i in test_idx]
        train_groups = set(row["sequence_id"] for row in train_rows)
        test_groups = set(row["sequence_id"] for row in test_rows)
        if train_groups & test_groups:
            raise RuntimeError(f"Fold {fold_id}发生sequence_id泄漏。")
        for row in test_rows:
            image_id = int(row["image_id"])
            if image_id in fold_by_image:
                raise RuntimeError(f"image_id={image_id}被分配到多个OOF折。")
            fold_by_image[image_id] = fold_id

        print(
            f"\n[FOLD {fold_id}/{N_SPLITS}] train={len(train_rows)} images/"
            f"{len(train_groups)} sequences; test={len(test_rows)} images/"
            f"{len(test_groups)} sequences"
        )

        for budget_name, budget_fraction in BUDGETS.items():
            rule = fit_rule(train_rows, budget_fraction)
            test_selected = apply_rule(test_rows, rule)
            diagnostics = evaluate_rule_diagnostics(test_rows, test_selected)

            for index, row in enumerate(test_rows):
                image_id = int(row["image_id"])
                is_selected = int(bool(test_selected[index]))
                if is_selected:
                    selected_by_budget[budget_name].add(image_id)
                assignment_rows.append({
                    "budget": budget_name,
                    "max_trigger_budget": budget_fraction,
                    "fold": fold_id,
                    "image_id": image_id,
                    "file_name": row["file_name"],
                    "sequence_id": row["sequence_id"],
                    "selected_for_c2_2048": is_selected,
                    "selector_label_audit_only": row["selector_label"],
                    "n_det_conf025": row["n_det_conf025"],
                    "median_box_area_640": "" if row["median_box_area_640"] is None else row["median_box_area_640"],
                    "tiny_ratio": row["tiny_ratio"],
                    "rule_type": rule.get("rule_type"),
                    "rule_conditions": json.dumps(rule.get("conditions", []), ensure_ascii=False),
                })

            fold_rules_by_budget[budget_name].append({
                "fold": fold_id,
                "train_groups": sorted(train_groups),
                "test_groups": sorted(test_groups),
                "train_images": len(train_rows),
                "test_images": len(test_rows),
                "rule": rule,
                "test_diagnostics": diagnostics,
            })
            print(
                f"  {budget_name}: max_budget={budget_fraction:.0%}; "
                f"rule={rule.get('conditions', []) or 'NONE'}; "
                f"heldout_trigger={diagnostics['selected_images']}/{diagnostics['images']} "
                f"({diagnostics['trigger_rate']:.1%}); "
                f"beneficial/harmful={diagnostics['beneficial_selected']}/"
                f"{diagnostics['harmful_selected']}"
            )

    expected_vis_ids = {int(row["image_id"]) for row in vis_rows}
    if set(fold_by_image) != expected_vis_ids or len(fold_by_image) != EXPECTED_COUNTS["visdrone"]:
        raise RuntimeError("GroupKFold没有且仅有一次覆盖全部247张VisDrone图片。")

    for budget_name, selected_ids in selected_by_budget.items():
        if not selected_ids.issubset(expected_vis_ids):
            raise RuntimeError(f"{budget_name}产生了非VisDrone image_id。")
        rows = [row for row in assignment_rows if row["budget"] == budget_name]
        assigned_ids = [int(row["image_id"]) for row in rows]
        if len(assigned_ids) != len(expected_vis_ids) or set(assigned_ids) != expected_vis_ids:
            raise RuntimeError(f"{budget_name}的OOF决策没有完整覆盖247张图片。")

    # Exact reproduction checks: B30=Strict, B50=Medium, B70=Loose.
    for budget_name, c3_policy in C3_POLICY_FOR_BUDGET.items():
        for image_id in expected_vis_ids:
            old = c3_assignments[c3_policy][image_id]
            new_rows = [
                row for row in assignment_rows
                if row["budget"] == budget_name and int(row["image_id"]) == image_id
            ]
            if len(new_rows) != 1:
                raise RuntimeError(f"{budget_name} image_id={image_id}的OOF记录不唯一。")
            new = new_rows[0]
            if int(new["fold"]) != old["fold"] or int(new["selected_for_c2_2048"]) != old["selected"]:
                raise RuntimeError(
                    f"{budget_name}不能复现Stage 6.3C {c3_policy}：image_id={image_id}, "
                    f"old_fold/selected={old['fold']}/{old['selected']}, "
                    f"new_fold/selected={new['fold']}/{new['selected']}。"
                )
        print(f"[C3 REPRODUCTION PASSED] {budget_name} == Stage 6.3C {c3_policy}")

    return selected_by_budget, fold_rules_by_budget, assignment_rows


# ============================================================================
# 8. Create combined prediction JSONs and run official COCO evaluation
# ============================================================================
def write_budget_prediction(
    budget_name: str,
    selected_vis_ids: set[int],
    feature_by_id: dict[int, dict[str, Any]],
    a_by_id: dict[int, list[dict[str, Any]]],
    c2_by_id: dict[int, list[dict[str, Any]]],
) -> tuple[Path, dict[str, Any]]:
    budget_num = int(budget_name[1:])
    output_path = OUT_DIR / f"budget_{budget_num}_predictions.json"
    output: list[dict[str, Any]] = []
    selected_detection_count = 0
    original_detection_count = 0

    for image_id in sorted(feature_by_id):
        row = feature_by_id[image_id]
        use_c2 = row["source"] == "visdrone" and image_id in selected_vis_ids
        records = c2_by_id.get(image_id, []) if use_c2 else a_by_id.get(image_id, [])
        output.extend(records)
        if use_c2:
            selected_detection_count += len(records)
        else:
            original_detection_count += len(records)

    write_json_atomic(output_path, output, compact=True)

    # UAVDT must remain original-only for every budget.
    for image_id, row in feature_by_id.items():
        if row["source"] == "uavdt":
            if prediction_signature(a_by_id.get(image_id, [])) != prediction_signature(c2_by_id.get(image_id, [])):
                raise RuntimeError(f"UAVDT原图预测一致性检查失败：image_id={image_id}")

    info = {
        "path": str(output_path.relative_to(ROOT)),
        "sha256": sha256_file(output_path),
        "prediction_count": len(output),
        "selected_visdrone_images": len(selected_vis_ids),
        "visdrone_images": EXPECTED_COUNTS["visdrone"],
        "actual_trigger_rate": len(selected_vis_ids) / EXPECTED_COUNTS["visdrone"],
        "uavdt_images_using_original": EXPECTED_COUNTS["uavdt"],
        "detections_from_selected_c2_images": selected_detection_count,
        "detections_from_original_path_images": original_detection_count,
    }
    print(
        f"[{budget_name}] prediction JSON saved: {output_path}; "
        f"trigger={len(selected_vis_ids)}/247 ({info['actual_trigger_rate']:.1%}); "
        f"detections={len(output)}"
    )
    del output
    gc.collect()
    return output_path, info


def evaluate_prediction_file(
    coco_gt,
    category_ids: list[int],
    prediction_path: Path,
    image_ids: list[int],
    scope_name: str,
) -> dict[str, float]:
    print(f"\nCOCO evaluation: {prediction_path.name} | {scope_name} | images={len(image_ids)}")
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
    evaluator.params.areaRng = [area.copy() for area in EVAL_AREA_RANGES]
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
    for output_key, input_key in key_map.items():
        if input_key not in stats:
            raise RuntimeError(f"评价器缺少指标{input_key}，实际字段：{list(stats.keys())}")
        result[output_key] = float(stats[input_key])
    del evaluator, coco_dt
    gc.collect()
    return result


def metric_delta(current: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    return {key: float(current[key] - baseline[key]) for key in METRIC_KEYS}


# ============================================================================
# 9. Runtime recomposition from Stage 6.3D-B measured per-image components
# ============================================================================
def estimate_budget_runtime_from_components(
    budget_name: str,
    selected_vis_ids: set[int],
    per_image_timings: dict[int, dict[str, dict[int, dict[str, Any]]]],
    runtime_summary: dict[str, Any],
) -> dict[str, Any]:
    """Recompose each budget's runtime from Stage 6.3D-B measured image components.

    Read/original costs come from A_Original rows; tile/fusion costs come from
    C2_area2048 rows but are included only for this budget's selected VisDrone
    image IDs. Feature cost is copied from the measured live feature pass.
    For B50/B70, use the matching Medium/Loose measured rule cost; for other
    budgets, use the per-image mean of the existing Loose and Medium rule costs.
    """
    repeat_rows = []
    repeat_numbers = sorted(per_image_timings)
    if not repeat_numbers:
        raise RuntimeError("没有可用于预算耗时重组的Stage 6.3D-B逐图数据。")

    if budget_name == "B50":
        rule_policy = "Medium"
        rule_method = "matching_measured_Medium_rule_ms"
        runtime_kind = "reconstructed_existing_measured_policy"
    elif budget_name == "B70":
        rule_policy = "Loose"
        rule_method = "matching_measured_Loose_rule_ms"
        runtime_kind = "reconstructed_existing_measured_policy"
    else:
        rule_policy = None
        rule_method = "mean_of_measured_Loose_and_Medium_rule_ms"
        runtime_kind = "estimated_from_existing_measured_components"

    for repeat in repeat_numbers:
        tables = per_image_timings[repeat]
        a_rows = tables["A_Original"]
        c2_rows = tables["C2_area2048"]
        loose_rows = tables["Loose"]
        medium_rows = tables["Medium"]

        component_ms = {name: 0.0 for name in ("read", "original", "feature", "rule", "tile", "fusion")}
        image_latencies_ms = []

        for image_id in sorted(a_rows):
            a = a_rows[image_id]
            source = a["source"]
            latency = a["read_ms"] + a["original_ms"]
            component_ms["read"] += a["read_ms"]
            component_ms["original"] += a["original_ms"]

            if source == "visdrone":
                loose = loose_rows[image_id]
                medium = medium_rows[image_id]
                feature_ms = 0.5 * (loose["feature_ms"] + medium["feature_ms"])
                if rule_policy == "Loose":
                    rule_ms = loose["rule_ms"]
                elif rule_policy == "Medium":
                    rule_ms = medium["rule_ms"]
                else:
                    rule_ms = 0.5 * (loose["rule_ms"] + medium["rule_ms"])
                latency += feature_ms + rule_ms
                component_ms["feature"] += feature_ms
                component_ms["rule"] += rule_ms

                if image_id in selected_vis_ids:
                    c2 = c2_rows[image_id]
                    tile_ms = c2["tile_ms"]
                    fusion_ms = c2["fusion_ms"]
                    if tile_ms <= 0 or fusion_ms < 0:
                        raise RuntimeError(
                            f"被选中的VisDrone图像缺少Stage 6.3D-B切片耗时：repeat={repeat}, image_id={image_id}"
                        )
                    latency += tile_ms + fusion_ms
                    component_ms["tile"] += tile_ms
                    component_ms["fusion"] += fusion_ms
            elif image_id in selected_vis_ids:
                raise RuntimeError(f"预算选择器错误地选择了UAVDT图像：image_id={image_id}")

            image_latencies_ms.append(latency)

        total_seconds = float(sum(image_latencies_ms) / 1000.0)
        mean_image_ms = float(np.mean(image_latencies_ms))
        p95_image_ms = float(np.percentile(image_latencies_ms, 95))
        repeat_rows.append({
            "repeat": repeat,
            "total_seconds": total_seconds,
            "images_per_second": EXPECTED_TOTAL / total_seconds if total_seconds else 0.0,
            "mean_image_ms": mean_image_ms,
            "p95_image_ms": p95_image_ms,
            "component_seconds": {key: value / 1000.0 for key, value in component_ms.items()},
        })

    total_values = np.asarray([row["total_seconds"] for row in repeat_rows], dtype=np.float64)
    fps_values = np.asarray([row["images_per_second"] for row in repeat_rows], dtype=np.float64)
    mean_values = np.asarray([row["mean_image_ms"] for row in repeat_rows], dtype=np.float64)
    p95_values = np.asarray([row["p95_image_ms"] for row in repeat_rows], dtype=np.float64)
    component_means = {
        key: float(np.mean([row["component_seconds"][key] for row in repeat_rows]))
        for key in ("read", "original", "feature", "rule", "tile", "fusion")
    }
    measured_c2_seconds = float(runtime_summary["policies"]["C2_area2048"]["total_seconds_mean"])

    result = {
        "runtime_type": runtime_kind,
        "runtime_source": "Stage 6.3D-B stage6_3d_runtime_per_image.csv; no new inference was run",
        "repeats": len(repeat_rows),
        "rule_time_method": rule_method,
        "selected_visdrone_images": len(selected_vis_ids),
        "estimated_total_seconds_mean": float(np.mean(total_values)),
        "estimated_total_seconds_std": float(np.std(total_values, ddof=1)) if len(total_values) > 1 else 0.0,
        "estimated_mean_image_ms": float(np.mean(mean_values)),
        "estimated_p95_image_ms": float(np.mean(p95_values)),
        "estimated_images_per_second": float(np.mean(fps_values)),
        "estimated_speedup_vs_C2": measured_c2_seconds / float(np.mean(total_values)),
        "estimated_time_reduction_vs_C2_percent": (1.0 - float(np.mean(total_values)) / measured_c2_seconds) * 100.0,
        "component_seconds_mean": component_means,
        "repeat_results": repeat_rows,
    }

    # B50/B70 are exact replays of the already measured Medium/Loose policies.
    if budget_name in ("B50", "B70"):
        matching_policy = "Medium" if budget_name == "B50" else "Loose"
        measured = runtime_summary["policies"][matching_policy]
        difference = result["estimated_total_seconds_mean"] - float(measured["total_seconds_mean"])
        result["stage6_3d_b_runtime_reproduction"] = {
            "matched_policy": matching_policy,
            "measured_total_seconds_mean": float(measured["total_seconds_mean"]),
            "reconstructed_total_seconds_mean": result["estimated_total_seconds_mean"],
            "difference_seconds": difference,
            "passed": abs(difference) <= 1e-4,
        }
        if abs(difference) > 1e-4:
            raise RuntimeError(
                f"{budget_name}耗时重组未能复现Stage 6.3D-B {matching_policy}："
                f"difference={difference:.8f}s"
            )
    return result


# ============================================================================
# 10. Reproduction checks, metrics table and final summary
# ============================================================================
def check_c3_reproduction(
    budget_name: str,
    prediction_info: dict[str, Any],
    metrics: dict[str, Any],
    c3_summary: dict[str, Any],
) -> dict[str, Any] | None:
    c3_policy = C3_POLICY_FOR_BUDGET.get(budget_name)
    if c3_policy is None:
        return None

    c3_policy_result = c3_summary["policies"][c3_policy]
    expected_prediction_sha = c3_policy_result["prediction"]["sha256"]
    if prediction_info["sha256"] != expected_prediction_sha:
        raise RuntimeError(
            f"{budget_name}预测文件不能复现Stage 6.3C {c3_policy}："
            f"expected_sha={expected_prediction_sha}, actual_sha={prediction_info['sha256']}"
        )

    comparisons = {}
    for scope in ("full", "visdrone"):
        expected_metrics = c3_policy_result["coco_metrics"][scope]
        actual_metrics = metrics[scope]
        for key in METRIC_KEYS:
            difference = abs(float(actual_metrics[key]) - float(expected_metrics[key]))
            if difference > 1e-9:
                raise RuntimeError(
                    f"{budget_name} {scope} {key}不能复现Stage 6.3C {c3_policy}："
                    f"difference={difference}"
                )
        comparisons[scope] = "passed"
    return {
        "matched_stage6_3c_policy": c3_policy,
        "prediction_sha256_match": True,
        "coco_metric_reproduction": comparisons,
    }


def build_metrics_row(
    budget_name: str,
    budget_fraction: float,
    diagnostics: dict[str, Any],
    full_metrics: dict[str, float],
    vis_metrics: dict[str, float],
    delta_full_vs_a: dict[str, float],
    delta_full_vs_c2: dict[str, float],
    runtime_estimate: dict[str, float],
) -> dict[str, Any]:
    return {
        "budget": budget_name,
        "max_trigger_budget_percent": int(round(budget_fraction * 100)),
        "selected_visdrone_images": diagnostics["selected_images"],
        "visdrone_images": EXPECTED_COUNTS["visdrone"],
        "actual_trigger_rate_percent": diagnostics["trigger_rate"] * 100.0,
        "beneficial_selected": diagnostics["beneficial_selected"],
        "harmful_selected": diagnostics["harmful_selected"],
        "beneficial_precision_percent": diagnostics["beneficial_precision_among_selected"] * 100.0,
        "beneficial_recall_percent": diagnostics["beneficial_recall"] * 100.0,
        "net_label_utility": diagnostics["net_label_utility"],
        "full_AP50_95": full_metrics["AP50-95"],
        "full_AP50": full_metrics["AP50"],
        "full_APs": full_metrics["APs"],
        "full_APm": full_metrics["APm"],
        "full_APl": full_metrics["APl"],
        "visdrone_AP50_95": vis_metrics["AP50-95"],
        "visdrone_AP50": vis_metrics["AP50"],
        "visdrone_APs": vis_metrics["APs"],
        "visdrone_APm": vis_metrics["APm"],
        "visdrone_APl": vis_metrics["APl"],
        "delta_full_AP50_95_vs_A": delta_full_vs_a["AP50-95"],
        "delta_full_APs_vs_A": delta_full_vs_a["APs"],
        "delta_full_AP50_95_vs_C2": delta_full_vs_c2["AP50-95"],
        "delta_full_APs_vs_C2": delta_full_vs_c2["APs"],
        "estimated_total_seconds": runtime_estimate["estimated_total_seconds_mean"],
        "estimated_mean_image_ms": runtime_estimate["estimated_mean_image_ms"],
        "estimated_p95_image_ms": runtime_estimate["estimated_p95_image_ms"],
        "estimated_images_per_second": runtime_estimate["estimated_images_per_second"],
        "estimated_time_reduction_vs_C2_percent": runtime_estimate["estimated_time_reduction_vs_C2_percent"],
        "estimated_speedup_vs_C2": runtime_estimate["estimated_speedup_vs_C2"],
    }


def get_bootstrap_reference(d3_bootstrap: dict[str, Any]) -> dict[str, Any]:
    distributions = d3_bootstrap.get("metric_distributions", {})
    result: dict[str, Any] = {}
    for policy in ("Loose", "Medium"):
        result[policy] = {}
        for metric_key in ("full_AP50-95", "full_APs", "visdrone_AP50-95", "visdrone_APs"):
            source_key = f"{policy}_{metric_key}"
            if source_key in distributions:
                result[policy][metric_key] = distributions[source_key]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 6.3E budget sensitivity sweep")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="仅覆盖Stage 6.3E独立输出目录中的同名文件；绝不修改Stage 6.3C/6.3D。",
    )
    args = parser.parse_args()

    print("=" * 88)
    print("Stage 6.3E | Budget Sweep B30-B90")
    print("No YOLO training. No YOLO inference. No feature expansion. No classifier change.")
    print("=" * 88)

    # Prevent accidental replacement of a previous Stage 6.3E run.
    if OUT_DIR.exists() and any(OUT_DIR.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Stage 6.3E输出目录已经有文件：{OUT_DIR}\n"
            "如确实需要重跑并覆盖Stage 6.3E文件，请显式添加 --overwrite。"
        )

    common = check_common_inputs()
    feature_by_id, data_info = load_feature_and_utility_rows()
    source_by_id = load_and_check_mapping(feature_by_id)
    valid_ids = set(feature_by_id)

    print("\n[1/7] 输入文件、哈希、环境与Stage 6.3C/6.3D协议检查通过")
    print("  images:", len(feature_by_id))
    print("  source counts:", data_info["source_counts"])
    print("  VisDrone labels:", data_info["label_counts"])
    print("  environment:", common["environment"])

    print("\n[2/7] 读取既有Original与C2-2048预测，不运行模型推理")
    a_by_id, a_count = load_prediction_groups(A_JSON, valid_ids)
    c2_by_id, c2_count = load_prediction_groups(C2_JSON, valid_ids)
    uavdt_check = check_uavdt_original_only(feature_by_id, a_by_id, c2_by_id)
    print("  A_Original detections:", a_count)
    print("  C2_area2048 detections:", c2_count)
    print("  UAVDT original-only check:", uavdt_check)

    print("\n[3/7] 读取Stage 6.3C OOF决策，并检查B30/B50/B70复现基线")
    c3_assignments = read_c3_assignments()
    runtime_timings = load_stage6_3d_per_image_timings(
        source_by_id, feature_by_id, c3_assignments, common["d3_runtime"]
    )

    print("\n[4/7] 执行7档预算的5折GroupKFold OOF规则学习")
    selected_by_budget, fold_rules, assignment_rows = run_groupkfold_budget_sweep(
        feature_by_id, c3_assignments
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv_atomic(OUT_ASSIGNMENTS, assignment_rows)
    write_json_atomic(OUT_DIR / "budget_sweep_fold_rules.json", {
        "experiment": "Stage6.3E_Budget_Sensitivity_Sweep",
        "splitter": "GroupKFold",
        "n_splits": N_SPLITS,
        "group_key": "sequence_id",
        "feature_columns": FEATURE_COLUMNS,
        "max_rule_conditions": MAX_RULE_CONDITIONS,
        "rule_logic": "AND",
        "quantile_grid": list(QUANTILE_GRID),
        "budgets": BUDGETS,
        "fold_rules": fold_rules,
    })

    print("\n[5/7] 生成7组OOF组合预测文件")
    prediction_paths: dict[str, Path] = {}
    prediction_info: dict[str, dict[str, Any]] = {}
    for budget_name in BUDGETS:
        path, info = write_budget_prediction(
            budget_name,
            selected_by_budget[budget_name],
            feature_by_id,
            a_by_id,
            c2_by_id,
        )
        prediction_paths[budget_name] = path
        prediction_info[budget_name] = info

    print("\n[6/7] 使用faster-coco-eval 1.8.0执行完整集与VisDrone子集评价")
    if version("faster-coco-eval") != EXPECTED_EVALUATOR_VERSION:
        raise RuntimeError("faster-coco-eval版本在运行期间发生变化。")
    coco_gt = COCO(str(GT_JSON))
    gt_image_ids = sorted(int(item["id"]) for item in coco_gt.dataset["images"])
    if gt_image_ids != sorted(valid_ids):
        raise RuntimeError("COCO GT image IDs与特征CSV不一致。")
    category_ids = sorted(int(x) for x in coco_gt.getCatIds())
    if not category_ids:
        raise RuntimeError("COCO GT没有类别。")

    full_ids = sorted(valid_ids)
    vis_ids = sorted(image_id for image_id, source in source_by_id.items() if source == "visdrone")
    reference_metrics = common["c3_summary"]["reference_metrics"]
    baseline_a_full = reference_metrics["A_Original"]["full"]
    baseline_c2_full = reference_metrics["C2_area2048"]["full"]

    all_budget_results: dict[str, Any] = {}
    metric_rows: list[dict[str, Any]] = []

    for budget_name, budget_fraction in BUDGETS.items():
        print(f"\n--- Evaluate {budget_name} ({budget_fraction:.0%}) ---")
        prediction_path = prediction_paths[budget_name]
        full_metrics = evaluate_prediction_file(
            coco_gt, category_ids, prediction_path, full_ids, "full_1567"
        )
        vis_metrics = evaluate_prediction_file(
            coco_gt, category_ids, prediction_path, vis_ids, "visdrone_247"
        )

        # Held-out labels are used only for post-hoc diagnostics, never for fitting.
        budget_assignment_rows = [row for row in assignment_rows if row["budget"] == budget_name]
        budget_assignment_rows.sort(key=lambda row: int(row["image_id"]))
        selected_mask = np.asarray(
            [bool(int(row["selected_for_c2_2048"])) for row in budget_assignment_rows],
            dtype=bool,
        )
        diagnostic_rows = [feature_by_id[int(row["image_id"])] for row in budget_assignment_rows]
        diagnostics = evaluate_rule_diagnostics(diagnostic_rows, selected_mask)

        delta_full_vs_a = metric_delta(full_metrics, baseline_a_full)
        delta_full_vs_c2 = metric_delta(full_metrics, baseline_c2_full)
        runtime_estimate = estimate_budget_runtime_from_components(
            budget_name, selected_by_budget[budget_name], runtime_timings, common["d3_runtime"]
        )
        metric_bundle = {
            "full": full_metrics,
            "visdrone": vis_metrics,
            "delta_full_vs_A_Original": delta_full_vs_a,
            "delta_full_vs_C2_area2048": delta_full_vs_c2,
        }
        c3_reproduction = check_c3_reproduction(
            budget_name, prediction_info[budget_name], metric_bundle, common["c3_summary"]
        )

        all_budget_results[budget_name] = {
            "max_trigger_budget_fraction": budget_fraction,
            "max_trigger_budget_percent": int(round(budget_fraction * 100)),
            "oof_selection_diagnostics": diagnostics,
            "prediction": prediction_info[budget_name],
            "coco_metrics": metric_bundle,
            "runtime_estimate": runtime_estimate,
            "stage6_3c_reproduction_check": c3_reproduction,
            "fold_rules": fold_rules[budget_name],
        }
        metric_rows.append(build_metrics_row(
            budget_name,
            budget_fraction,
            diagnostics,
            full_metrics,
            vis_metrics,
            delta_full_vs_a,
            delta_full_vs_c2,
            runtime_estimate,
        ))
        print(
            f"[{budget_name}] actual_trigger={diagnostics['selected_images']}/247 "
            f"({diagnostics['trigger_rate']:.1%}); "
            f"Full AP50-95={full_metrics['AP50-95']:.6f}; "
            f"Full APs={full_metrics['APs']:.6f}; "
            f"VisDrone APs={vis_metrics['APs']:.6f}; "
            f"estimated_runtime={runtime_estimate['estimated_total_seconds_mean']:.3f}s"
        )

    write_csv_atomic(OUT_METRICS, metric_rows)

    d3_bootstrap = common["d3_bootstrap"]
    summary = {
        "experiment": "Stage6.3E_Budget_Sensitivity_Sweep",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "completed",
        "purpose": (
            "Test whether changing only the maximum training-fold tile-trigger budget "
            "improves the precision-efficiency operating point of the existing scene selector."
        ),
        "strict_scope": {
            "no_yolo_training": True,
            "no_yolo_inference": True,
            "no_feature_expansion": True,
            "no_classifier_change": True,
            "no_dataset_or_split_change": True,
            "stage6_3c_files_modified": False,
            "stage6_3d_files_modified": False,
            "validation_development_data_only": True,
            "test_set_used": False,
        },
        "protocol": {
            "images": EXPECTED_TOTAL,
            "source_counts": data_info["source_counts"],
            "visdrone_label_counts": data_info["label_counts"],
            "splitter": "GroupKFold",
            "n_splits": N_SPLITS,
            "group_key": "sequence_id",
            "number_of_visdrone_sequence_groups": 94,
            "feature_columns": FEATURE_COLUMNS,
            "feature_directions": FEATURE_SPECS,
            "max_rule_conditions": MAX_RULE_CONDITIONS,
            "rule_logic": "AND",
            "quantile_grid": list(QUANTILE_GRID),
            "minimum_trigger_fraction": MIN_TRIGGER_FRACTION,
            "minimum_trigger_absolute": MIN_TRIGGER_ABSOLUTE,
            "budgets": BUDGETS,
            "budget_semantics": "maximum trigger fraction on each training fold; not the held-out actual trigger rate",
            "target": "Stage 6.3B2 selector_label: beneficial=+1, harmful=-1, neutral=0",
            "prediction_policy": "UAVDT always A_Original; VisDrone uses C2_area2048 only when selected by its held-out fold rule",
            "evaluator": "faster_coco_eval",
            "evaluator_version": EXPECTED_EVALUATOR_VERSION,
            "imgsz": 640,
            "conf": 0.001,
            "nms_iou": 0.7,
            "max_det": 300,
            "iou_thresholds": [float(x) for x in EVAL_IOU_THRESHOLDS],
            "max_dets": EVAL_MAX_DETS,
            "area_ranges": EVAL_AREA_RANGES,
            "area_labels": EVAL_AREA_LABELS,
        },
        "input_sha256": {
            **common["input_sha256"],
            **common["additional_input_sha256"],
        },
        "read_only_reference_stages": {
            "stage6_3c": {
                "summary_status": common["c3_summary"]["status"],
                "summary_sha256": common["additional_input_sha256"]["c3_summary_json"],
                "existing_oof_prediction_sha256": {
                    name: common["c3_summary"]["policies"][name]["prediction"]["sha256"]
                    for name in ("Loose", "Medium", "Strict")
                },
            },
            "stage6_3d_a_bootstrap": {
                "status": d3_bootstrap["status"],
                "bootstrap": d3_bootstrap["bootstrap"],
                "interpretation_limit": d3_bootstrap.get("interpretation_limit"),
                "existing_policy_ci_reference": get_bootstrap_reference(d3_bootstrap),
            },
            "stage6_3d_b_runtime": {
                "status": common["d3_runtime"]["status"],
                "hardware": common["d3_runtime"].get("hardware", {}),
                "measured_policy_results": common["d3_runtime"]["policies"],
                "runtime_estimation_method": (
                    "Recompose per-image read/original/feature/rule/tile/fusion components from "
                    "Stage 6.3D-B's 3-repeat CSV; for B50/B70 use the matching measured rule-cost "
                    "series, for other budgets use mean Loose/Medium rule cost. No new inference."
                ),
                "runtime_per_image_csv_sha256": common["additional_input_sha256"]["d3_runtime_per_image_csv"],
            },
        },
        "uavdt_original_only_check": uavdt_check,
        "stage6_3c_reproduction_mapping": C3_POLICY_FOR_BUDGET,
        "budgets": all_budget_results,
        "outputs": {
            "summary_json": str(OUT_SUMMARY.relative_to(ROOT)),
            "metrics_csv": str(OUT_METRICS.relative_to(ROOT)),
            "oof_assignments_csv": str(OUT_ASSIGNMENTS.relative_to(ROOT)),
            "fold_rules_json": str((OUT_DIR / "budget_sweep_fold_rules.json").relative_to(ROOT)),
            "prediction_jsons": {
                name: str(path.relative_to(ROOT)) for name, path in prediction_paths.items()
            },
        },
        "environment": common["environment"],
    }
    write_json_atomic(OUT_SUMMARY, summary)

    print("\n" + "=" * 88)
    print("Stage 6.3E completed.")
    print("No YOLO training or inference was performed.")
    print("All outputs are isolated under:", OUT_DIR)
    print(" -", OUT_SUMMARY)
    print(" -", OUT_METRICS)
    print(" -", OUT_ASSIGNMENTS)
    print(" -", OUT_DIR / "budget_sweep_fold_rules.json")
    print(" - budget_30_predictions.json ... budget_90_predictions.json")
    print("=" * 88)


if __name__ == "__main__":
    main()
