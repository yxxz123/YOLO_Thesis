#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 6.3D: sequence-cluster bootstrap + real inference-cost benchmark.

Modes
-----
bootstrap : paired sequence-cluster bootstrap using committed COCO predictions.
benchmark : time the exact Original / C2_area2048 / Loose / Medium inference paths.

The script does NOT train a model and does NOT modify existing Stage 6.3C outputs.
Run from the YOLO_Thesis repository root.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np


# ============================================================================
# 1. Fixed repository paths and evaluation protocol
# ============================================================================
ROOT = Path(__file__).resolve().parent
SCENE_DIR = ROOT / "runs/detect/stage6_scene_selector"
ORACLE_DIR = ROOT / "runs/detect/stage6_selective_oracle"
C2_DIR = ROOT / "runs/detect/stage6_size_aware_fusion_iou07"
GT_JSON = ROOT / "runs/detect/source_validation/Stage5-Baseline-Unified/annotations_auto_detect_val.json"
FEATURE_CSV = SCENE_DIR / "corrected_scene_features.csv"
ASSIGNMENTS_CSV = SCENE_DIR / "selector_oof_assignments.csv"
OOF_SUMMARY_JSON = SCENE_DIR / "selector_oof_summary.json"
REFERENCE_JSON = C2_DIR / "faster_eval_metrics.json"

PREDICTION_FILES = {
    "A_Original": ORACLE_DIR / "all_original_predictions.json",
    "C2_area2048": C2_DIR / "C2_area2048_predictions.json",
    "Loose": SCENE_DIR / "selector_oof_predictions_loose.json",
    "Medium": SCENE_DIR / "selector_oof_predictions_medium.json",
}

EXPECTED_SOURCE_COUNTS = {"uavdt": 1320, "visdrone": 247}
EXPECTED_TOTAL = 1567
EXPECTED_EVALUATOR_VERSION = "1.8.0"

IMGSZ = 640
CONF = 0.001
NMS_IOU = 0.7
MAX_DET = 300
OVERLAP = 0.15
AREA_THRESHOLD = 2048.0
HALF = False
AUGMENT = False

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
METRIC_MAP = {
    "AP50-95": "AP_all",
    "AP50": "AP_50",
    "APs": "AP_small",
    "APm": "AP_medium",
    "APl": "AP_large",
}
BOOTSTRAP_METRICS = ["AP50-95", "APs"]
BOOTSTRAP_SCOPES = ["full", "visdrone"]


# ============================================================================
# 2. Common utilities and integrity checks
# ============================================================================
def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_files(paths: list[Path]) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("缺少必要文件：\n" + "\n".join(missing))


def read_feature_rows() -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with FEATURE_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        for raw in csv.DictReader(f):
            image_id = int(raw["image_id"])
            rows[image_id] = {
                **raw,
                "image_id": image_id,
                "source": raw["source"].strip().lower(),
                "sequence_id": raw["sequence_id"].strip(),
            }
    return rows


def check_common_inputs(require_predictions: bool = True) -> dict[str, Any]:
    paths = [FEATURE_CSV, GT_JSON, OOF_SUMMARY_JSON, ASSIGNMENTS_CSV, REFERENCE_JSON]
    if require_predictions:
        paths.extend(PREDICTION_FILES.values())
    require_files(paths)

    summary = load_json(OOF_SUMMARY_JSON)
    if summary.get("status") != "completed":
        raise RuntimeError("Stage 6.3C summary不是completed状态。")

    reference = load_json(REFERENCE_JSON)
    reference_protocol = reference.get("protocol", {})
    expected_reference_protocol = {
        "evaluator_version": EXPECTED_EVALUATOR_VERSION,
        "imgsz": IMGSZ,
        "conf": CONF,
        "nms_iou": NMS_IOU,
        "nms_max_det": MAX_DET,
        "overlap": OVERLAP,
        "half": HALF,
        "augment": AUGMENT,
    }
    for key, expected in expected_reference_protocol.items():
        actual = reference_protocol.get(key)
        if isinstance(expected, float):
            matched = actual is not None and math.isclose(float(actual), expected, abs_tol=1e-9)
        else:
            matched = actual == expected
        if not matched:
            raise RuntimeError(
                f"Stage 6参考评价协议不匹配：{key}，expected={expected}，actual={actual}。"
            )

    expected_hashes = summary.get("input_sha256", {})
    hash_paths = {
        "features": FEATURE_CSV,
        "original": PREDICTION_FILES["A_Original"],
        "c2": PREDICTION_FILES["C2_area2048"],
        "gt": GT_JSON,
    }
    actual_hashes = {}
    for key, path in hash_paths.items():
        actual = sha256_file(path)
        actual_hashes[key] = actual
        expected = expected_hashes.get(key)
        if expected and actual != expected:
            raise RuntimeError(
                f"Stage 6.3C输入哈希不匹配：{key}\n"
                f"expected={expected}\nactual={actual}\n"
                "请先核对当前工作区与已审核的Stage 6.3C文件。"
            )

    ref_pred_hashes = reference.get("prediction_sha256", {})
    for ref_key, local_key in (("A_Original", "original"), ("C2_area2048", "c2")):
        if ref_pred_hashes.get(ref_key) != actual_hashes[local_key]:
            raise RuntimeError(f"faster_eval_metrics.json中的{ref_key}预测哈希与本地文件不一致。")
    if reference.get("ground_truth", {}).get("sha256") != actual_hashes["gt"]:
        raise RuntimeError("faster_eval_metrics.json中的GT哈希与本地GT不一致。")

    features = read_feature_rows()
    if len(features) != EXPECTED_TOTAL:
        raise RuntimeError(f"特征表应有{EXPECTED_TOTAL}行，实际{len(features)}行。")
    source_counts = Counter(r["source"] for r in features.values())
    if dict(source_counts) != EXPECTED_SOURCE_COUNTS:
        raise RuntimeError(f"来源数量异常：{dict(source_counts)}")

    gt = load_json(GT_JSON)
    gt_ids = {int(x["id"]) for x in gt["images"]}
    if gt_ids != set(features):
        raise RuntimeError("COCO GT image_id与corrected_scene_features.csv不一致。")

    protocol = summary.get("protocol", {})
    if protocol.get("n_images") != EXPECTED_TOTAL:
        raise RuntimeError("Stage 6.3C记录的图片总数与当前协议不一致。")
    if protocol.get("group_key") != "sequence_id":
        raise RuntimeError("Stage 6.3C并非按sequence_id进行分组。")
    if int(protocol.get("n_splits", -1)) != 5:
        raise RuntimeError("Stage 6.3C不是5折GroupKFold结果。")
    if int(protocol.get("imgsz", -1)) != IMGSZ:
        raise RuntimeError("Stage 6.3C imgsz不是640。")
    if not math.isclose(float(protocol.get("nms_iou", -1)), NMS_IOU, abs_tol=1e-9):
        raise RuntimeError("Stage 6.3C NMS IoU不是0.7。")

    return {
        "summary": summary,
        "features": features,
        "gt": gt,
        "actual_hashes": actual_hashes,
        "source_counts": dict(source_counts),
    }


def read_assignments() -> dict[str, dict[int, dict[str, Any]]]:
    assignments: dict[str, dict[int, dict[str, Any]]] = {
        "Loose": {}, "Medium": {}, "Strict": {}
    }
    with ASSIGNMENTS_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        for raw in csv.DictReader(f):
            policy = raw["policy"]
            image_id = int(raw["image_id"])
            if policy not in assignments:
                raise RuntimeError(f"未知策略：{policy}")
            if image_id in assignments[policy]:
                raise RuntimeError(f"{policy}中image_id重复：{image_id}")
            raw["image_id"] = image_id
            raw["selected_for_c2_2048"] = bool(int(raw["selected_for_c2_2048"]))
            assignments[policy][image_id] = raw

    vis_ids = {
        image_id for image_id, row in read_feature_rows().items()
        if row["source"] == "visdrone"
    }
    for policy in ("Loose", "Medium", "Strict"):
        if set(assignments[policy]) != vis_ids:
            raise RuntimeError(
                f"{policy} OOF assignment必须且只能覆盖247张VisDrone图片。"
            )
    return assignments


def load_prediction_index(path: Path) -> dict[int, list[dict[str, Any]]]:
    data = load_json(path)
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in data:
        grouped[int(item["image_id"])].append(item)
    return dict(grouped)


def check_evaluator_version() -> None:
    try:
        current = version("faster-coco-eval")
    except PackageNotFoundError as exc:
        raise RuntimeError("未安装faster-coco-eval，请安装1.8.0。") from exc
    if current != EXPECTED_EVALUATOR_VERSION:
        raise RuntimeError(
            f"faster-coco-eval版本必须为{EXPECTED_EVALUATOR_VERSION}，当前为{current}。"
        )


# ============================================================================
# 3. Stage 6.3D-A: paired sequence-cluster bootstrap
# ============================================================================
def make_bootstrap_coco_dataset(
    base_gt: dict[str, Any],
    features: dict[int, dict[str, Any]],
    sampled_groups: list[str],
) -> tuple[Any, list[int], list[int], list[tuple[int, int]]]:
    """Keep UAVDT fixed; resample whole VisDrone sequence groups with replacement.

    Each repeated occurrence receives a fresh synthetic image_id and annotation id,
    so repeated clusters are represented as repeated independent bootstrap units.
    """
    from faster_coco_eval import COCO

    image_by_id = {int(x["id"]): x for x in base_gt["images"]}
    anns_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in base_gt["annotations"]:
        anns_by_image[int(ann["image_id"])].append(ann)

    uavdt_ids = sorted(
        image_id for image_id, row in features.items() if row["source"] == "uavdt"
    )
    group_to_ids: dict[str, list[int]] = defaultdict(list)
    for image_id, row in features.items():
        if row["source"] == "visdrone":
            group_to_ids[row["sequence_id"]].append(image_id)
    for ids in group_to_ids.values():
        ids.sort()

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    vis_pairs: list[tuple[int, int]] = []
    full_ids: list[int] = []
    vis_bootstrap_ids: list[int] = []

    # UAVDT is held fixed in every replicate.
    for old_id in uavdt_ids:
        images.append(dict(image_by_id[old_id]))
        full_ids.append(old_id)
        for ann in anns_by_image.get(old_id, []):
            annotations.append(dict(ann))

    next_image_id = max(int(x["id"]) for x in base_gt["images"]) + 1
    next_ann_id = max((int(x["id"]) for x in base_gt["annotations"]), default=0) + 1

    for group in sampled_groups:
        for old_id in group_to_ids[group]:
            new_id = next_image_id
            next_image_id += 1
            new_image = dict(image_by_id[old_id])
            new_image["id"] = new_id
            images.append(new_image)
            full_ids.append(new_id)
            vis_bootstrap_ids.append(new_id)
            vis_pairs.append((old_id, new_id))

            for ann in anns_by_image.get(old_id, []):
                new_ann = dict(ann)
                new_ann["id"] = next_ann_id
                next_ann_id += 1
                new_ann["image_id"] = new_id
                annotations.append(new_ann)

    dataset = {
        "info": dict(base_gt.get("info", {})),
        "licenses": list(base_gt.get("licenses", [])),
        "images": images,
        "annotations": annotations,
        "categories": [dict(x) for x in base_gt["categories"]],
    }
    coco_gt = COCO()
    coco_gt.dataset = dataset
    coco_gt.createIndex()
    return coco_gt, sorted(full_ids), sorted(vis_bootstrap_ids), vis_pairs


def remap_bootstrap_predictions(
    prediction_index: dict[int, list[dict[str, Any]]],
    uavdt_ids: list[int],
    vis_pairs: list[tuple[int, int]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for image_id in uavdt_ids:
        output.extend(dict(x) for x in prediction_index.get(image_id, []))
    for old_id, new_id in vis_pairs:
        for item in prediction_index.get(old_id, []):
            new_item = dict(item)
            new_item["image_id"] = new_id
            output.append(new_item)
    return output


def evaluate_coco_scope(
    coco_gt: Any,
    predictions: list[dict[str, Any]],
    image_ids: list[int],
    category_ids: list[int],
) -> dict[str, float]:
    from faster_coco_eval import COCOeval_faster

    if not image_ids:
        raise RuntimeError("Bootstrap样本的评价图片列表为空。")
    coco_dt = coco_gt.loadRes(predictions)
    evaluator = COCOeval_faster(
        coco_gt,
        coco_dt,
        iouType="bbox",
        lvis_style=False,
        print_function=lambda *args, **kwargs: None,
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
    result = {name: float(stats[key]) for name, key in METRIC_MAP.items()}
    del evaluator, coco_dt
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def distribution(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "ci95_percentile_low": float(np.percentile(arr, 2.5)),
        "ci95_percentile_high": float(np.percentile(arr, 97.5)),
    }


def run_bootstrap(args: argparse.Namespace) -> None:
    check_evaluator_version()
    common = check_common_inputs(require_predictions=True)
    features = common["features"]
    base_gt = common["gt"]
    assignments = read_assignments()

    group_to_ids: dict[str, list[int]] = defaultdict(list)
    for image_id, row in features.items():
        if row["source"] == "visdrone":
            group_to_ids[row["sequence_id"]].append(image_id)
    group_ids = sorted(group_to_ids)
    if len(group_ids) != 94:
        raise RuntimeError(f"预期94个VisDrone序列组，实际{len(group_ids)}。")

    prediction_indexes = {
        name: load_prediction_index(path)
        for name, path in PREDICTION_FILES.items()
    }
    all_ids = set(features)
    for name, index in prediction_indexes.items():
        extra = set(index) - all_ids
        if extra:
            raise RuntimeError(f"{name}预测文件包含未知image_id：{sorted(extra)[:5]}")

    uavdt_ids = sorted(
        image_id for image_id, row in features.items() if row["source"] == "uavdt"
    )
    category_ids = sorted(int(x["id"]) for x in base_gt["categories"])

    samples_path = SCENE_DIR / "stage6_3d_bootstrap_samples.csv"
    summary_path = SCENE_DIR / "stage6_3d_bootstrap_summary.json"
    existing_rows: list[dict[str, Any]] = []
    completed: set[int] = set()
    if args.resume and samples_path.is_file():
        with samples_path.open("r", encoding="utf-8-sig", newline="") as f:
            existing_rows = list(csv.DictReader(f))
        completed = {int(x["replicate"]) for x in existing_rows}
        print(f"恢复模式：已发现{len(completed)}个已完成replicate。")

    rows = existing_rows
    start_time = time.perf_counter()
    for replicate in range(args.n_bootstrap):
        if replicate in completed:
            continue

        rng = np.random.default_rng(args.seed + replicate)
        sampled_groups = rng.choice(group_ids, size=len(group_ids), replace=True).tolist()
        coco_gt, full_ids, vis_ids, vis_pairs = make_bootstrap_coco_dataset(
            base_gt, features, sampled_groups
        )

        row: dict[str, Any] = {
            "replicate": replicate,
            "sampled_sequence_groups": len(sampled_groups),
            "sampled_visdrone_images": len(vis_ids),
        }
        for policy, pred_index in prediction_indexes.items():
            pred = remap_bootstrap_predictions(pred_index, uavdt_ids, vis_pairs)
            full_metrics = evaluate_coco_scope(coco_gt, pred, full_ids, category_ids)
            vis_metrics = evaluate_coco_scope(coco_gt, pred, vis_ids, category_ids)
            for metric in BOOTSTRAP_METRICS:
                row[f"{policy}_full_{metric}"] = full_metrics[metric]
                row[f"{policy}_visdrone_{metric}"] = vis_metrics[metric]

        # Paired differences: every strategy uses the exact same sampled groups.
        for policy in ("Loose", "Medium", "A_Original"):
            for scope in BOOTSTRAP_SCOPES:
                for metric in BOOTSTRAP_METRICS:
                    key = f"{policy}_{scope}_{metric}"
                    ref = f"C2_area2048_{scope}_{metric}"
                    row[f"delta_{policy}_vs_C2_{scope}_{metric}"] = row[key] - row[ref]
        for scope in BOOTSTRAP_SCOPES:
            for metric in BOOTSTRAP_METRICS:
                row[f"delta_Loose_vs_Medium_{scope}_{metric}"] = (
                    row[f"Loose_{scope}_{metric}"] - row[f"Medium_{scope}_{metric}"]
                )

        rows.append(row)
        completed.add(replicate)
        if len(completed) % args.checkpoint_every == 0 or replicate == args.n_bootstrap - 1:
            rows.sort(key=lambda x: int(x["replicate"]))
            write_csv(samples_path, rows)
            elapsed = time.perf_counter() - start_time
            print(
                f"Bootstrap {len(completed)}/{args.n_bootstrap} | "
                f"本轮VisDrone图片={len(vis_ids)} | elapsed={elapsed / 60:.1f} min"
            )

    rows.sort(key=lambda x: int(x["replicate"]))
    write_csv(samples_path, rows)
    if len(rows) < args.n_bootstrap:
        raise RuntimeError(f"Bootstrap仅完成{len(rows)}/{args.n_bootstrap}次。")

    metric_columns = [
        f"{policy}_{scope}_{metric}"
        for policy in PREDICTION_FILES
        for scope in BOOTSTRAP_SCOPES
        for metric in BOOTSTRAP_METRICS
    ]
    delta_columns = [
        f"delta_{policy}_vs_C2_{scope}_{metric}"
        for policy in ("Loose", "Medium", "A_Original")
        for scope in BOOTSTRAP_SCOPES
        for metric in BOOTSTRAP_METRICS
    ] + [
        f"delta_Loose_vs_Medium_{scope}_{metric}"
        for scope in BOOTSTRAP_SCOPES
        for metric in BOOTSTRAP_METRICS
    ]

    summary: dict[str, Any] = {
        "experiment": "Stage6.3D_A_Paired_VisDrone_Sequence_Cluster_Bootstrap",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "completed",
        "bootstrap": {
            "replicates": args.n_bootstrap,
            "seed": args.seed,
            "resampling_unit": "VisDrone sequence_id cluster",
            "number_of_sequence_groups": len(group_ids),
            "sampling": "sample 94 sequence groups with replacement; include all images in each sampled group",
            "uavdt_handling": "all 1320 UAVDT images held fixed in every replicate",
            "confidence_interval": "percentile 2.5th and 97.5th percentiles",
            "paired_comparison": "all four prediction policies evaluated on the same sampled sequence groups",
        },
        "input_sha256": common["actual_hashes"],
        "interpretation_limit": (
            "This is a conditional cluster bootstrap for VisDrone sequence-sampling uncertainty. "
            "UAVDT is fixed, and the 247 VisDrone development images are resampled; this is not an independent external test."
        ),
        "samples_csv": str(samples_path.relative_to(ROOT)),
        "metric_distributions": {},
        "paired_delta_distributions": {},
    }
    for col in metric_columns:
        summary["metric_distributions"][col] = distribution([float(x[col]) for x in rows])
    for col in delta_columns:
        vals = [float(x[col]) for x in rows]
        summary["paired_delta_distributions"][col] = {
            **distribution(vals),
            "probability_delta_gt_zero": float(np.mean(np.asarray(vals) > 0.0)),
            "probability_delta_ge_minus_0_002": float(np.mean(np.asarray(vals) >= -0.002)),
        }

    # Include the already-published Stage 6.3C point estimates for direct reference.
    summary["stage6_3c_point_estimates"] = {
        policy: common["summary"]["policies"][policy]["coco_metrics"]
        for policy in ("Loose", "Medium", "Strict")
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nStage 6.3D-A完成。")
    print("Bootstrap样本：", samples_path)
    print("汇总报告：", summary_path)


# ============================================================================
# 4. Stage 6.3D-B: real inference-cost benchmark
# ============================================================================
def normalize_name(value: str) -> str:
    return str(value).replace("\\", "/").lstrip("./")


def build_image_index(image_dir: Path) -> tuple[dict[str, Path], dict[str, list[Path]]]:
    files = [
        p for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    ]
    if not files:
        raise RuntimeError(f"验证图片目录为空：{image_dir}")
    by_relative: dict[str, Path] = {}
    by_basename: dict[str, list[Path]] = defaultdict(list)
    for p in files:
        by_relative[normalize_name(p.relative_to(image_dir))] = p
        by_basename[p.name].append(p)
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
    basename = Path(key).name
    matches = by_basename.get(basename, [])
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(f"图片文件名重复，无法唯一定位：{filename}")
    raise FileNotFoundError(f"找不到本地图片：{filename}")


def empty_result() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros((0, 4), dtype=np.float32),
        np.zeros((0,), dtype=np.float32),
        np.zeros((0,), dtype=np.int64),
    )


def make_tiles(image: np.ndarray) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    height, width = image.shape[:2]
    tile_w = min(math.ceil(width / (2 - OVERLAP)), width)
    tile_h = min(math.ceil(height / (2 - OVERLAP)), height)
    xs = sorted(set([0, max(0, width - tile_w)]))
    ys = sorted(set([0, max(0, height - tile_h)]))
    tiles: list[np.ndarray] = []
    offsets: list[tuple[int, int]] = []
    for y in ys:
        for x in xs:
            tiles.append(image[y:y + tile_h, x:x + tile_w].copy())
            offsets.append((x, y))
    return tiles, offsets


def extract_result(result: Any, offset_x: int = 0, offset_y: int = 0):
    if result.boxes is None or len(result.boxes) == 0:
        return empty_result()
    boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    scores = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
    classes = result.boxes.cls.detach().cpu().numpy().astype(np.int64)
    boxes[:, [0, 2]] += offset_x
    boxes[:, [1, 3]] += offset_y
    return boxes, scores, classes


def merge_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    width: int,
    height: int,
    torch: Any,
    batched_nms: Any,
):
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


def predict_original(model: Any, image: np.ndarray, pred_args: dict[str, Any], torch: Any, batched_nms: Any):
    result = model.predict(source=image, batch=1, **pred_args)[0]
    boxes, scores, classes = extract_result(result)
    height, width = image.shape[:2]
    return merge_nms(boxes, scores, classes, width, height, torch, batched_nms)


def predict_tiled(model: Any, image: np.ndarray, pred_args: dict[str, Any], torch: Any, batched_nms: Any):
    tiles, offsets = make_tiles(image)
    results = model.predict(source=tiles, batch=len(tiles), **pred_args)
    if len(results) != len(tiles):
        raise RuntimeError(f"切片数{len(tiles)}与模型返回结果数{len(results)}不一致。")
    boxes_all, scores_all, classes_all = [], [], []
    for result, (offset_x, offset_y) in zip(results, offsets):
        boxes, scores, classes = extract_result(result, offset_x, offset_y)
        if len(boxes):
            boxes_all.append(boxes)
            scores_all.append(scores)
            classes_all.append(classes)
    if not boxes_all:
        return empty_result(), len(tiles)
    boxes = np.concatenate(boxes_all, axis=0)
    scores = np.concatenate(scores_all, axis=0)
    classes = np.concatenate(classes_all, axis=0)
    height, width = image.shape[:2]
    return merge_nms(boxes, scores, classes, width, height, torch, batched_nms), len(tiles)


def fuse_c2_area2048(
    original: tuple[np.ndarray, np.ndarray, np.ndarray],
    tiled: tuple[np.ndarray, np.ndarray, np.ndarray],
    width: int,
    height: int,
    torch: Any,
    batched_nms: Any,
):
    """Replicate slice_val_size_aware_fusion.py C2_area2048 exactly."""
    b0, s0, c0 = original
    b1, s1, c1 = tiled
    if len(b1):
        areas = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
        keep_tile = areas < AREA_THRESHOLD  # strict less-than, as in Stage 6.1
        b1, s1, c1 = b1[keep_tile], s1[keep_tile], c1[keep_tile]
    if len(b1) == 0:
        # Stage 6.1 keeps the original prediction unchanged when no tile box passes.
        return original
    if len(b0) == 0:
        boxes, scores, classes = b1, s1, c1
    else:
        boxes = np.concatenate([b0, b1], axis=0)
        scores = np.concatenate([s0, s1], axis=0)
        classes = np.concatenate([c0, c1], axis=0)
    return merge_nms(boxes, scores, classes, width, height, torch, batched_nms)


def prediction_to_coco(image_id: int, result: tuple[np.ndarray, np.ndarray, np.ndarray], category_id: int):
    boxes, scores, classes = result
    output = []
    for box, score, cls in zip(boxes, scores, classes):
        x1, y1, x2, y2 = [float(x) for x in box]
        output.append({
            "image_id": int(image_id),
            "category_id": int(category_id),
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "score": float(score),
        })
    return output


def prediction_signature(records: list[dict[str, Any]]) -> Counter:
    return Counter(
        (
            int(x["category_id"]),
            round(float(x["score"]), 7),
            tuple(round(float(v), 5) for v in x["bbox"]),
        )
        for x in records
    )


def load_selected_ids(policy: str, assignments: dict[str, dict[int, dict[str, Any]]]) -> set[int]:
    return {
        image_id for image_id, row in assignments[policy].items()
        if row["selected_for_c2_2048"]
    }


def synchronize_if_needed(torch: Any, device: str) -> None:
    if torch.cuda.is_available() and str(device).lower() not in {"cpu", "-1"}:
        torch.cuda.synchronize()


def load_verify_predictions(path: Path, wanted_ids: set[int]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        image_id = int(item["image_id"])
        if image_id in wanted_ids:
            grouped[image_id].append(item)
    return dict(grouped)


def run_benchmark(args: argparse.Namespace) -> None:
    common = check_common_inputs(require_predictions=True)
    features = common["features"]
    assignments = read_assignments()

    # Lazy imports: bootstrap mode does not require Ultralytics or OpenCV.
    try:
        import cv2
        import torch
        from torchvision.ops import batched_nms
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError(
            "推理耗时测试需要opencv-python、torch、torchvision和ultralytics。"
        ) from exc

    weights = Path(args.weights)
    image_dir = Path(args.image_dir)
    require_files([weights, image_dir])
    by_relative, by_basename = build_image_index(image_dir)

    gt = common["gt"]
    gt_images = {int(x["id"]): x for x in gt["images"]}
    records = []
    for image_id in sorted(features):
        info = gt_images[image_id]
        image_path = resolve_image_path(info["file_name"], image_dir, by_relative, by_basename)
        records.append({
            "image_id": image_id,
            "source": features[image_id]["source"],
            "file_name": info["file_name"],
            "image_path": image_path,
            "width": int(info["width"]),
            "height": int(info["height"]),
        })

    # Verify image count and source composition before loading the model.
    if len(records) != EXPECTED_TOTAL:
        raise RuntimeError(f"待测图片应为{EXPECTED_TOTAL}张，实际{len(records)}张。")
    for record in records:
        if not record["image_path"].is_file():
            raise FileNotFoundError(record["image_path"])

    device = str(args.device)
    pred_args = {
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

    print("=" * 72)
    print("Stage 6.3D-B | Real inference-cost benchmark")
    print("weights:", weights)
    print("image_dir:", image_dir)
    print("device:", device)
    print("images:", len(records), "| source:", common["source_counts"])
    print("repeats:", args.repeats, "| warmup:", args.warmup)
    print("=" * 72)

    model = YOLO(str(weights))
    if len(model.names) != 1:
        raise RuntimeError(f"当前模型类别数为{len(model.names)}，与单类别实验不一致。")

    vis_records = [x for x in records if x["source"] == "visdrone"]
    if not vis_records:
        raise RuntimeError("没有VisDrone图片。")

    # Warm-up is excluded from all reported timings.
    warm_image = cv2.imread(str(records[0]["image_path"]))
    warm_vis_image = cv2.imread(str(vis_records[0]["image_path"]))
    if warm_image is None or warm_vis_image is None:
        raise RuntimeError("Warm-up图片读取失败。")
    warm_tiles, _ = make_tiles(warm_vis_image)
    for _ in range(args.warmup):
        synchronize_if_needed(torch, device)
        model.predict(source=warm_image, batch=1, **pred_args)
        synchronize_if_needed(torch, device)
        model.predict(source=warm_tiles, batch=len(warm_tiles), **pred_args)
        synchronize_if_needed(torch, device)
    del warm_image, warm_vis_image, warm_tiles

    loose_ids = load_selected_ids("Loose", assignments)
    medium_ids = load_selected_ids("Medium", assignments)
    vis_ids = {x["image_id"] for x in vis_records}
    if not loose_ids.issubset(vis_ids) or not medium_ids.issubset(vis_ids):
        raise RuntimeError("OOF策略选择了非VisDrone图片。")

    # Optional prediction verification on a fixed sample prevents benchmarking
    # a different checkpoint or a changed preprocessing/fusion implementation.
    verify_ids = set()
    if not args.skip_verify and args.verify_samples > 0:
        verify_ids.update(x["image_id"] for x in records if x["source"] == "uavdt" and len(verify_ids) < args.verify_samples)
        vis_verify = [x["image_id"] for x in vis_records[:args.verify_samples]]
        verify_ids.update(vis_verify)
    verify_a = load_verify_predictions(PREDICTION_FILES["A_Original"], verify_ids) if verify_ids else {}
    verify_c2 = load_verify_predictions(PREDICTION_FILES["C2_area2048"], verify_ids) if verify_ids else {}
    verified_a: set[int] = set()
    verified_c2: set[int] = set()

    per_image_rows: list[dict[str, Any]] = []
    repeat_summaries: list[dict[str, Any]] = []
    category_id = int(gt["categories"][0]["id"])

    for repeat in range(args.repeats):
        print(f"\n[REPEAT {repeat + 1}/{args.repeats}]")
        read_seconds: dict[int, float] = {}
        original_seconds: dict[int, float] = {}
        tile_seconds: dict[int, float] = {}
        fusion_seconds: dict[int, float] = {}
        tile_counts: dict[int, int] = {}

        for index, record in enumerate(records, start=1):
            image_id = record["image_id"]
            t0 = time.perf_counter()
            image = cv2.imread(str(record["image_path"]))
            read_seconds[image_id] = time.perf_counter() - t0
            if image is None:
                raise RuntimeError(f"图片读取失败：{record['image_path']}")
            h, w = image.shape[:2]
            if w != record["width"] or h != record["height"]:
                raise RuntimeError(
                    f"图片尺寸与GT不一致：{record['file_name']}，"
                    f"实际{w}x{h}，GT{record['width']}x{record['height']}"
                )

            synchronize_if_needed(torch, device)
            t0 = time.perf_counter()
            original_result = predict_original(model, image, pred_args, torch, batched_nms)
            synchronize_if_needed(torch, device)
            original_seconds[image_id] = time.perf_counter() - t0

            if image_id in verify_ids and image_id not in verified_a:
                actual = prediction_to_coco(image_id, original_result, category_id)
                expected = verify_a.get(image_id, [])
                if prediction_signature(actual) != prediction_signature(expected):
                    raise RuntimeError(
                        f"原图预测与已提交A_Original不一致：image_id={image_id}。"
                        "请核对best.pt、推理参数和软件环境；本次耗时结果不予采信。"
                    )
                verified_a.add(image_id)

            if record["source"] == "visdrone":
                synchronize_if_needed(torch, device)
                t0 = time.perf_counter()
                tiled_result, n_tiles = predict_tiled(model, image, pred_args, torch, batched_nms)
                synchronize_if_needed(torch, device)
                tile_seconds[image_id] = time.perf_counter() - t0
                tile_counts[image_id] = n_tiles
                if n_tiles != 4:
                    raise RuntimeError(f"image_id={image_id}切片数不是4，而是{n_tiles}。")

                synchronize_if_needed(torch, device)
                t0 = time.perf_counter()
                c2_result = fuse_c2_area2048(
                    original_result, tiled_result, w, h, torch, batched_nms
                )
                synchronize_if_needed(torch, device)
                fusion_seconds[image_id] = time.perf_counter() - t0

                if image_id in verify_ids and image_id not in verified_c2:
                    actual = prediction_to_coco(image_id, c2_result, category_id)
                    expected = verify_c2.get(image_id, [])
                    if prediction_signature(actual) != prediction_signature(expected):
                        raise RuntimeError(
                            f"C2_area2048预测与已提交结果不一致：image_id={image_id}。"
                            "请核对best.pt、切片参数、面积阈值和NMS；本次耗时结果不予采信。"
                        )
                    verified_c2.add(image_id)

            if index % 100 == 0 or index == len(records):
                print(f"  processed {index}/{len(records)}")

        # Reconstruct each strategy's full-validation-set wall-time from the
        # measured per-image components. The Original pass is common to all.
        policy_selected = {
            "A_Original": set(),
            "C2_area2048": vis_ids,
            "Loose": loose_ids,
            "Medium": medium_ids,
        }
        repeat_row: dict[str, Any] = {"repeat": repeat + 1}
        repeat_image_rows = []
        for policy, selected in policy_selected.items():
            image_latency = []
            total_read = sum(read_seconds.values())
            total_original = sum(original_seconds.values())
            total_tile = sum(tile_seconds.get(i, 0.0) for i in selected)
            total_fusion = sum(fusion_seconds.get(i, 0.0) for i in selected)
            total_seconds = total_read + total_original + total_tile + total_fusion
            for record in records:
                image_id = record["image_id"]
                latency = read_seconds[image_id] + original_seconds[image_id]
                if image_id in selected:
                    latency += tile_seconds[image_id] + fusion_seconds[image_id]
                image_latency.append(latency)
                repeat_image_rows.append({
                    "repeat": repeat + 1,
                    "policy": policy,
                    "image_id": image_id,
                    "source": record["source"],
                    "selected_for_tiling": int(image_id in selected),
                    "read_ms": read_seconds[image_id] * 1000.0,
                    "original_ms": original_seconds[image_id] * 1000.0,
                    "tile_ms": tile_seconds.get(image_id, 0.0) * 1000.0 if image_id in selected else 0.0,
                    "fusion_ms": fusion_seconds.get(image_id, 0.0) * 1000.0 if image_id in selected else 0.0,
                    "total_image_ms": latency * 1000.0,
                })
            repeat_row[f"{policy}_total_seconds"] = total_seconds
            repeat_row[f"{policy}_images_per_second"] = len(records) / total_seconds if total_seconds > 0 else 0.0
            repeat_row[f"{policy}_mean_image_ms"] = float(np.mean(image_latency) * 1000.0)
            repeat_row[f"{policy}_p95_image_ms"] = float(np.percentile(image_latency, 95) * 1000.0)
            repeat_row[f"{policy}_selected_visdrone_images"] = len(selected)
            repeat_row[f"{policy}_trigger_rate_visdrone"] = len(selected) / len(vis_ids)
            repeat_row[f"{policy}_tile_crops"] = len(selected) * 4
            repeat_row[f"{policy}_read_seconds"] = total_read
            repeat_row[f"{policy}_original_seconds"] = total_original
            repeat_row[f"{policy}_tile_seconds"] = total_tile
            repeat_row[f"{policy}_fusion_seconds"] = total_fusion
        repeat_summaries.append(repeat_row)
        per_image_rows.extend(repeat_image_rows)

        print("  Repeat totals (seconds):")
        for policy in policy_selected:
            print(f"    {policy:14s}: {repeat_row[f'{policy}_total_seconds']:.3f}s")

    # Aggregate repeats using mean and standard deviation.
    policies = ["A_Original", "C2_area2048", "Loose", "Medium"]
    final_summary: dict[str, Any] = {
        "experiment": "Stage6.3D_B_Real_Inference_Cost_Benchmark",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "completed",
        "weights": str(weights),
        "image_dir": str(image_dir),
        "device": device,
        "hardware": {
            "torch": getattr(torch, "__version__", "unknown"),
            "cuda": getattr(torch.version, "cuda", None),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() and device != "cpu" else "CPU",
        },
        "protocol": {
            "images": EXPECTED_TOTAL,
            "source_counts": common["source_counts"],
            "imgsz": IMGSZ,
            "conf": CONF,
            "nms_iou": NMS_IOU,
            "max_det": MAX_DET,
            "overlap": OVERLAP,
            "tile_layout": "2x2",
            "tile_area_filter": "tile box area < 2048 in original-image pixel coordinates",
            "half": HALF,
            "augment": AUGMENT,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "timing_method": (
                "per-image measured components; policy total = all image reads + all original inference "
                "+ selected VisDrone tile inference + selected C2 fusion postprocess"
            ),
            "fusion_implementation": "same area<2048 filter and global class-aware NMS IoU=0.7 max_det=300 as Stage 6.1",
        },
        "verification": {
            "enabled": not args.skip_verify and args.verify_samples > 0,
            "verified_original_images": len(verified_a),
            "verified_c2_visdrone_images": len(verified_c2),
            "result": (
                "skipped" if args.skip_verify else
                "skipped_no_samples" if args.verify_samples == 0 else
                "passed" if (
                    len(verified_a)
                    == min(args.verify_samples, EXPECTED_SOURCE_COUNTS["uavdt"])
                    + min(args.verify_samples, EXPECTED_SOURCE_COUNTS["visdrone"])
                    and len(verified_c2)
                    == min(args.verify_samples, EXPECTED_SOURCE_COUNTS["visdrone"])
                ) else "not_fully_checked"
            ),
        },
        "policies": {},
        "repeat_results": repeat_summaries,
        "per_image_csv": str((SCENE_DIR / "stage6_3d_runtime_per_image.csv").relative_to(ROOT)),
    }
    for policy in policies:
        totals = np.asarray([x[f"{policy}_total_seconds"] for x in repeat_summaries], dtype=np.float64)
        fps = np.asarray([x[f"{policy}_images_per_second"] for x in repeat_summaries], dtype=np.float64)
        mean_ms = np.asarray([x[f"{policy}_mean_image_ms"] for x in repeat_summaries], dtype=np.float64)
        p95_ms = np.asarray([x[f"{policy}_p95_image_ms"] for x in repeat_summaries], dtype=np.float64)
        final_summary["policies"][policy] = {
            "selected_visdrone_images": repeat_summaries[-1][f"{policy}_selected_visdrone_images"],
            "trigger_rate_visdrone": repeat_summaries[-1][f"{policy}_trigger_rate_visdrone"],
            "tile_crops": repeat_summaries[-1][f"{policy}_tile_crops"],
            "total_seconds_mean": float(np.mean(totals)),
            "total_seconds_std": float(np.std(totals, ddof=1)) if len(totals) > 1 else 0.0,
            "images_per_second_mean": float(np.mean(fps)),
            "mean_image_ms_mean": float(np.mean(mean_ms)),
            "p95_image_ms_mean": float(np.mean(p95_ms)),
            "component_seconds_mean": {
                component: float(np.mean([x[f"{policy}_{component}_seconds"] for x in repeat_summaries]))
                for component in ("read", "original", "tile", "fusion")
            },
        }

    per_image_path = SCENE_DIR / "stage6_3d_runtime_per_image.csv"
    summary_path = SCENE_DIR / "stage6_3d_runtime_summary.json"
    write_csv(per_image_path, per_image_rows)
    summary_path.write_text(json.dumps(final_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nStage 6.3D-B完成。")
    print("逐图耗时：", per_image_path)
    print("耗时汇总：", summary_path)


# ============================================================================
# 5. CLI
# ============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 6.3D paired sequence bootstrap and inference-cost benchmark"
    )
    parser.add_argument("--mode", choices=["bootstrap", "benchmark"], required=True)

    # Bootstrap arguments
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")

    # Runtime benchmark arguments
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/detect/stage2_screening/YOLO11s-DySample/weights/best.pt",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="datasets/HighwayVehicle-v1.0/images/val",
    )
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--verify-samples", type=int, default=20)
    parser.add_argument("--skip-verify", action="store_true")

    args = parser.parse_args()
    if args.n_bootstrap < 1:
        parser.error("--n-bootstrap必须大于0")
    if args.checkpoint_every < 1:
        parser.error("--checkpoint-every必须大于0")
    if args.repeats < 1:
        parser.error("--repeats必须大于0")
    if args.warmup < 0 or args.verify_samples < 0:
        parser.error("--warmup和--verify-samples不能小于0")
    return args


def main() -> None:
    args = parse_args()
    if not ROOT.joinpath("ultralytics").is_dir():
        raise RuntimeError("请将stage6_3d_validation.py放在YOLO_Thesis仓库根目录运行。")
    if args.mode == "bootstrap":
        run_bootstrap(args)
    elif args.mode == "benchmark":
        run_benchmark(args)


if __name__ == "__main__":
    main()
