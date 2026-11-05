#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 6.3F-A: paired VisDrone sequence-cluster bootstrap for M3/M4.

Run from YOLO_Thesis repository root:
    python -u stage6_3f_paired_cluster_bootstrap.py \
        --n-bootstrap 1000 --seed 42 --checkpoint-every 10

Resume an interrupted run:
    python -u stage6_3f_paired_cluster_bootstrap.py \
        --n-bootstrap 1000 --seed 42 --checkpoint-every 10 --resume

This is a conditional bootstrap of already-created OOF prediction policies.
It does NOT retrain YOLO, does NOT run YOLO inference, and does NOT use the
independent test set. UAVDT is held fixed; whole VisDrone sequence groups are
resampled with replacement, exactly following Stage 6.3D-A.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import time
from collections import defaultdict
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from faster_coco_eval import COCO, COCOeval_faster

ROOT = Path(__file__).resolve().parent
SCENE_DIR = ROOT / "runs/detect/stage6_scene_selector"
ORACLE_DIR = ROOT / "runs/detect/stage6_selective_oracle"
C2_DIR = ROOT / "runs/detect/stage6_size_aware_fusion_iou07"
E3A_DIR = SCENE_DIR / "stage6_3e_budget_sweep"
E3B_DIR = SCENE_DIR / "stage6_3e_b_model_comparison"

FEATURE_CSV = SCENE_DIR / "corrected_scene_features.csv"
GT_JSON = ROOT / "runs/detect/source_validation/Stage5-Baseline-Unified/annotations_auto_detect_val.json"
E3A_SUMMARY = E3A_DIR / "budget_sweep_summary.json"
E3B_SUMMARY = E3B_DIR / "model_comparison_summary.json"

PREDICTION_FILES = {
    "A_Original": ORACLE_DIR / "all_original_predictions.json",
    "C2_area2048": C2_DIR / "C2_area2048_predictions.json",
    "B80": E3A_DIR / "budget_80_predictions.json",
    "M3_Logistic_F0": E3B_DIR / "predictions/m3_logistic_f0.json",
    "M4_Logistic_F1": E3B_DIR / "predictions/m4_logistic_f1.json",
}
M1_REPRODUCTION_FILE = E3B_DIR / "predictions/m1_rule_f0.json"

OUT_DIR = SCENE_DIR / "stage6_3f_paired_cluster_bootstrap"
OUT_SAMPLES = OUT_DIR / "paired_cluster_bootstrap_samples.csv"
OUT_SUMMARY = OUT_DIR / "paired_cluster_bootstrap_summary.json"

EXPECTED_EVALUATOR_VERSION = "1.8.0"
EXPECTED_NUMPY_VERSION = "1.26.4"
EXPECTED_TOTAL = 1567
EXPECTED_SOURCE_COUNTS = {"uavdt": 1320, "visdrone": 247}
EXPECTED_VIS_GROUPS = 94

BOOTSTRAP_METRICS = ["AP50-95", "APs"]
BOOTSTRAP_SCOPES = ["full", "visdrone"]
METRIC_MAP = {
    "AP50-95": "AP_all",
    "APs": "AP_small",
}
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


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_features() -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with FEATURE_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"image_id", "source", "sequence_id", "file_name"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"corrected_scene_features.csv缺少字段：{sorted(missing)}")
        for raw in reader:
            image_id = int(raw["image_id"])
            if image_id in rows:
                raise RuntimeError(f"特征CSV重复image_id={image_id}")
            rows[image_id] = {
                "image_id": image_id,
                "source": raw["source"].strip().lower(),
                "sequence_id": raw["sequence_id"].strip(),
                "file_name": raw["file_name"],
            }
    if len(rows) != EXPECTED_TOTAL:
        raise RuntimeError(f"特征图片数={len(rows)}，预期={EXPECTED_TOTAL}")
    counts: dict[str, int] = defaultdict(int)
    for row in rows.values():
        counts[row["source"]] += 1
    if dict(counts) != EXPECTED_SOURCE_COUNTS:
        raise RuntimeError(f"来源数量不一致：{dict(counts)}")
    vis_groups = {r["sequence_id"] for r in rows.values() if r["source"] == "visdrone"}
    if len(vis_groups) != EXPECTED_VIS_GROUPS:
        raise RuntimeError(f"VisDrone sequence_id组数={len(vis_groups)}，预期={EXPECTED_VIS_GROUPS}")
    return rows


def load_prediction_index(path: Path, valid_ids: set[int]) -> dict[int, list[dict[str, Any]]]:
    raw = load_json(path)
    if not isinstance(raw, list):
        raise RuntimeError(f"预测文件不是COCO detection list：{path}")
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, item in enumerate(raw):
        image_id = int(item["image_id"])
        if image_id not in valid_ids:
            raise RuntimeError(f"{path.name}包含未知image_id={image_id}")
        bbox = [float(x) for x in item["bbox"]]
        score = float(item["score"])
        if len(bbox) != 4 or not all(math.isfinite(x) for x in bbox):
            raise RuntimeError(f"{path.name}第{index}条bbox非法")
        if not math.isfinite(score):
            raise RuntimeError(f"{path.name}第{index}条score非法")
        grouped[image_id].append(item)
    return dict(grouped)


def prediction_signature(records: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    return sorted(
        (
            int(p["image_id"]),
            int(p["category_id"]),
            round(float(p["score"]), 8),
            tuple(round(float(v), 6) for v in p["bbox"]),
        )
        for p in records
    )


def make_bootstrap_coco_dataset(
    base_gt: dict[str, Any],
    features: dict[int, dict[str, Any]],
    sampled_groups: list[str],
) -> tuple[Any, list[int], list[int], list[tuple[int, int]]]:
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
            copied = dict(item)
            copied["image_id"] = new_id
            output.append(copied)
    return output


def evaluate_scope(
    coco_gt: Any,
    predictions: list[dict[str, Any]],
    image_ids: list[int],
    category_ids: list[int],
) -> dict[str, float]:
    if not image_ids:
        raise RuntimeError("Bootstrap评价图片列表为空")
    coco_dt = coco_gt.loadRes(predictions)
    evaluator = COCOeval_faster(
        coco_gt, coco_dt, iouType="bbox", lvis_style=False,
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


def distribution(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) < 2 or not np.all(np.isfinite(arr)):
        raise RuntimeError("Bootstrap分布样本不足或包含NaN/Inf")
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=1)),
        "ci95_percentile_low": float(np.percentile(arr, 2.5)),
        "ci95_percentile_high": float(np.percentile(arr, 97.5)),
    }


def check_point_metrics(
    actual: dict[str, float],
    expected: dict[str, Any],
    label: str,
    tolerance: float = 1e-6,
) -> None:
    for metric in BOOTSTRAP_METRICS:
        if metric not in expected:
            raise RuntimeError(f"{label}参考指标缺少{metric}")
        diff = abs(float(actual[metric]) - float(expected[metric]))
        if diff > tolerance:
            raise RuntimeError(
                f"{label}点估计复现失败 {metric}: actual={actual[metric]:.9f}, "
                f"expected={float(expected[metric]):.9f}, diff={diff:.3g}"
            )


def load_existing_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def summarize_samples(rows: list[dict[str, Any]], args: argparse.Namespace,
                      input_hashes: dict[str, str], point_metrics: dict[str, Any],
                      reproduction: dict[str, Any]) -> dict[str, Any]:
    policies = list(PREDICTION_FILES)
    summary: dict[str, Any] = {
        "experiment": "Stage6.3F_A_Paired_VisDrone_Sequence_Cluster_Bootstrap",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "completed",
        "bootstrap": {
            "replicates": args.n_bootstrap,
            "seed": args.seed,
            "resampling_unit": "VisDrone sequence_id cluster",
            "number_of_sequence_groups": EXPECTED_VIS_GROUPS,
            "sampling": "sample 94 sequence groups with replacement; include every image in each sampled group",
            "uavdt_handling": "all 1320 UAVDT images held fixed in every replicate",
            "confidence_interval": "paired percentile 95% CI (2.5th, 97.5th percentile)",
            "paired_policy_comparisons": "all policies use the same sampled sequence groups in each replicate",
        },
        "interpretation_limit": (
            "Conditional cluster bootstrap of fixed OOF prediction files. It quantifies "
            "sequence-sampling uncertainty conditional on the already-fitted OOF selectors; "
            "it does not include selector retraining/split uncertainty and is not an independent test."
        ),
        "point_metrics_reproduction": point_metrics,
        "m1_b80_prediction_identity": reproduction,
        "input_sha256": input_hashes,
        "metric_distributions": {},
        "paired_delta_distributions": {},
        "outputs": {
            "samples_csv": str(OUT_SAMPLES.relative_to(ROOT)),
            "summary_json": str(OUT_SUMMARY.relative_to(ROOT)),
        },
    }

    for policy in policies:
        for scope in BOOTSTRAP_SCOPES:
            for metric in BOOTSTRAP_METRICS:
                col = f"{policy}_{scope}_{metric}"
                summary["metric_distributions"][col] = distribution(
                    [float(row[col]) for row in rows]
                )

    comparisons = [
        ("M3_Logistic_F0", "B80"),
        ("M4_Logistic_F1", "B80"),
        ("M3_Logistic_F0", "C2_area2048"),
        ("M4_Logistic_F1", "C2_area2048"),
        ("M4_Logistic_F1", "M3_Logistic_F0"),
        ("B80", "C2_area2048"),
    ]
    for policy, reference in comparisons:
        for scope in BOOTSTRAP_SCOPES:
            for metric in BOOTSTRAP_METRICS:
                col = f"delta_{policy}_vs_{reference}_{scope}_{metric}"
                values = np.asarray([float(row[col]) for row in rows], dtype=np.float64)
                summary["paired_delta_distributions"][col] = {
                    **distribution(values.tolist()),
                    "probability_delta_gt_zero": float(np.mean(values > 0.0)),
                    "probability_delta_ge_minus_0_002": float(np.mean(values >= -0.002)),
                }

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 6.3F-A paired sequence-cluster bootstrap for M3/M4"
    )
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.n_bootstrap < 100:
        parser.error("--n-bootstrap至少100；正式结果建议1000")
    if args.checkpoint_every < 1:
        parser.error("--checkpoint-every必须>=1")
    if version("faster-coco-eval") != EXPECTED_EVALUATOR_VERSION:
        raise RuntimeError(
            f"faster-coco-eval版本必须为{EXPECTED_EVALUATOR_VERSION}，"
            f"当前为{version('faster-coco-eval')}"
        )
    if np.__version__ != EXPECTED_NUMPY_VERSION:
        raise RuntimeError(
            f"NumPy版本必须为{EXPECTED_NUMPY_VERSION}，当前为{np.__version__}"
        )

    required = [
        FEATURE_CSV, GT_JSON, E3A_SUMMARY, E3B_SUMMARY, M1_REPRODUCTION_FILE,
        *PREDICTION_FILES.values(),
    ]
    missing = [p for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("缺少输入文件：\n" + "\n".join(map(str, missing)))

    if OUT_DIR.exists() and any(OUT_DIR.iterdir()):
        if args.overwrite and args.resume:
            raise RuntimeError("--overwrite与--resume不能同时使用")
        if not args.overwrite and not args.resume:
            raise FileExistsError(
                f"输出目录已有内容：{OUT_DIR}。继续中断任务使用--resume；"
                "确认重做才使用--overwrite。"
            )
        if args.overwrite:
            for p in OUT_DIR.iterdir():
                if p.is_file() and p.name in {
                    OUT_SAMPLES.name, OUT_SUMMARY.name,
                    OUT_SAMPLES.name + ".tmp", OUT_SUMMARY.name + ".tmp",
                }:
                    p.unlink()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("Stage 6.3F-A | Paired sequence-cluster bootstrap")
    print("Policies: A_Original, C2_area2048, B80, M3_Logistic_F0, M4_Logistic_F1")
    print("No YOLO training. No YOLO inference. No independent test set.")
    print("=" * 88)

    features = read_features()
    valid_ids = set(features)
    vis_groups = sorted({
        row["sequence_id"] for row in features.values()
        if row["source"] == "visdrone"
    })
    uavdt_ids = sorted(
        image_id for image_id, row in features.items() if row["source"] == "uavdt"
    )

    e3a = load_json(E3A_SUMMARY)
    e3b = load_json(E3B_SUMMARY)
    if e3a.get("status") != "completed" or e3b.get("status") != "completed":
        raise RuntimeError("Stage 6.3E-A或6.3E-B summary不是completed")

    gt = load_json(GT_JSON)
    gt_ids = {int(x["id"]) for x in gt["images"]}
    if gt_ids != valid_ids:
        raise RuntimeError("GT image_id与corrected_scene_features.csv不一致")
    category_ids = sorted(int(x["id"]) for x in gt["categories"])
    if not category_ids:
        raise RuntimeError("GT没有类别")

    pred_indexes = {
        name: load_prediction_index(path, valid_ids)
        for name, path in PREDICTION_FILES.items()
    }

    # The M1 rule policy must be exactly the B80 control, as already verified
    # in Stage 6.3E-B. Recheck it here before bootstrap.
    m1_raw = load_json(M1_REPRODUCTION_FILE)
    b80_raw = load_json(PREDICTION_FILES["B80"])
    if prediction_signature(m1_raw) != prediction_signature(b80_raw):
        raise RuntimeError("M1_Rule_F0预测与B80不一致，停止Bootstrap")

    input_hashes = {
        "corrected_scene_features.csv": sha256_file(FEATURE_CSV),
        "ground_truth": sha256_file(GT_JSON),
        "stage6_3e_a_summary": sha256_file(E3A_SUMMARY),
        "stage6_3e_b_summary": sha256_file(E3B_SUMMARY),
        "m1_rule_f0_prediction": sha256_file(M1_REPRODUCTION_FILE),
        **{f"{name}_prediction": sha256_file(path)
           for name, path in PREDICTION_FILES.items()},
    }

    # Verify M3/M4 prediction hashes against Stage 6.3E-B summary.
    for name in ("M3_Logistic_F0", "M4_Logistic_F1"):
        expected_hash = e3b["models"][name]["prediction_sha256"]
        actual_hash = input_hashes[f"{name}_prediction"]
        if expected_hash != actual_hash:
            raise RuntimeError(
                f"{name}预测SHA256与Stage 6.3E-B summary不一致："
                f"expected={expected_hash}, actual={actual_hash}"
            )

    # Reproduce the full-data point estimates before any bootstrap sampling.
    base_coco = COCO(str(GT_JSON))
    all_ids = sorted(valid_ids)
    vis_ids = sorted(
        image_id for image_id, row in features.items()
        if row["source"] == "visdrone"
    )
    point_metrics: dict[str, Any] = {}
    expected_metrics = {
        "A_Original": e3b["baseline_metrics"]["A_Original"],
        "C2_area2048": e3b["baseline_metrics"]["C2_area2048"],
        "B80": e3b["baseline_metrics"]["B80"],
        "M3_Logistic_F0": {
            "full": e3b["models"]["M3_Logistic_F0"]["full"],
            "visdrone": e3b["models"]["M3_Logistic_F0"]["visdrone"],
        },
        "M4_Logistic_F1": {
            "full": e3b["models"]["M4_Logistic_F1"]["full"],
            "visdrone": e3b["models"]["M4_Logistic_F1"]["visdrone"],
        },
    }
    for policy, index in pred_indexes.items():
        raw_predictions = [
            item for image_id in sorted(index) for item in index[image_id]
        ]
        full = evaluate_scope(base_coco, raw_predictions, all_ids, category_ids)
        vis = evaluate_scope(base_coco, raw_predictions, vis_ids, category_ids)
        check_point_metrics(full, expected_metrics[policy]["full"], f"{policy}/full")
        check_point_metrics(vis, expected_metrics[policy]["visdrone"], f"{policy}/visdrone")
        point_metrics[policy] = {"full": full, "visdrone": vis}
        print(
            f"[POINT PASS] {policy}: full APs={full['APs']:.6f}, "
            f"full AP50-95={full['AP50-95']:.6f}"
        )

    # Resume support: each replicate is deterministic from seed + replicate id.
    rows: list[dict[str, Any]] = []
    completed: set[int] = set()
    if args.resume and OUT_SAMPLES.is_file():
        rows = load_existing_rows(OUT_SAMPLES)
        completed = {int(row["replicate"]) for row in rows}
        if len(completed) != len(rows):
            raise RuntimeError("已有Bootstrap CSV包含重复replicate")
        if any(x < 0 or x >= args.n_bootstrap for x in completed):
            raise RuntimeError("已有Bootstrap replicate超出当前n-bootstrap范围")
        print(f"[RESUME] 已完成{len(completed)}/{args.n_bootstrap}次")

    start = time.perf_counter()
    for replicate in range(args.n_bootstrap):
        if replicate in completed:
            continue
        rng = np.random.default_rng(args.seed + replicate)
        sampled_groups = rng.choice(
            vis_groups, size=len(vis_groups), replace=True
        ).tolist()

        coco_boot, full_boot_ids, vis_boot_ids, vis_pairs = make_bootstrap_coco_dataset(
            gt, features, sampled_groups
        )
        row: dict[str, Any] = {
            "replicate": replicate,
            "sampled_sequence_groups": len(sampled_groups),
            "sampled_visdrone_images": len(vis_boot_ids),
        }
        for policy, index in pred_indexes.items():
            pred = remap_bootstrap_predictions(index, uavdt_ids, vis_pairs)
            full = evaluate_scope(coco_boot, pred, full_boot_ids, category_ids)
            vis = evaluate_scope(coco_boot, pred, vis_boot_ids, category_ids)
            for metric in BOOTSTRAP_METRICS:
                row[f"{policy}_full_{metric}"] = full[metric]
                row[f"{policy}_visdrone_{metric}"] = vis[metric]
            del pred, full, vis

        del coco_boot
        if (replicate + 1) % 25 == 0:
            gc.collect()

        # Paired deltas on the exact same sampled sequence groups.
        comparisons = [
            ("M3_Logistic_F0", "B80"),
            ("M4_Logistic_F1", "B80"),
            ("M3_Logistic_F0", "C2_area2048"),
            ("M4_Logistic_F1", "C2_area2048"),
            ("M4_Logistic_F1", "M3_Logistic_F0"),
            ("B80", "C2_area2048"),
        ]
        for policy, reference in comparisons:
            for scope in BOOTSTRAP_SCOPES:
                for metric in BOOTSTRAP_METRICS:
                    row[f"delta_{policy}_vs_{reference}_{scope}_{metric}"] = (
                        row[f"{policy}_{scope}_{metric}"]
                        - row[f"{reference}_{scope}_{metric}"]
                    )

        rows.append(row)
        completed.add(replicate)
        if len(completed) % args.checkpoint_every == 0 or replicate == args.n_bootstrap - 1:
            rows.sort(key=lambda x: int(x["replicate"]))
            write_csv_atomic(OUT_SAMPLES, rows)
            elapsed = (time.perf_counter() - start) / 60.0
            print(
                f"[BOOTSTRAP] {len(completed)}/{args.n_bootstrap} | "
                f"last sampled VisDrone images={len(vis_boot_ids)} | "
                f"elapsed={elapsed:.1f} min",
                flush=True,
            )

    rows.sort(key=lambda x: int(x["replicate"]))
    write_csv_atomic(OUT_SAMPLES, rows)
    if len(rows) != args.n_bootstrap:
        raise RuntimeError(f"Bootstrap仅完成{len(rows)}/{args.n_bootstrap}次")

    # A stable report requires one row for every replicate, no duplicates.
    if {int(x["replicate"]) for x in rows} != set(range(args.n_bootstrap)):
        raise RuntimeError("Bootstrap replicate ID不连续或缺失")

    summary = summarize_samples(
        rows, args, input_hashes, point_metrics,
        {
            "status": "passed",
            "checked": "M1_Rule_F0 == Stage 6.3E-A B80 prediction records",
            "prediction_records_equal": True,
        },
    )
    write_json_atomic(OUT_SUMMARY, summary)

    print("\n" + "=" * 88)
    print("Stage 6.3F-A completed.")
    print("Samples:", OUT_SAMPLES)
    print("Summary:", OUT_SUMMARY)
    for key in (
        "delta_M3_Logistic_F0_vs_B80_full_APs",
        "delta_M4_Logistic_F1_vs_B80_full_APs",
        "delta_M4_Logistic_F1_vs_M3_Logistic_F0_full_APs",
        "delta_M4_Logistic_F1_vs_C2_area2048_full_APs",
        "delta_M4_Logistic_F1_vs_B80_full_AP50-95",
        "delta_M4_Logistic_F1_vs_C2_area2048_full_AP50-95",
    ):
        print(key, summary["paired_delta_distributions"][key])
    print("=" * 88)


if __name__ == "__main__":
    main()
