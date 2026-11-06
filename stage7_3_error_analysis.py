#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 7.3 — Error diagnosis and runtime distribution analysis.

Uses the frozen Stage 7.2 predictions. It does NOT train or run YOLO inference.
Diagnostic matching:
  confidence threshold = 0.25
  IoU thresholds = 0.50 and 0.75
  greedy one-to-one matching, descending prediction confidence

This is a diagnostic analysis, not a replacement for COCO AP evaluation.
Run from the YOLO_Thesis repository root.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
STAGE7_DIR = ROOT / "runs/detect/stage7_p2bridge_slice/stage7_2_independent_test"
OUT_DEFAULT = ROOT / "runs/detect/stage7_p2bridge_slice/stage7_3_error_analysis"

MANIFEST_DEFAULT = ROOT / "runs/detect/stage6_scene_selector/stage6_4_preflight/independent_test_manifest.csv"
GT_DEFAULT = ROOT / "runs/detect/stage6_scene_selector/stage6_4b_independent_test/test_gt_coco.json"
DATASET_DEFAULT = ROOT / "datasets/HighwayVehicle-v1.0"

CONF_THRESHOLD = 0.25
IOU_THRESHOLDS = (0.50, 0.75)
AREA_BOUNDS = (1024.0, 9216.0)
POLICIES = ("A_original", "B_C2_area2048")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]):
    if not rows:
        raise RuntimeError(f"拒绝写入空CSV：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def xywh_to_xyxy(box):
    x, y, w, h = map(float, box)
    return np.asarray([x, y, x + w, y + h], dtype=np.float32)


def box_area_xyxy(box):
    return max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))


def size_group(area: float) -> str:
    if area < AREA_BOUNDS[0]:
        return "small"
    if area < AREA_BOUNDS[1]:
        return "medium"
    return "large"


def iou_matrix(pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return np.zeros((len(pred_boxes), len(gt_boxes)), dtype=np.float32)

    lt = np.maximum(pred_boxes[:, None, :2], gt_boxes[None, :, :2])
    rb = np.minimum(pred_boxes[:, None, 2:], gt_boxes[None, :, 2:])
    wh = np.maximum(rb - lt, 0.0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    p_area = np.maximum(pred_boxes[:, 2] - pred_boxes[:, 0], 0) * np.maximum(
        pred_boxes[:, 3] - pred_boxes[:, 1], 0
    )
    g_area = np.maximum(gt_boxes[:, 2] - gt_boxes[:, 0], 0) * np.maximum(
        gt_boxes[:, 3] - gt_boxes[:, 1], 0
    )
    union = p_area[:, None] + g_area[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def load_predictions(path: Path, valid_ids: set[int], category_id: int):
    raw = load_json(path)
    if not isinstance(raw, list):
        raise RuntimeError(f"预测文件不是COCO detection list：{path}")
    by_image = defaultdict(list)
    for idx, item in enumerate(raw):
        iid = int(item["image_id"])
        if iid not in valid_ids:
            raise RuntimeError(f"{path.name}第{idx}条image_id不在GT中：{iid}")
        if int(item["category_id"]) != category_id:
            raise RuntimeError(f"{path.name}第{idx}条category_id异常")
        bbox = [float(v) for v in item["bbox"]]
        score = float(item["score"])
        if len(bbox) != 4 or not all(math.isfinite(v) for v in bbox):
            raise RuntimeError(f"{path.name}第{idx}条bbox非法")
        if bbox[2] <= 0 or bbox[3] <= 0 or not math.isfinite(score) or not 0 <= score <= 1:
            raise RuntimeError(f"{path.name}第{idx}条bbox/score范围异常")
        by_image[iid].append({
            "bbox_xyxy": xywh_to_xyxy(bbox),
            "bbox_xywh": bbox,
            "score": score,
            "category_id": int(item["category_id"]),
        })
    return dict(by_image)


def match_one_image(gt_items, pred_items, iou_threshold: float, conf_threshold: float):
    """Greedy score-ordered one-to-one matching and diagnostic FP categorization."""
    gt_boxes = np.asarray([xywh_to_xyxy(x["bbox"]) for x in gt_items], dtype=np.float32)
    gt_areas = [
        float(x.get("area", float(x["bbox"][2]) * float(x["bbox"][3])))
        for x in gt_items
    ]
    gt_sizes = [size_group(a) for a in gt_areas]

    preds = [p for p in pred_items if p["score"] >= conf_threshold]
    preds.sort(key=lambda x: x["score"], reverse=True)
    pred_boxes = np.asarray([p["bbox_xyxy"] for p in preds], dtype=np.float32).reshape(-1, 4)

    overlaps = iou_matrix(pred_boxes, gt_boxes)
    matched_gt: set[int] = set()
    result = {
        "tp": 0, "fp": 0, "fn": 0,
        "fp_duplicate": 0, "fp_localization": 0, "fp_background": 0,
        "gt_count": len(gt_items), "pred_count": len(preds),
        "tp_small": 0, "tp_medium": 0, "tp_large": 0,
        "fn_small": 0, "fn_medium": 0, "fn_large": 0,
        "fp_small": 0, "fp_medium": 0, "fp_large": 0,
    }

    for pi, pred in enumerate(preds):
        row = overlaps[pi] if len(gt_items) else np.zeros((0,), dtype=np.float32)
        unmatched = [gi for gi in range(len(gt_items)) if gi not in matched_gt]

        best_unmatched_iou = -1.0
        best_unmatched_gt = None
        if unmatched:
            local = row[unmatched]
            pos = int(np.argmax(local))
            best_unmatched_iou = float(local[pos])
            best_unmatched_gt = unmatched[pos]

        max_iou = float(row.max()) if len(row) else 0.0

        if best_unmatched_gt is not None and best_unmatched_iou >= iou_threshold:
            matched_gt.add(best_unmatched_gt)
            result["tp"] += 1
            result[f"tp_{gt_sizes[best_unmatched_gt]}"] += 1
        else:
            result["fp"] += 1
            pred_area = box_area_xyxy(pred["bbox_xyxy"])
            result[f"fp_{size_group(pred_area)}"] += 1
            if max_iou >= iou_threshold:
                result["fp_duplicate"] += 1
            elif max_iou >= 0.10:
                result["fp_localization"] += 1
            else:
                result["fp_background"] += 1

    result["fn"] = len(gt_items) - len(matched_gt)
    for gi, size in enumerate(gt_sizes):
        if gi not in matched_gt:
            result[f"fn_{size}"] += 1
    return result


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count_keys = [
        "tp", "fp", "fn", "fp_duplicate", "fp_localization", "fp_background",
        "gt_count", "pred_count",
        "tp_small", "tp_medium", "tp_large",
        "fn_small", "fn_medium", "fn_large",
        "fp_small", "fp_medium", "fp_large",
    ]
    total = {k: int(sum(int(r[k]) for r in rows)) for k in count_keys}
    tp, fp, fn = total["tp"], total["fp"], total["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    total.update({
        "precision_at_conf": precision,
        "recall_at_conf": recall,
        "f1_at_conf": f1,
    })
    total["size_breakdown"] = {}
    for size in ("small", "medium", "large"):
        stp, sfp, sfn = total[f"tp_{size}"], total[f"fp_{size}"], total[f"fn_{size}"]
        sp = stp / (stp + sfp) if stp + sfp else 0.0
        sr = stp / (stp + sfn) if stp + sfn else 0.0
        total["size_breakdown"][size] = {
            "tp": stp, "fp_by_predicted_box_size": sfp, "fn": sfn,
            "precision_diagnostic": sp, "recall_diagnostic": sr,
        }
    return total


def percentile_stats(values):
    arr = np.asarray(values, dtype=np.float64)
    if not len(arr):
        return {"count": 0}
    return {
        "count": int(len(arr)),
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "max_ms": float(arr.max()),
    }


def runtime_analysis(runtime_rows):
    output = {}
    for source in ("full", "uavdt", "visdrone"):
        subset = runtime_rows if source == "full" else [
            r for r in runtime_rows if r["source"].lower() == source
        ]
        if not subset:
            continue
        a_model = [float(r["original_ms"]) for r in subset]
        b_model = [
            float(r["original_ms"]) + float(r["tile_ms"]) + float(r["fusion_ms"])
            for r in subset
        ]
        read = [float(r["read_ms"]) for r in subset]
        output[source] = {
            "A_original_model_components": percentile_stats(a_model),
            "B_C2_area2048_model_components": percentile_stats(b_model),
            "B_minus_A_mean_ms": float(np.mean(b_model) - np.mean(a_model)),
            "B_vs_A_mean_percent": float(
                (np.mean(b_model) / np.mean(a_model) - 1.0) * 100.0
            ) if np.mean(a_model) else None,
            "image_read_component": percentile_stats(read),
            "note": "组件统计，不含模型加载、warm-up、checkpoint及完整部署调度开销。",
        }
    return output


def draw_case(cv2, image_path: Path, gt_items, pred_a, pred_b, out_path: Path,
              image_id: int, source: str, conf: float, iou_thr: float,
              stats_a: dict, stats_b: dict):
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"无法读取可视化图片：{image_path}")

    h, w = image.shape[:2]
    max_w, max_h = 960, 640
    scale = min(max_w / w, max_h / h, 1.0)
    target_w, target_h = max(1, int(w * scale)), max(1, int(h * scale))

    def make_panel(title, pred_items=None, pred_color=(255, 128, 0)):
        panel = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
        # Ground truth: green.
        for gt in gt_items:
            x, y, bw, bh = map(float, gt["bbox"])
            p1 = (int(x * scale), int(y * scale))
            p2 = (int((x + bw) * scale), int((y + bh) * scale))
            cv2.rectangle(panel, p1, p2, (0, 220, 0), 2)
        if pred_items is not None:
            chosen = sorted(
                [p for p in pred_items if p["score"] >= conf],
                key=lambda x: x["score"], reverse=True
            )[:80]
            for pred in chosen:
                x1, y1, x2, y2 = map(float, pred["bbox_xyxy"])
                p1 = (int(x1 * scale), int(y1 * scale))
                p2 = (int(x2 * scale), int(y2 * scale))
                cv2.rectangle(panel, p1, p2, pred_color, 2)
                cv2.putText(
                    panel, f"{pred['score']:.2f}", (p1[0], max(12, p1[1] - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, pred_color, 1, cv2.LINE_AA
                )
        header = np.full((42, target_w, 3), 245, dtype=np.uint8)
        cv2.putText(header, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (20, 20, 20), 2, cv2.LINE_AA)
        return np.vstack([header, panel])

    panel_gt = make_panel(f"GT | {source} | image_id={image_id}", None)
    panel_a = make_panel(
        f"A Original | TP {stats_a['tp']} FP {stats_a['fp']} FN {stats_a['fn']}",
        pred_a, (255, 128, 0)
    )
    panel_b = make_panel(
        f"B C2-2048 | TP {stats_b['tp']} FP {stats_b['fp']} FN {stats_b['fn']}",
        pred_b, (0, 0, 255)
    )
    canvas = np.hstack([panel_gt, panel_a, panel_b])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), canvas):
        raise RuntimeError(f"图片写入失败：{out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DATASET_DEFAULT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_DEFAULT)
    parser.add_argument("--gt", type=Path, default=GT_DEFAULT)
    parser.add_argument("--stage7-dir", type=Path, default=STAGE7_DIR)
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--vis-limit", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    metrics_path = args.stage7_dir / "metrics_summary.json"
    pred_paths = {
        "A_original": args.stage7_dir / "test_predictions_A_original.json",
        "B_C2_area2048": args.stage7_dir / "test_predictions_B_C2_area2048.json",
    }
    runtime_path = args.stage7_dir / "runtime_per_image.csv"
    required = [metrics_path, *pred_paths.values(), runtime_path, args.manifest, args.gt]
    missing = [p for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Stage 7.2必要输入缺失。不要重新推理，先恢复本地文件：\n"
            + "\n".join(map(str, missing))
        )
    if not args.dataset_root.is_dir():
        raise FileNotFoundError(f"数据集目录不存在：{args.dataset_root}")

    if args.out.exists() and any(args.out.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"输出目录已有内容：{args.out}；确认重做请加--overwrite")
        import shutil
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    summary_72 = load_json(metrics_path)
    if summary_72.get("status") != "completed":
        raise RuntimeError("Stage 7.2状态不是completed")
    if summary_72.get("experiment") != "Stage7.2_P2Bridge_MLCA_TD_Independent_Test":
        raise RuntimeError("Stage 7.2实验标识不匹配")
    if summary_72["protocol"].get("evaluator_version") != "1.8.0":
        raise RuntimeError("Stage 7.2 evaluator版本不是1.8.0")

    manifest_sha = sha256_file(args.manifest)
    gt_sha = sha256_file(args.gt)
    if summary_72.get("inputs_sha256", {}).get("manifest") != manifest_sha:
        raise RuntimeError("manifest SHA256与Stage 7.2记录不一致")
    if summary_72.get("inputs_sha256", {}).get("test_gt") != gt_sha:
        raise RuntimeError("GT SHA256与Stage 7.2记录不一致")

    manifest = read_csv(args.manifest)
    if len(manifest) != 1372:
        raise RuntimeError(f"manifest应为1372行，实际{len(manifest)}")
    gt = load_json(args.gt)
    gt_images = {int(x["id"]): x for x in gt["images"]}
    valid_ids = set(gt_images)
    if len(valid_ids) != 1372:
        raise RuntimeError("COCO GT图片数不是1372")
    if len(gt.get("categories", [])) != 1:
        raise RuntimeError("COCO GT不是单类别")
    category_id = int(gt["categories"][0]["id"])

    ann_by_image = defaultdict(list)
    for ann in gt["annotations"]:
        ann_by_image[int(ann["image_id"])].append(ann)

    pred_indexes = {}
    for policy, path in pred_paths.items():
        actual = sha256_file(path)
        expected = summary_72.get("predictions_sha256", {}).get(policy)
        if actual != expected:
            raise RuntimeError(f"{policy}预测文件SHA256不匹配")
        pred_indexes[policy] = load_predictions(path, valid_ids, category_id)

    manifest_by_id = {}
    for row in manifest:
        iid = int(row["image_id"])
        if iid in manifest_by_id or iid not in valid_ids:
            raise RuntimeError(f"manifest image_id重复或不在GT：{iid}")
        manifest_by_id[iid] = row
    if set(manifest_by_id) != valid_ids:
        raise RuntimeError("manifest与GT的image_id集合不一致")

    runtime_rows = read_csv(runtime_path)
    if len(runtime_rows) != 1372:
        raise RuntimeError("runtime_per_image.csv不是1372行")
    runtime_by_id = {int(x["image_id"]): x for x in runtime_rows}
    if len(runtime_by_id) != 1372 or set(runtime_by_id) != valid_ids:
        raise RuntimeError("runtime CSV的image_id集合不一致")

    print("=" * 78)
    print("Stage 7.3 — Error diagnosis + runtime distribution")
    print("Frozen Stage 7.2 predictions only; no YOLO inference or training.")
    print("Images:", len(valid_ids), "| confidence:", CONF_THRESHOLD,
          "| IoU thresholds:", IOU_THRESHOLDS)
    print("=" * 78, flush=True)

    all_diagnostic_rows = []
    primary_case_stats = {}
    primary_image_rows = []
    aggregate_summary = {}

    for iou_thr in IOU_THRESHOLDS:
        key = f"conf{CONF_THRESHOLD:.2f}_iou{iou_thr:.2f}"
        aggregate_summary[key] = {}
        for scope in ("full", "uavdt", "visdrone"):
            ids = sorted(valid_ids) if scope == "full" else sorted(
                iid for iid, row in manifest_by_id.items()
                if row["source"].lower() == scope
            )
            aggregate_summary[key][scope] = {}
            for policy in POLICIES:
                per_image = []
                for iid in ids:
                    gt_items = ann_by_image.get(iid, [])
                    pred_items = pred_indexes[policy].get(iid, [])
                    one = match_one_image(gt_items, pred_items, iou_thr, CONF_THRESHOLD)
                    row = manifest_by_id[iid]
                    record = {
                        "image_id": iid,
                        "file_name": row["file_name"],
                        "source": row["source"].lower(),
                        "group_id": row["group_id"],
                        "policy": policy,
                        "conf_threshold": CONF_THRESHOLD,
                        "iou_threshold": iou_thr,
                        **one,
                    }
                    per_image.append(record)
                    all_diagnostic_rows.append(record)
                    if iou_thr == 0.50:
                        primary_case_stats[(policy, iid)] = one
                aggregate_summary[key][scope][policy] = aggregate(per_image)

    # Paired A/B image-level differences at the primary diagnostic point.
    for iid in sorted(valid_ids):
        row = manifest_by_id[iid]
        a = primary_case_stats[("A_original", iid)]
        b = primary_case_stats[("B_C2_area2048", iid)]
        primary_image_rows.append({
            "image_id": iid,
            "file_name": row["file_name"],
            "source": row["source"].lower(),
            "group_id": row["group_id"],
            "A_tp": a["tp"], "B_tp": b["tp"], "delta_tp_B_minus_A": b["tp"] - a["tp"],
            "A_fp": a["fp"], "B_fp": b["fp"], "delta_fp_B_minus_A": b["fp"] - a["fp"],
            "A_fn": a["fn"], "B_fn": b["fn"], "delta_fn_B_minus_A": b["fn"] - a["fn"],
            "A_fp_duplicate": a["fp_duplicate"], "B_fp_duplicate": b["fp_duplicate"],
            "A_fp_localization": a["fp_localization"], "B_fp_localization": b["fp_localization"],
            "A_fp_background": a["fp_background"], "B_fp_background": b["fp_background"],
        })

    diagnostic_csv = args.out / "per_image_error_diagnostics.csv"
    paired_csv = args.out / "paired_image_differences.csv"
    write_csv(diagnostic_csv, all_diagnostic_rows)
    write_csv(paired_csv, primary_image_rows)

    # Runtime distribution: component sums only, not end-to-end deployment latency.
    runtime_summary = runtime_analysis(runtime_rows)

    # Select VisDrone examples where B recovers GT, and examples where B adds errors.
    vis_rows = [r for r in primary_image_rows if r["source"] == "visdrone"]
    improved = sorted(
        [r for r in vis_rows if r["delta_fn_B_minus_A"] < 0 or r["delta_tp_B_minus_A"] > 0],
        key=lambda r: (
            -max(0, -r["delta_fn_B_minus_A"]),
            -max(0, r["delta_tp_B_minus_A"]),
            r["delta_fp_B_minus_A"],
        )
    )[:args.vis_limit]
    regressed = sorted(
        [r for r in vis_rows if r["delta_fn_B_minus_A"] > 0 or r["delta_fp_B_minus_A"] > 0],
        key=lambda r: (
            -max(0, r["delta_fn_B_minus_A"]),
            -max(0, r["delta_fp_B_minus_A"]),
        )
    )[:args.vis_limit]

    import cv2
    vis_dir = args.out / "visdrone_cases"
    for label, cases in (("improved", improved), ("regressed", regressed)):
        for rank, case in enumerate(cases, 1):
            iid = int(case["image_id"])
            row = manifest_by_id[iid]
            rel = Path(row["relative_path"])
            if rel.is_absolute() or ".." in rel.parts:
                raise RuntimeError(f"manifest relative_path非法：{rel}")
            image_path = args.dataset_root / "images" / "test" / rel
            a_stat = primary_case_stats[("A_original", iid)]
            b_stat = primary_case_stats[("B_C2_area2048", iid)]
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(row["file_name"]).stem)
            out_path = vis_dir / label / f"{rank:02d}_{iid}_{safe_name}.jpg"
            draw_case(
                cv2, image_path, ann_by_image.get(iid, []),
                pred_indexes["A_original"].get(iid, []),
                pred_indexes["B_C2_area2048"].get(iid, []),
                out_path, iid, row["source"].lower(),
                CONF_THRESHOLD, 0.50, a_stat, b_stat
            )

    final = {
        "experiment": "Stage7.3_P2Bridge_C2_2048_Error_And_Runtime_Analysis",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "completed",
        "source_stage": "Stage7.2 frozen predictions",
        "diagnostic_protocol": {
            "confidence_threshold": CONF_THRESHOLD,
            "iou_thresholds": list(IOU_THRESHOLDS),
            "matching": "greedy one-to-one matching, predictions sorted by descending confidence",
            "fp_categories": {
                "duplicate": "IoU >= matching threshold but all qualifying GT objects are already matched",
                "localization": "maximum IoU is >= 0.10 but below matching threshold",
                "background": "maximum IoU < 0.10 or no GT object",
            },
            "area_groups_px2": {
                "small": "< 1024",
                "medium": "1024 to < 9216",
                "large": ">= 9216",
            },
            "warning": "These fixed-threshold diagnostics are not COCO AP and must not replace Stage 7.2 official COCO metrics.",
        },
        "official_stage7_2_metrics_reference": summary_72["metrics"],
        "diagnostic_metrics": aggregate_summary,
        "runtime_component_distribution": runtime_summary,
        "max_det_300_saturation_reference": summary_72.get("max_det_300_saturation"),
        "visual_cases": {
            "improved_visdrone": len(improved),
            "regressed_visdrone": len(regressed),
            "directory": str(vis_dir.relative_to(ROOT)),
            "note": "Panels show GT, A predictions, and B predictions. Only top 80 predictions by confidence are drawn per panel; metric calculations use all predictions above confidence threshold.",
        },
        "input_sha256": {
            "stage7_2_metrics": sha256_file(metrics_path),
            "manifest": manifest_sha,
            "test_gt": gt_sha,
            "A_predictions": sha256_file(pred_paths["A_original"]),
            "B_predictions": sha256_file(pred_paths["B_C2_area2048"]),
            "runtime_per_image": sha256_file(runtime_path),
        },
        "interpretation_limit": summary_72.get("interpretation_limit", ""),
        "outputs": {
            "summary_json": str((args.out / "stage7_3_error_analysis_summary.json").relative_to(ROOT)),
            "per_image_error_diagnostics": str(diagnostic_csv.relative_to(ROOT)),
            "paired_image_differences": str(paired_csv.relative_to(ROOT)),
            "visual_cases": str(vis_dir.relative_to(ROOT)),
        },
    }
    write_json(args.out / "stage7_3_error_analysis_summary.json", final)

    print("\nStage 7.3 completed.")
    print("Summary:", args.out / "stage7_3_error_analysis_summary.json")
    print("Per-image diagnostics:", diagnostic_csv)
    print("Paired A/B differences:", paired_csv)
    print("VisDrone visual cases:", vis_dir)
    for key in ("conf0.25_iou0.50", "conf0.25_iou0.75"):
        print("\n", key)
        for scope in ("full", "visdrone"):
            print(scope)
            for policy in POLICIES:
                m = aggregate_summary[key][scope][policy]
                print(policy, "TP/FP/FN=", m["tp"], m["fp"], m["fn"],
                      "P/R/F1=", round(m["precision_at_conf"], 4),
                      round(m["recall_at_conf"], 4), round(m["f1_at_conf"], 4))
    print("\nRuntime component distributions:")
    for scope, data in runtime_summary.items():
        print(scope,
              "A mean/p95=", round(data["A_original_model_components"]["mean_ms"], 3),
              round(data["A_original_model_components"]["p95_ms"], 3),
              "B mean/p95=", round(data["B_C2_area2048_model_components"]["mean_ms"], 3),
              round(data["B_C2_area2048_model_components"]["p95_ms"], 3))


if __name__ == "__main__":
    main()
