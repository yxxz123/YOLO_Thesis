# ============================================================
# Stage 6.3B2: COCO-aligned per-image small-object gain analysis
# Run from repository root: python stage6_analyze_selector_gain_coco.py
# Reuses existing predictions; does NOT train or run model inference.
# ============================================================

import csv
import gc
import hashlib
import json
import math
import platform
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
from faster_coco_eval import COCO, COCOeval_faster


# ------------------------------ Paths ------------------------------
ROOT = Path(__file__).resolve().parent
SCENE_DIR = ROOT / "runs/detect/stage6_scene_selector"
ORACLE_DIR = ROOT / "runs/detect/stage6_selective_oracle"
C2_DIR = ROOT / "runs/detect/stage6_size_aware_fusion_iou07"

FEATURE_CSV = SCENE_DIR / "corrected_scene_features.csv"
A_JSON = ORACLE_DIR / "all_original_predictions.json"
C2_JSON = C2_DIR / "C2_area2048_predictions.json"
GT_JSON = ROOT / "runs/detect/source_validation/Stage5-Baseline-Unified/annotations_auto_detect_val.json"
REFERENCE_JSON = C2_DIR / "faster_eval_metrics.json"
OUT_CSV = SCENE_DIR / "coco_utility_analysis.csv"
OUT_JSON = SCENE_DIR / "coco_utility_summary.json"


# ------------------------- Fixed protocol --------------------------
TOP_K = 100
IOU_THRS = np.linspace(0.50, 0.95, 10, dtype=np.float64)
REC_THRS = np.linspace(0.0, 1.0, 101, dtype=np.float64)
MAX_DETS = [1, 10, 100]
AREA_RANGES = [[0, 1e10], [0, 32**2], [32**2, 96**2], [96**2, 1e5**2]]
AREA_LABELS = ["all", "small", "medium", "large"]
SMALL_RANGE = AREA_RANGES[1]
SMALL_AREA_INDEX = AREA_LABELS.index("small")
MAX_DET_INDEX = MAX_DETS.index(TOP_K)
EXPECTED_IMAGES = 1567
EXPECTED_SOURCE_COUNTS = {"uavdt": 1320, "visdrone": 247}
LABEL_EPS = 1e-12

# Lock this script to the exact Stage 6.3B data already reviewed.
EXPECTED_SHA256 = {
    "features": "db33bd0a0cd47b32298d36fab9fd99e7f8835def5a03a28bed8c7cadc566106d",
    "original": "4979c79d6966539b51f67592f8bdf10efc21d71f126065d13c730dc20a79fc23",
    "c2": "7e018f981f2cbcf39b512226892b58a067655c5e49c4ace8a6c07282ca3ef5b6",
    "gt": "75098d09c3a02e8f8e32eab99df6c977474748dd177a97b1692f766bbdcf0de4",
}


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def finite_float(x, name="value"):
    try:
        x = float(x)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{name} is not numeric: {x!r}") from e
    if not math.isfinite(x):
        raise ValueError(f"{name} is not finite: {x!r}")
    return x


def bbox_area(box):
    _, _, w, h = map(float, box)
    return max(0.0, w) * max(0.0, h)


def bbox_iou_coco(dt, gt, iscrowd=False):
    """COCO bbox IoU; crowd GT uses intersection / detection area."""
    x, y, w, h = map(float, dt)
    gx, gy, gw, gh = map(float, gt)
    ix1, iy1 = max(x, gx), max(y, gy)
    ix2, iy2 = min(x + w, gx + gw), min(y + h, gy + gh)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    da, ga = max(0.0, w) * max(0.0, h), max(0.0, gw) * max(0.0, gh)
    denom = da if iscrowd else da + ga - inter
    return inter / denom if denom > 0 else 0.0


def ratio(a, b):
    return float(a / b) if b > 0 else 0.0


def prf(tp, fp, fn):
    p, r = ratio(tp, tp + fp), ratio(tp, tp + fn)
    return p, r, ratio(2 * p * r, p + r)


# ------------------------- Load input data -------------------------
def load_features():
    required = {
        "image_id", "file_name", "source", "sequence_id", "width", "height",
        "n_det_conf025", "median_box_area_640", "tiny_ratio", "tiny_count",
    }
    out = {}
    with FEATURE_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"Feature CSV missing columns: {sorted(missing)}")
        for row in reader:
            image_id = int(row["image_id"])
            if image_id in out:
                raise RuntimeError(f"Duplicate feature image_id={image_id}")
            source = row["source"].strip().lower()
            if source not in EXPECTED_SOURCE_COUNTS:
                raise RuntimeError(f"Unknown source {source!r} for image {image_id}")
            out[image_id] = {
                "image_id": image_id,
                "file_name": row["file_name"],
                "source": source,
                "sequence_id": row["sequence_id"],
                "width": int(row["width"]),
                "height": int(row["height"]),
                "n_det_conf025": int(row["n_det_conf025"]),
                "median_box_area_640": (
                    finite_float(row["median_box_area_640"], "median_box_area_640")
                    if row["median_box_area_640"] not in ("", None) else None
                ),
                "tiny_ratio": finite_float(row["tiny_ratio"], "tiny_ratio"),
                "tiny_count": int(row["tiny_count"]),
            }
    if len(out) != EXPECTED_IMAGES:
        raise RuntimeError(f"Expected {EXPECTED_IMAGES} feature rows, got {len(out)}")
    counts = Counter(x["source"] for x in out.values())
    if dict(counts) != EXPECTED_SOURCE_COUNTS:
        raise RuntimeError(f"Source counts mismatch: {dict(counts)}")
    return out


def load_gt(feature_by_id):
    data = load_json(GT_JSON)
    images = {int(x["id"]): x for x in data["images"]}
    if set(images) != set(feature_by_id):
        raise RuntimeError("GT image IDs do not exactly match corrected_scene_features.csv")
    categories = sorted(int(x["id"]) for x in data["categories"])
    if not categories:
        raise RuntimeError("No categories in GT JSON")
    grouped = defaultdict(lambda: defaultdict(list))
    for ann in data["annotations"]:
        image_id, category_id = int(ann["image_id"]), int(ann["category_id"])
        if "area" not in ann:
            raise RuntimeError("GT annotation lacks area; do not substitute bbox area")
        box = ann.get("bbox")
        if box is None or len(box) != 4:
            raise RuntimeError(f"Invalid GT bbox: {ann}")
        grouped[image_id][category_id].append({
            "id": int(ann["id"]), "bbox": [float(v) for v in box],
            "area": finite_float(ann["area"], "GT area"),
            "iscrowd": int(ann.get("iscrowd", 0)),
            "ignore": int(ann.get("iscrowd", 0)),
        })
    return categories, grouped


def load_predictions(path, feature_by_id):
    raw = load_json(path)
    if not isinstance(raw, list):
        raise RuntimeError(f"Prediction JSON must be a list: {path}")
    grouped = defaultdict(lambda: defaultdict(list))
    for pred in raw:
        image_id, category_id = int(pred["image_id"]), int(pred["category_id"])
        if image_id not in feature_by_id:
            raise RuntimeError(f"Prediction has unknown image_id={image_id}: {path}")
        box = pred.get("bbox")
        if box is None or len(box) != 4:
            raise RuntimeError(f"Invalid prediction bbox: {pred}")
        box = [finite_float(v, "prediction bbox") for v in box]
        if box[2] < 0 or box[3] < 0:
            raise RuntimeError(f"Negative prediction width/height: {pred}")
        # Preserve category_id in the normalized record. The grouped dictionary
        # is keyed by category_id, but check_uavdt_original_only flattens the
        # category groups before pred_signature() compares A and C2 predictions.
        grouped[image_id][category_id].append({
            "category_id": category_id,
            "bbox": box,
            "score": finite_float(pred["score"], "prediction score"),
        })
    for image_id in grouped:
        for category_id in grouped[image_id]:
            grouped[image_id][category_id].sort(key=lambda x: x["score"], reverse=True)
    return grouped, len(raw)


def pred_signature(records):
    return Counter((int(p["category_id"]), round(float(p["score"]), 8),
                    tuple(round(float(v), 6) for v in p["bbox"])) for p in records)


def check_uavdt_original_only(features, a_by_id, c2_by_id):
    mismatches = []
    for image_id, feat in features.items():
        if feat["source"] != "uavdt":
            continue
        a = [p for group in a_by_id.get(image_id, {}).values() for p in group]
        c = [p for group in c2_by_id.get(image_id, {}).values() for p in group]
        if pred_signature(a) != pred_signature(c):
            mismatches.append(image_id)
    return {"checked_images": 1320, "mismatch_images": len(mismatches),
            "mismatch_preview": mismatches[:20], "passed": not mismatches}


# ---------------- COCO-compatible per-image small matching ----------------
def eval_one_image_category(gt_records, dt_records):
    """COCO evaluateImg semantics for one image/category and area=small."""
    low, high = SMALL_RANGE
    detections = list(dt_records[:TOP_K])  # stable score-descending order

    gt = []
    for ann in gt_records:
        area = float(ann["area"])
        crowd = bool(ann.get("iscrowd", 0))
        ignored = bool(ann.get("ignore", 0)) or area < low or area > high
        gt.append({**ann, "crowd": crowd, "ignored": ignored})
    gt.sort(key=lambda x: int(x["ignored"]))  # valid GT first, stable

    valid_gt = sum(not x["ignored"] for x in gt)
    ious = np.zeros((len(detections), len(gt)), dtype=np.float64)
    for di, det in enumerate(detections):
        for gi, ann in enumerate(gt):
            ious[di, gi] = bbox_iou_coco(det["bbox"], ann["bbox"], ann["crowd"])

    result = {"gt": int(valid_gt), "ignored_gt": len(gt) - valid_gt,
              "tp": [], "fp": [], "fn": [], "ignored_dt": []}
    for threshold in IOU_THRS:
        gt_matched = np.zeros(len(gt), dtype=bool)
        tp = fp = ignored_dt = 0
        for di, det in enumerate(detections):
            best_iou, match = min(float(threshold), 1.0 - 1e-10), -1
            for gi, ann in enumerate(gt):
                if gt_matched[gi] and not ann["crowd"]:
                    continue
                if match >= 0 and not gt[match]["ignored"] and ann["ignored"]:
                    break
                if ious[di, gi] >= best_iou:
                    best_iou, match = float(ious[di, gi]), gi
            if match >= 0:
                gt_matched[match] = True
                if gt[match]["ignored"]:
                    ignored_dt += 1
                else:
                    tp += 1
            else:
                det_area = bbox_area(det["bbox"])
                if det_area < low or det_area > high:
                    ignored_dt += 1
                else:
                    fp += 1
        result["tp"].append(int(tp))
        result["fp"].append(int(fp))
        result["fn"].append(int(valid_gt - tp))
        result["ignored_dt"].append(int(ignored_dt))
    return result


def eval_one_image(image_id, categories, gt_by_id, pred_by_id):
    total = {"gt": 0, "ignored_gt": 0,
             "tp": np.zeros(len(IOU_THRS), dtype=np.int64),
             "fp": np.zeros(len(IOU_THRS), dtype=np.int64),
             "fn": np.zeros(len(IOU_THRS), dtype=np.int64),
             "ignored_dt": np.zeros(len(IOU_THRS), dtype=np.int64)}
    per_category = {}
    for cat in categories:
        gt = gt_by_id.get(image_id, {}).get(cat, [])
        dt = pred_by_id.get(image_id, {}).get(cat, [])
        if not gt and not dt:
            item = {"gt": 0, "ignored_gt": 0, "tp": [0]*10,
                    "fp": [0]*10, "fn": [0]*10, "ignored_dt": [0]*10}
        else:
            item = eval_one_image_category(gt, dt)
        per_category[cat] = item
        total["gt"] += item["gt"]
        total["ignored_gt"] += item["ignored_gt"]
        for key in ("tp", "fp", "fn", "ignored_dt"):
            total[key] += np.asarray(item[key], dtype=np.int64)
    return total, per_category


def make_metrics(counts):
    out = {"small_gt_count": int(counts["gt"]),
           "small_ignored_gt_count": int(counts["ignored_gt"])}
    per_iou = []
    for i in range(10):
        tp, fp, fn = int(counts["tp"][i]), int(counts["fp"][i]), int(counts["fn"][i])
        p, r, f1 = prf(tp, fp, fn)
        per_iou.append((tp, fp, fn, p, r, f1))
    for suffix, idx in (("50", 0), ("75", 5)):
        tp, fp, fn, p, r, f1 = per_iou[idx]
        for key, value in (("tp", tp), ("fp", fp), ("fn", fn),
                           ("precision", p), ("recall", r), ("f1", f1)):
            out[f"small_{key}{suffix}" if key != "f1" else f"small_f1_{suffix}"] = value
    out["small_mean_tp_50_95"] = float(np.mean(counts["tp"]))
    out["small_mean_fp_50_95"] = float(np.mean(counts["fp"]))
    out["small_mean_fn_50_95"] = float(np.mean(counts["fn"]))
    out["small_mean_ignored_dt_50_95"] = float(np.mean(counts["ignored_dt"]))
    out["small_mean_precision_50_95"] = float(np.mean([x[3] for x in per_iou]))
    out["small_mean_recall_50_95"] = float(np.mean([x[4] for x in per_iou]))
    out["small_mean_f1_50_95"] = float(np.mean([x[5] for x in per_iou]))
    return out


def add_cat_aggregates(aggregate, per_category):
    for cat, item in per_category.items():
        aggregate[cat]["gt"] += int(item["gt"])
        for key in ("tp", "fp", "fn"):
            aggregate[cat][key] += np.asarray(item[key], dtype=np.int64)


def source_summary(rows, source, prefix):
    selected = [r for r in rows if r["source"] == source]
    out = {"images": len(selected)}
    for suffix in ("50", "75"):
        tp = sum(int(r[f"{prefix}_small_tp{suffix}"]) for r in selected)
        fp = sum(int(r[f"{prefix}_small_fp{suffix}"]) for r in selected)
        fn = sum(int(r[f"{prefix}_small_fn{suffix}"]) for r in selected)
        p, rec, f1 = prf(tp, fp, fn)
        out.update({f"micro_tp_{suffix}": tp, f"micro_fp_{suffix}": fp,
                    f"micro_fn_{suffix}": fn, f"micro_precision_{suffix}": p,
                    f"micro_recall_{suffix}": rec, f"micro_f1_{suffix}": f1})
    for key in ("small_mean_tp_50_95", "small_mean_fp_50_95", "small_mean_fn_50_95",
                "small_mean_precision_50_95", "small_mean_recall_50_95", "small_mean_f1_50_95"):
        out[f"mean_per_image_{key}"] = float(np.mean([r[f"{prefix}_{key}"] for r in selected]))
    return out


# --------------- Cross-check with official COCO evaluator ---------------
def verify_faster_recall(pred_path, image_ids, categories, manual_agg, label):
    coco_gt = COCO(str(GT_JSON))
    coco_dt = coco_gt.loadRes(str(pred_path))
    evaluator = COCOeval_faster(coco_gt, coco_dt, iouType="bbox", lvis_style=False,
                                print_function=lambda *_a, **_k: None,
                                extra_calc=False, separate_eval=False)
    p = evaluator.params
    p.imgIds = sorted(int(x) for x in image_ids)
    p.catIds = sorted(int(x) for x in categories)
    p.iouThrs = IOU_THRS.copy()
    p.recThrs = REC_THRS.copy()
    p.maxDets = list(MAX_DETS)
    p.areaRng = [list(x) for x in AREA_RANGES]
    p.areaRngLbl = list(AREA_LABELS)
    p.useCats = 1
    evaluator.evaluate()
    evaluator.accumulate()

    recall = np.asarray(evaluator.eval["recall"], dtype=np.float64)
    expected_shape = (10, len(categories), 4, 3)
    if recall.shape != expected_shape:
        raise RuntimeError(f"Unexpected faster_coco_eval recall shape {recall.shape}; expected {expected_shape}")

    comparisons, max_diff = [], 0.0
    for ci, cat in enumerate(sorted(categories)):
        gt_count = int(manual_agg[cat]["gt"])
        if gt_count == 0:
            continue
        for ti, threshold in enumerate(IOU_THRS):
            manual = float(manual_agg[cat]["tp"][ti] / gt_count)
            official = float(recall[ti, ci, SMALL_AREA_INDEX, MAX_DET_INDEX])
            diff = abs(manual - official)
            max_diff = max(max_diff, diff)
            comparisons.append({"category_id": cat, "iou": float(threshold),
                                "manual_recall": manual, "faster_coco_recall": official,
                                "absolute_difference": diff})

    result = {"prediction": label, "image_count": len(image_ids),
              "max_abs_difference": max_diff, "tolerance": 1e-6,
              "passed": max_diff <= 1e-6, "comparisons": comparisons}
    del evaluator, coco_dt, coco_gt, recall
    gc.collect()
    return result


# -------------------------------- Main ----------------------------------
def main():
    for path in (FEATURE_CSV, A_JSON, C2_JSON, GT_JSON, REFERENCE_JSON):
        if not path.is_file():
            raise FileNotFoundError(path)
    SCENE_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/7] SHA256 check")
    hashes = {"features": sha256_file(FEATURE_CSV), "original": sha256_file(A_JSON),
              "c2": sha256_file(C2_JSON), "gt": sha256_file(GT_JSON)}
    for key, expected in EXPECTED_SHA256.items():
        if hashes[key] != expected:
            raise RuntimeError(f"SHA256 mismatch ({key}): expected={expected}, actual={hashes[key]}")

    print("[2/7] Load features and GT")
    features = load_features()
    categories, gt_by_id = load_gt(features)

    print("[3/7] Load existing predictions")
    a_by_id, a_count = load_predictions(A_JSON, features)
    c2_by_id, c2_count = load_predictions(C2_JSON, features)

    print("[4/7] Check UAVDT original-only path")
    uavdt_check = check_uavdt_original_only(features, a_by_id, c2_by_id)
    if not uavdt_check["passed"]:
        raise RuntimeError(f"UAVDT C2 != A predictions: {uavdt_check}")

    print("[5/7] COCO-style per-image matching")
    aggregate_a = {c: {"gt": 0, "tp": np.zeros(10, dtype=np.int64),
                       "fp": np.zeros(10, dtype=np.int64), "fn": np.zeros(10, dtype=np.int64)} for c in categories}
    aggregate_c2 = {c: {"gt": 0, "tp": np.zeros(10, dtype=np.int64),
                        "fp": np.zeros(10, dtype=np.int64), "fn": np.zeros(10, dtype=np.int64)} for c in categories}
    rows = []

    for n, image_id in enumerate(sorted(features), 1):
        feat = features[image_id]
        a_counts, a_cat = eval_one_image(image_id, categories, gt_by_id, a_by_id)
        c_counts, c_cat = eval_one_image(image_id, categories, gt_by_id, c2_by_id)
        add_cat_aggregates(aggregate_a, a_cat)
        add_cat_aggregates(aggregate_c2, c_cat)
        am, cm = make_metrics(a_counts), make_metrics(c_counts)

        row = {**feat, "label_available": int(feat["source"] == "visdrone")}
        row.pop("image_id", None)
        row["image_id"] = image_id
        for k, v in am.items(): row[f"A_{k}"] = v
        for k, v in cm.items(): row[f"C2_{k}"] = v
        for k in am:
            if k.startswith("small_"):
                row[f"delta_{k}"] = float(cm[k] - am[k])

        if row["label_available"]:
            if row["A_small_gt_count"] > 0:
                gain = row["delta_small_mean_f1_50_95"]
                label = "beneficial" if gain > LABEL_EPS else "harmful" if gain < -LABEL_EPS else "neutral"
                reason = "mean_small_f1_change"
            else:
                gain = row["delta_small_mean_fp_50_95"]
                label = "beneficial" if gain < -LABEL_EPS else "harmful" if gain > LABEL_EPS else "neutral"
                reason = "small_area_fp_change_no_small_gt"
        else:
            label, reason = "not_measured", "uavdt_original_only"
        row["selector_label"] = label
        row["selector_label_reason"] = reason
        rows.append(row)
        if n % 200 == 0 or n == len(features):
            print(f"  {n}/{len(features)} images")

    print("[6/7] Write per-image CSV and verify Recall against faster_coco_eval")
    fields = list(rows[0].keys())
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    all_ids = sorted(features)
    verify_a = verify_faster_recall(A_JSON, all_ids, categories, aggregate_a, "A_Original")
    verify_c2 = verify_faster_recall(C2_JSON, all_ids, categories, aggregate_c2, "C2_area2048")
    verify_passed = verify_a["passed"] and verify_c2["passed"]

    print("[7/7] Write summary")
    source_stats = {
        src: {"A_Original": source_summary(rows, src, "A"),
              "C2_area2048": source_summary(rows, src, "C2")}
        for src in ("uavdt", "visdrone")
    }
    vis_rows = [r for r in rows if r["source"] == "visdrone"]
    label_counts = dict(Counter(r["selector_label"] for r in vis_rows))
    reference = load_json(REFERENCE_JSON)
    ref_hashes = reference.get("prediction_sha256", {})
    if ref_hashes.get("A_Original") != hashes["original"]:
        raise RuntimeError("faster_eval_metrics.json A_Original hash does not match current input")
    if ref_hashes.get("C2_area2048") != hashes["c2"]:
        raise RuntimeError("faster_eval_metrics.json C2_area2048 hash does not match current input")
    ref_gt_hash = reference.get("ground_truth", {}).get("sha256")
    if ref_gt_hash != hashes["gt"]:
        raise RuntimeError("faster_eval_metrics.json GT hash does not match current input")

    evaluator_version = version("faster-coco-eval")
    if evaluator_version != "1.8.0":
        raise RuntimeError(
            f"Expected faster-coco-eval==1.8.0 to preserve Stage 6 protocol, got {evaluator_version}"
        )

    summary = {
        "experiment": "Stage6.3B2_COCO_Aligned_PerImage_SmallObject_Gain",
        "protocol": {
            "official_aggregate_evaluator": "faster_coco_eval",
            "per_image_matcher": "COCO evaluateImg matching semantics",
            "top_k_per_image_category": TOP_K,
            "iou_thresholds": [float(x) for x in IOU_THRS],
            "small_area_range": SMALL_RANGE,
            "small_area_boundary": "inclusive, matching COCO area-range logic",
            "max_dets": MAX_DETS,
            "crowd_iou": "intersection / detection area",
            "important_note": "Per-image Precision/Recall/F1 are diagnostics, not official per-image AP. Final APs and AP50-95 must be evaluated on the complete image set.",
        },
        "input": {"sha256": hashes,
                  "prediction_count": {"A_Original": a_count, "C2_area2048": c2_count}},
        "image_count": {"total": len(rows), "uavdt": 1320, "visdrone": 247,
                        "visdrone_labelled": len(vis_rows)},
        "uavdt_original_only_check": uavdt_check,
        "selector_label_counts_visdrone": label_counts,
        "source_summary": source_stats,
        "official_reference_ap_from_existing_metrics": {
            "A_Original": reference["results"]["A_Original"],
            "C2_area2048": reference["results"]["C2_area2048"],
        },
        "verification": {"passed": verify_passed, "A_Original": verify_a,
                         "C2_area2048": verify_c2},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "faster_coco_eval": evaluator_version},
        "output": {"csv": str(OUT_CSV.relative_to(ROOT)),
                   "summary": str(OUT_JSON.relative_to(ROOT))},
    }
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, allow_nan=False)

    print(json.dumps({"label_counts_visdrone": label_counts,
                      "verification_passed": verify_passed,
                      "A_max_abs_diff": verify_a["max_abs_difference"],
                      "C2_max_abs_diff": verify_c2["max_abs_difference"],
                      "csv": str(OUT_CSV), "summary": str(OUT_JSON)},
                     ensure_ascii=False, indent=2))

    if not verify_passed:
        raise RuntimeError("COCO Recall consistency check FAILED. Do not start Stage 6.3C.")
    print("PASS: manual per-image small Recall matches faster_coco_eval.")
    print("No model inference or training was performed.")


if __name__ == "__main__":
    main()
