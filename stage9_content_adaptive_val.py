#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 9: source-independent content-adaptive C2-2048 on HighwayVehicle Val.

Run in YOLO_Thesis repository root. Reuses exact Stage 7.1 predictions/fusion.
No YOLO retraining. The selector learns only from development Val data.
The OOF aggregate is developmental cross-validation, NOT an independent Test.

Phases:
  --phase preflight
  --phase extract [--resume] [--limit-images 20 --out ...]
  --phase evaluate
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import stage7_1_p2bridge_slice_validation as s7
import stage6_analyze_selector_gain_coco as small_eval

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "runs/detect/stage9_content_adaptive_val"
DEFAULT_FEATURE_META = ROOT / "runs/detect/stage6_scene_selector/corrected_scene_features.csv"
REFERENCE_SUMMARY = ROOT / "runs/detect/stage7_p2bridge_slice/stage7_1_validation/metrics_summary.json"
EXPECTED_HASH = "06f713f6c5513661280d7530a095f196151041c232bbe27ddd3d7b8c0b914a84"
EXPECTED_COUNTS = {"uavdt": 1320, "visdrone": 247}
FEATURE_NAMES = ["n_det_conf025", "median_box_area_640", "tiny_ratio"]
THRESHOLD = 0.5      # predetermined, NOT tuned on held-out OOF folds
FOLDS = 5
LABEL_EPS = 1e-12


def read_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    os.replace(tmp, p)


def save_csv(p, rows):
    if not rows:
        raise RuntimeError(f"Cannot write empty CSV: {p}")
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_metadata(path):
    by_id = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        rdr = csv.DictReader(f)
        required = {"image_id", "file_name", "source", "sequence_id"}
        if not required.issubset(set(rdr.fieldnames or [])):
            raise RuntimeError(f"Meta CSV missing columns {required}")
        for r in rdr:
            iid = int(r["image_id"])
            if iid in by_id:
                raise RuntimeError(f"Duplicate metadata image_id={iid}")
            seq = str(r["sequence_id"]).strip()
            if not seq:
                raise RuntimeError(f"Missing sequence_id: {iid}")
            by_id[iid] = dict(
                image_id=iid,
                file_name=Path(r["file_name"]).name,
                source=r["source"].strip().lower(),
                group=f'{r["source"].strip().lower()}::{seq}'
            )
    if len(by_id) != 1567:
        raise RuntimeError(f"Expected 1567 feature metadata rows, got {len(by_id)}")
    return by_id


def setup(args):
    paths = {
        "weights": Path(args.weights).expanduser().resolve(),
        "image_dir": Path(args.image_dir).expanduser().resolve(),
        "gt": Path(args.gt).expanduser().resolve(),
        "source_map": Path(args.source_map).expanduser().resolve(),
        "meta": Path(args.meta).expanduser().resolve(),
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{name}: {path}")

    if version("faster-coco-eval") != "1.8.0":
        raise RuntimeError("Requires faster-coco-eval==1.8.0")
    if sha(paths["weights"]) != EXPECTED_HASH:
        raise RuntimeError("Final checkpoint SHA256 mismatch; do not silently switch model")

    gt, records, cat_id, cat_name, counts = s7.load_records(
        paths["gt"], paths["image_dir"], paths["source_map"]
    )
    if counts != EXPECTED_COUNTS or len(records) != 1567:
        raise RuntimeError(f"Unexpected Val source counts: {counts}")
    metadata = load_metadata(paths["meta"])
    if set(metadata) != {int(r["image_id"]) for r in records}:
        raise RuntimeError("Meta image IDs and Stage7 GT IDs disagree")
    for rec in records:
        m = metadata[int(rec["image_id"])]
        if m["file_name"] != rec["file_name"] or m["source"] != rec["source"]:
            raise RuntimeError(f"Meta filename/source mismatch on {rec['image_id']}")
        rec["group"] = m["group"]
    records.sort(key=lambda r: int(r["image_id"]))

    # Both image source and sequence are audit/split metadata, NEVER selector features.
    fingerprints = {name: sha(path) for name, path in paths.items() if path.is_file()}
    fingerprints.update({
        "stage7_1_sha": sha(Path(s7.__file__).resolve()),
        "source_feature_columns": FEATURE_NAMES,
        "val_total": len(records),
        "protocol": {
            "imgsz": s7.IMGSZ, "conf": s7.CONF,
            "nms_iou": s7.NMS_IOU, "max_det": s7.MAX_DET,
            "overlap": s7.OVERLAP, "area_threshold": s7.TILE_AREA,
        }
    })
    return paths, gt, records, cat_id, fingerprints


def subsample_for_smoke(records, n):
    if not n:
        return records
    if n < 2 or n % 2:
        raise RuntimeError("--limit-images must be an even integer >= 2")
    groups = {s: [r for r in records if r["source"] == s] for s in EXPECTED_COUNTS}
    half = n // 2
    if any(len(g) < half for g in groups.values()):
        raise RuntimeError("Smoke request exceeds one source's sample size")
    return sorted([*groups["uavdt"][:half], *groups["visdrone"][:half]],
                  key=lambda r: int(r["image_id"]))


def feature_from_original(original, width, height):
    b, scores, _ = original
    ids = scores >= 0.25
    b = b[ids]
    n = len(b)
    if not n:
        return dict(n_det_conf025=0, median_box_area_640=None, tiny_ratio=0.0)
    scale = min(640 / float(width), 640 / float(height))
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) * scale**2
    if np.any(area <= 0) or not np.isfinite(area).all():
        raise RuntimeError("Invalid predicted box areas")
    return dict(
        n_det_conf025=int(n),
        median_box_area_640=round(float(np.median(area)), 4),
        tiny_ratio=round(float((area < 1024).mean()), 6),
    )


def prepare_model(args):
    import cv2
    import torch
    from torchvision.ops import batched_nms
    from ultralytics import YOLO
    model = YOLO(str(Path(args.weights).expanduser().resolve()))
    if len(model.names) != 1:
        raise RuntimeError(f"Expected one vehicle category, got {model.names}")
    pred_args = s7.pred_args(args.device)
    return cv2, torch, batched_nms, model, pred_args


def preflight(args, records):
    cv2, torch, bnms, model, pred_args = prepare_model(args)
    for source in EXPECTED_COUNTS:
        rec = next(r for r in records if r["source"] == source)
        img = cv2.imread(str(rec["path"]))
        if img is None:
            raise RuntimeError(f"Cannot read {rec['path']}")
        orig = s7.predict_original(model, img, pred_args, torch, bnms)
        tiles = s7.predict_tiles(model, img, pred_args, torch, bnms)
        fused = s7.fuse_c2(orig, tiles, img.shape[1], img.shape[0], torch, bnms)
        print(f"[PASS] {source}: original={len(orig[0])} fused={len(fused[0])}")
    print("PREFLIGHT PASS: exact Stage7.1 pipeline, one source-independent route")


def read_jsonl(path, valid_ids):
    result = {}
    if not path.exists():
        return result
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                raise RuntimeError(
                    f"Malformed checkpoint line {line_number}; inspect {path}"
                ) from exc
            iid = int(row["image_id"])
            if iid not in valid_ids or iid in result:
                raise RuntimeError(f"Duplicate/unknown checkpoint image_id={iid}")
            result[iid] = row
    return result


def run_extract(args, paths, records, cat_id, fingerprints):
    out = Path(args.out).resolve()
    selected = subsample_for_smoke(records, args.limit_images)
    record_file = out / "paired_predictions.jsonl"
    manifest_file = out / "extract_manifest.json"
    config = {
        "inputs": fingerprints, "images": len(selected),
        "image_ids_sha256": hashlib.sha256(
            ",".join(str(x["image_id"]) for x in selected).encode()
        ).hexdigest(),
        "experiment": "Stage9_Content_Adaptive_Val",
        "limit_images": args.limit_images,
    }
    if manifest_file.exists():
        if read_json(manifest_file) != config:
            raise RuntimeError("Resume manifest differs. Use a fresh --out directory.")
        if not args.resume:
            raise FileExistsError("Existing extraction. Add --resume or choose another --out")
    else:
        if record_file.exists():
            raise FileExistsError(f"Checkpoint exists without manifest: {record_file}")
        save_json(manifest_file, config)

    valid_ids = {r["image_id"] for r in selected}
    already = read_jsonl(record_file, valid_ids)
    print(f"[EXTRACT] completed {len(already)}/{len(selected)}", flush=True)
    cv2, torch, bnms, model, pred_args = prepare_model(args)
    first = cv2.imread(str(selected[0]["path"]))
    if first is None:
        raise RuntimeError("Cannot read warmup image")
    for _ in range(3):
        s7.sync(torch, args.device)
        s7.predict_original(model, first, pred_args, torch, bnms)
        s7.predict_tiles(model, first, pred_args, torch, bnms)
        s7.sync(torch, args.device)

    with open(record_file, "a", encoding="utf-8", buffering=1) as f:
        for rec in selected:
            iid = rec["image_id"]
            if iid in already:
                continue
            img = cv2.imread(str(rec["path"]))
            if img is None:
                raise RuntimeError(f"Cannot read {rec['path']}")
            height, width = img.shape[:2]
            if (width, height) != (rec["width"], rec["height"]):
                raise RuntimeError(f"Image/COCO GT dimensions mismatch: {iid}")

            s7.sync(torch, args.device)
            start = time.perf_counter()
            orig = s7.predict_original(model, img, pred_args, torch, bnms)
            s7.sync(torch, args.device)
            original_ms = (time.perf_counter() - start) * 1000
            s7.sync(torch, args.device)
            start = time.perf_counter()
            tiled = s7.predict_tiles(model, img, pred_args, torch, bnms)
            s7.sync(torch, args.device)
            tiled_ms = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            fused = s7.fuse_c2(orig, tiled, width, height, torch, bnms)
            s7.sync(torch, args.device)
            fusion_ms = (time.perf_counter() - start) * 1000

            row = {
                "image_id": iid, "source": rec["source"], "group": rec["group"],
                "file_name": rec["file_name"],
                "features": feature_from_original(orig, width, height),
                "original": s7.coco_rows(iid, orig, cat_id),
                "full_c2": s7.coco_rows(iid, fused, cat_id),
                "original_ms": original_ms,
                "tile_ms": tiled_ms, "fusion_ms": fusion_ms,
            }
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
            already[iid] = True
            if len(already) % 10 == 0 or len(already) == len(selected):
                f.flush()
                os.fsync(f.fileno())
                print(f"[EXTRACT] {len(already)}/{len(selected)}", flush=True)
    print(f"EXTRACT COMPLETED: {len(already)} records in {record_file}", flush=True)


def load_utility_targets(gt, predictions):
    # Reuse Stage6.3B2 per-image small-object COCO-style 10-IoU matching.
    gt_by_id = defaultdict(lambda: defaultdict(list))
    for ann in gt["annotations"]:
        gt_by_id[int(ann["image_id"])][int(ann["category_id"])].append({
            "id": int(ann["id"]),
            "bbox": [float(v) for v in ann["bbox"]],
            "area": float(ann["area"]),
            "iscrowd": int(ann.get("iscrowd", 0)),
            "ignore": int(ann.get("ignore", ann.get("iscrowd", 0))),
        })
    categories = [int(x["id"]) for x in gt["categories"]]
    targets, utility_rows = [], []
    for row in predictions:
        iid = int(row["image_id"])
        original = defaultdict(list)
        full_c2 = defaultdict(list)
        for pred in row["original"]:
            original[int(pred["category_id"])].append(pred)
        for pred in row["full_c2"]:
            full_c2[int(pred["category_id"])].append(pred)
        for d in (original, full_c2):
            for key in d:
                d[key].sort(key=lambda x: -float(x["score"]))
        am, _ = small_eval.eval_one_image(iid, categories, gt_by_id, {iid: original})
        bm, _ = small_eval.eval_one_image(iid, categories, gt_by_id, {iid: full_c2})
        a = small_eval.make_metrics(am)
        b = small_eval.make_metrics(bm)
        if a["small_gt_count"] > 0:
            gain = b["small_mean_f1_50_95"] - a["small_mean_f1_50_95"]
            reason = "mean_small_F1_change"
        else:
            gain = a["small_mean_fp_50_95"] - b["small_mean_fp_50_95"]
            reason = "negative_small_FP_change_no_small_GT"
        label = "beneficial" if gain > LABEL_EPS else ("harmful" if gain < -LABEL_EPS else "neutral")
        targets.append(1 if gain > LABEL_EPS else 0)
        utility_rows.append({
            "image_id": iid, "source": row["source"], "group": row["group"],
            "small_gt": a["small_gt_count"], "gain": float(gain),
            "label": label, "reason": reason,
        })
    return np.asarray(targets, dtype=np.int64), utility_rows


def feature_matrix(rows):
    # Only fields measurable at inference from first-pass detections.
    X = []
    for row in rows:
        f = row["features"]
        median = f["median_box_area_640"]
        X.append([
            np.log1p(float(f["n_det_conf025"])),
            np.log1p(float(median)) if median is not None else np.nan,
            float(f["tiny_ratio"]),
        ])
    return np.asarray(X, dtype=np.float64)


def make_selector():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=0.01, class_weight="balanced", max_iter=2000,
            random_state=42, solver="liblinear"
        )),
    ])


def build_oof(rows, targets):
    groups = np.asarray([r["group"] for r in rows])
    X = feature_matrix(rows)
    cv = GroupKFold(n_splits=FOLDS)
    p = np.full(len(rows), np.nan, dtype=np.float64)
    fold_id = np.zeros(len(rows), dtype=np.int64)
    audit = []
    for k, (train, heldout) in enumerate(cv.split(X, targets, groups), 1):
        if len(np.unique(targets[train])) < 2:
            raise RuntimeError(f"OOF fold {k} training has one class. Do not fake scores.")
        if set(groups[train]) & set(groups[heldout]):
            raise RuntimeError(f"Sequence leakage in fold {k}")
        selector = make_selector().fit(X[train], targets[train])
        c = list(selector.named_steps["classifier"].classes_)
        p[heldout] = selector.predict_proba(X[heldout])[:, c.index(1)]
        fold_id[heldout] = k
        audit.append({
            "fold": k, "train_images": len(train), "heldout_images": len(heldout),
            "train_groups": len(set(groups[train])),
            "heldout_groups": len(set(groups[heldout])),
            "beneficial_train": int(targets[train].sum()),
            "beneficial_heldout": int(targets[heldout].sum()),
            "triggered_heldout": int((p[heldout] >= THRESHOLD).sum()),
            "group_overlap": 0,
        })
    if not np.isfinite(p).all() or np.any(fold_id == 0):
        raise RuntimeError("Incomplete OOF coverage")
    return p, fold_id, audit


def evaluate(args, gt, records, cat_id, fingerprints):
    out = Path(args.out).resolve()
    manifest_file = out / "extract_manifest.json"
    record_file = out / "paired_predictions.jsonl"
    if not manifest_file.is_file() or not record_file.is_file():
        raise FileNotFoundError("Run --phase extract first")
    manifest = read_json(manifest_file)
    if manifest["inputs"] != fingerprints:
        raise RuntimeError("Extract inputs changed since snapshot: refuse evaluation")
    if manifest["images"] != 1567 or manifest["limit_images"] != 0:
        raise RuntimeError("Only full 1,567-image extraction may be evaluated")
    got = read_jsonl(record_file, {r["image_id"] for r in records})
    if set(got) != {r["image_id"] for r in records}:
        raise RuntimeError(f"Missing data: {len(got)}/{len(records)} images")

    rows = [got[r["image_id"]] for r in records]
    targets, utilities = load_utility_targets(gt, rows)
    oof_p, fold_id, audits = build_oof(rows, targets)
    decisions = oof_p >= THRESHOLD

    modes = {
        "A_original": [],
        "B_all_C2": [],
        "C_source_C2": [],
        "D_adaptive_OOF": [],
    }
    decision_rows = []
    for i, row in enumerate(rows):
        use_source_c2 = row["source"] == "visdrone"
        use_adaptive = bool(decisions[i])
        modes["A_original"].extend(row["original"])
        modes["B_all_C2"].extend(row["full_c2"])
        modes["C_source_C2"].extend(row["full_c2"] if use_source_c2 else row["original"])
        modes["D_adaptive_OOF"].extend(row["full_c2"] if use_adaptive else row["original"])
        decision_rows.append({
            **utilities[i],
            **row["features"],
            "fold": int(fold_id[i]),
            "oof_probability": round(float(oof_p[i]), 8),
            "select_c2": int(use_adaptive),
            "original_ms": float(row["original_ms"]),
            "tile_ms": float(row["tile_ms"]),
            "fusion_ms": float(row["fusion_ms"]),
        })

    pred_dir = out / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_paths = {}
    for name, preds in modes.items():
        path = pred_dir / f"{name}.json"
        save_json(path, preds)
        pred_paths[name] = path
        print(f"[PREDICTIONS] {name}: {len(preds)} boxes", flush=True)

    scopes = {
        "full": [r["image_id"] for r in records],
        "uavdt": [r["image_id"] for r in records if r["source"] == "uavdt"],
        "visdrone": [r["image_id"] for r in records if r["source"] == "visdrone"],
    }
    metrics = {}
    for mode, path in pred_paths.items():
        metrics[mode] = {}
        for scope, ids in scopes.items():
            print(f"[COCO] {mode}/{scope}/{len(ids)}", flush=True)
            metrics[mode][scope] = s7.eval_scope(
                Path(args.gt).resolve(), path, ids, cat_id
            )

    # Internal consistency checks with fixed Stage7.1 A & source-aware C.
    if REFERENCE_SUMMARY.is_file():
        ref = read_json(REFERENCE_SUMMARY)
        for mode, ref_mode in [("A_original", "A_original"),
                               ("C_source_C2", "B_C2_area2048")]:
            for scope in scopes:
                for metric in ["AP50-95", "APs", "AP50"]:
                    current = metrics[mode][scope][metric]
                    previous = ref["metrics"][ref_mode][scope][metric]
                    if abs(current - previous) > 5e-4:
                        raise RuntimeError(
                            f"Stage7 reproduction failed: {mode}/{scope}/{metric}, "
                            f"new={current:.7f} old={previous:.7f}"
                        )
        print("[PASS] A_original and C_source_C2 reproduce Stage7.1", flush=True)

    def mean_runtime(kind):
        arr = []
        for i, r in enumerate(rows):
            value = r["original_ms"]
            if kind == "B_all_C2" or (
                kind == "C_source_C2" and r["source"] == "visdrone"
            ) or (kind == "D_adaptive_OOF" and decisions[i]):
                value += r["tile_ms"] + r["fusion_ms"]
            arr.append(value)
        return float(np.mean(arr))

    timing = {
        mode: {
            "inference_plus_fusion_ms_per_image_component_sum": mean_runtime(mode),
            "approximate_FPS_from_component_sum": 1000.0 / mean_runtime(mode),
            "timing_note": (
                "Offline component-sum estimate; D timing includes all-images "
                "counterfactual tile inference to estimate, not a live gated runtime. "
                "Model-feature/decision overhead not separately measured."
            ),
        } for mode in modes
    }
    trigger = {
        "overall": int(decisions.sum()),
        "overall_rate": float(decisions.mean()),
        "uavdt": int(sum(decisions[i] for i,r in enumerate(rows) if r["source"] == "uavdt")),
        "visdrone": int(sum(decisions[i] for i,r in enumerate(rows) if r["source"] == "visdrone")),
    }
    save_csv(out / "selector_utility_and_oof_decisions.csv", decision_rows)
    save_json(out / "fold_audit.json", audits)

    # Fit a final web/deployment selector after OOF evaluation. Never report
    # its training-set predictions as OOF / unbiased evidence.
    if len(set(targets)) < 2:
        raise RuntimeError("Final selector cannot train: one label class")
    final_selector = make_selector().fit(feature_matrix(rows), targets)
    import joblib
    joblib.dump(final_selector, out / "final_content_selector_val.joblib")
    save_json(out / "final_selector_metadata.json", {
        "input_features": FEATURE_NAMES,
        "feature_transform": [
            "log1p(n_det_conf025)", "log1p(median_box_area_640)",
            "tiny_ratio",
        ],
        "missing_median_input": "NaN; pipeline imputes 0",
        "threshold": THRESHOLD,
        "model": "LogisticRegression(C=0.01,class_weight=balanced)",
        "training_images": 1567,
        "evaluation": "Five-fold sequence-group OOF, development-only",
        "no_source_feature": True,
        "warning": "OOF Val is not an independent external deployment accuracy estimate",
    })
    summary = {
        "experiment": "Stage9_Content_Adaptive_C2_Val",
        "status": "completed",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": "HighwayVehicle-v1.0 / Val / 1567 images",
        "input_fingerprints": fingerprints,
        "model_checkpoint_sha256": EXPECTED_HASH,
        "selector": {
            "features": FEATURE_NAMES,
            "algorithm": "5-fold GroupKFold LogisticRegression",
            "threshold": THRESHOLD,
            "sources_used_as_features": False,
            "trigger": trigger,
            "positive_label": "small-object mean F1 gain, or lower small-area FP when no small GT",
            "development_oof_only": True,
        },
        "metrics": metrics,
        "timing": timing,
        "note": "B applies C2 on all Val images; C reproduces Stage7 source-aware C2; D is group-OOF learned from all Val images.",
    }
    save_json(out / "metrics_summary.json", summary)
    print("=" * 70)
    print("STAGE 9 COMPLETE: Val-only source-independent adaptive slicing")
    for key, val in metrics.items():
        print(key, "AP50-95", val["full"]["AP50-95"],
              "APs", val["full"]["APs"])
    print("OOF triggered:", trigger)
    print("Results:", out / "metrics_summary.json")
    print("=" * 70)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", required=True, choices=["preflight", "extract", "evaluate"])
    ap.add_argument("--weights", default=str(s7.DEFAULTS["weights"]))
    ap.add_argument("--image-dir", default=str(s7.DEFAULTS["image_dir"]))
    ap.add_argument("--gt", default=str(s7.DEFAULTS["gt"]))
    ap.add_argument("--source-map", default=str(s7.DEFAULTS["source_map"]))
    ap.add_argument("--meta", default=str(DEFAULT_FEATURE_META))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--device", default="0")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit-images", type=int, default=0,
                    help="Preflight/smoke only; evaluate rejects limited extracts")
    args = ap.parse_args()
    paths, gt, records, cat_id, fingerprints = setup(args)
    if args.phase == "preflight":
        preflight(args, records)
    elif args.phase == "extract":
        run_extract(args, paths, records, cat_id, fingerprints)
    else:
        evaluate(args, gt, records, cat_id, fingerprints)


if __name__ == "__main__":
    main()
