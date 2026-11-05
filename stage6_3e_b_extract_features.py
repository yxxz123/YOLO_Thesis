#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 6.3E-B-1: extract and audit expanded scene features.

Run from YOLO_Thesis repository root:
    python -u stage6_3e_b_extract_features.py

Reads:
  runs/detect/stage6_scene_selector/corrected_scene_features.csv
  runs/detect/stage6_selective_oracle/all_original_predictions.json

Writes only:
  runs/detect/stage6_scene_selector/stage6_3e_b_features/

No YOLO training/inference. No utility labels are read by this script.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent
SCENE_DIR = ROOT / "runs/detect/stage6_scene_selector"
FEATURES_CSV = SCENE_DIR / "corrected_scene_features.csv"
ORIGINAL_JSON = ROOT / "runs/detect/stage6_selective_oracle/all_original_predictions.json"

OUT_DIR = SCENE_DIR / "stage6_3e_b_features"
OUT_FEATURES = OUT_DIR / "expanded_scene_features.csv"
OUT_AUDIT = OUT_DIR / "feature_audit.json"
OUT_DEFINITIONS = OUT_DIR / "feature_definitions.json"

EXPECTED_TOTAL = 1567
EXPECTED_SOURCE_COUNTS = {"uavdt": 1320, "visdrone": 247}
FEATURE_IMGSZ = 640
FEATURE_CONF = 0.25
TINY_AREA_THRESHOLD = 32.0 ** 2

BASE_FEATURES = [
    "n_det_conf025",
    "median_box_area_640",
    "tiny_ratio",
]
EXPANDED_FEATURES = BASE_FEATURES + [
    "tiny_count",
    "det_density_per_mp",
    "mean_confidence_025",
    "box_area_cv_640",
    "grid_occupancy_4x4",
]


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


def read_feature_metadata(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "image_id", "file_name", "source", "sequence_id",
            "width", "height", "letterbox_scale", "n_det_conf025",
            "median_box_area_640", "tiny_ratio", "tiny_count",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"原始特征CSV缺少字段：{sorted(missing)}")
        for raw in reader:
            image_id = int(raw["image_id"])
            if image_id in rows:
                raise RuntimeError(f"原始特征CSV存在重复image_id={image_id}")
            width, height = int(raw["width"]), int(raw["height"])
            if width <= 0 or height <= 0:
                raise RuntimeError(f"非法图像尺寸：image_id={image_id}, {width}x{height}")
            rows[image_id] = {
                "image_id": image_id,
                "file_name": raw["file_name"],
                "source": raw["source"].strip().lower(),
                "sequence_id": raw["sequence_id"],
                "width": width,
                "height": height,
                "letterbox_scale": float(raw["letterbox_scale"]),
                "stored_n_det_conf025": int(raw["n_det_conf025"]),
                "stored_median_box_area_640": (
                    None if raw["median_box_area_640"].strip() == ""
                    else float(raw["median_box_area_640"])
                ),
                "stored_tiny_ratio": float(raw["tiny_ratio"]),
                "stored_tiny_count": int(float(raw["tiny_count"])),
            }
    if len(rows) != EXPECTED_TOTAL:
        raise RuntimeError(f"图片数={len(rows)}，预期={EXPECTED_TOTAL}")
    counts = Counter(r["source"] for r in rows.values())
    if dict(counts) != EXPECTED_SOURCE_COUNTS:
        raise RuntimeError(f"来源数量不匹配：{dict(counts)}")
    if any(r["source"] not in EXPECTED_SOURCE_COUNTS for r in rows.values()):
        raise RuntimeError("出现未知数据来源")
    return rows


def read_original_predictions(path: Path, valid_ids: set[int]) -> dict[int, list[dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise RuntimeError("all_original_predictions.json必须是COCO detection list")
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, pred in enumerate(data):
        try:
            image_id = int(pred["image_id"])
            score = float(pred["score"])
            bbox = [float(v) for v in pred["bbox"]]
        except Exception as exc:
            raise RuntimeError(f"第{index}条原始预测记录字段错误：{exc}") from exc
        if image_id not in valid_ids:
            raise RuntimeError(f"原始预测出现未知image_id={image_id}")
        if len(bbox) != 4 or not all(math.isfinite(x) for x in bbox):
            raise RuntimeError(f"非法bbox：index={index}, bbox={bbox}")
        if not math.isfinite(score) or score < 0 or score > 1:
            raise RuntimeError(f"非法score：index={index}, score={score}")
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            raise RuntimeError(f"预测框宽高非正：index={index}, bbox={bbox}")
        grouped[image_id].append({
            "score": score,
            "bbox": bbox,
        })
    return dict(grouped)


def extract_one(meta: dict[str, Any], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    width, height = meta["width"], meta["height"]
    scale = min(FEATURE_IMGSZ / float(width), FEATURE_IMGSZ / float(height))

    kept = [p for p in predictions if p["score"] >= FEATURE_CONF]
    n_det = len(kept)

    if n_det == 0:
        return {
            "n_det_conf025": 0,
            "median_box_area_640": "",
            "tiny_ratio": 0.0,
            "tiny_count": 0,
            "det_density_per_mp": 0.0,
            "mean_confidence_025": 0.0,
            "box_area_cv_640": 0.0,
            "grid_occupancy_4x4": 0.0,
        }

    areas: list[float] = []
    grid_cells: set[tuple[int, int]] = set()
    scores: list[float] = []

    for p in kept:
        x, y, w, h = p["bbox"]
        area640 = (w * scale) * (h * scale)
        if not math.isfinite(area640) or area640 <= 0:
            raise RuntimeError(f"非法640尺度面积：{meta['image_id']} {area640}")
        areas.append(area640)
        scores.append(float(p["score"]))

        cx = (x + 0.5 * w) / float(width)
        cy = (y + 0.5 * h) / float(height)
        # Clamp out-of-frame centers to the nearest valid cell.
        gx = min(3, max(0, int(math.floor(cx * 4.0))))
        gy = min(3, max(0, int(math.floor(cy * 4.0))))
        grid_cells.add((gx, gy))

    area_array = np.asarray(areas, dtype=np.float64)
    mean_area = float(np.mean(area_array))
    std_area = float(np.std(area_array, ddof=0))
    tiny_count = int(np.sum(area_array < TINY_AREA_THRESHOLD))
    tiny_ratio = tiny_count / float(n_det)
    megapixels = (width * height) / 1_000_000.0

    return {
        "n_det_conf025": int(n_det),
        "median_box_area_640": round(float(np.median(area_array)), 4),
        "tiny_ratio": round(float(tiny_ratio), 6),
        "tiny_count": tiny_count,
        "det_density_per_mp": round(float(n_det / megapixels), 8),
        "mean_confidence_025": round(float(np.mean(scores)), 8),
        "box_area_cv_640": round(float(std_area / mean_area), 8) if mean_area > 0 else 0.0,
        "grid_occupancy_4x4": round(float(len(grid_cells) / 16.0), 8),
    }


def spearman_matrix(rows: list[dict[str, Any]], features: list[str]) -> dict[str, Any]:
    matrix: dict[str, dict[str, float | None]] = {f: {} for f in features}
    for fa in features:
        for fb in features:
            pairs = []
            for row in rows:
                a, b = row.get(fa), row.get(fb)
                if a in ("", None) or b in ("", None):
                    continue
                a, b = float(a), float(b)
                if math.isfinite(a) and math.isfinite(b):
                    pairs.append((a, b))
            if len(pairs) < 3:
                matrix[fa][fb] = None
                continue
            a = np.asarray([p[0] for p in pairs], dtype=np.float64)
            b = np.asarray([p[1] for p in pairs], dtype=np.float64)
            ra, rb = rankdata(a), rankdata(b)
            if np.std(ra) == 0 or np.std(rb) == 0:
                matrix[fa][fb] = None
            else:
                matrix[fa][fb] = float(np.corrcoef(ra, rb)[0, 1])
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true",
                        help="仅覆盖stage6_3e_b_features目录中的本阶段文件")
    args = parser.parse_args()

    for path in (FEATURES_CSV, ORIGINAL_JSON):
        if not path.is_file():
            raise FileNotFoundError(path)

    if OUT_DIR.exists() and any(OUT_DIR.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"输出目录已有文件：{OUT_DIR}。检查后如需重跑，显式添加 --overwrite。"
        )

    print("=" * 84)
    print("Stage 6.3E-B-1 | Expanded scene feature extraction")
    print("No YOLO training. No YOLO inference. No labels used for feature extraction.")
    print("=" * 84)

    metadata = read_feature_metadata(FEATURES_CSV)
    valid_ids = set(metadata)
    pred_by_id = read_original_predictions(ORIGINAL_JSON, valid_ids)

    output_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for image_id in sorted(metadata):
        meta = metadata[image_id]
        new = extract_one(meta, pred_by_id.get(image_id, []))
        expected_scale = min(FEATURE_IMGSZ / float(meta["width"]),
                             FEATURE_IMGSZ / float(meta["height"]))
        if not math.isclose(expected_scale, meta["letterbox_scale"],
                            rel_tol=0.0, abs_tol=1e-6):
            raise RuntimeError(
                f"letterbox_scale与原始图像尺寸不一致 image_id={image_id}: "
                f"computed={expected_scale} stored={meta['letterbox_scale']}"
            )

        # Reproduce the three established Stage 6.3C features exactly.
        if new["n_det_conf025"] != meta["stored_n_det_conf025"]:
            raise RuntimeError(
                f"n_det_conf025复现失败 image_id={image_id}: "
                f"new={new['n_det_conf025']} old={meta['stored_n_det_conf025']}"
            )
        old_median = meta["stored_median_box_area_640"]
        new_median = new["median_box_area_640"]
        if old_median is None or new_median == "":
            if not (old_median is None and new_median == ""):
                raise RuntimeError(f"median空值状态不一致 image_id={image_id}")
        elif not math.isclose(float(old_median), float(new_median), rel_tol=0.0, abs_tol=1e-3):
            raise RuntimeError(
                f"median_box_area_640复现失败 image_id={image_id}: "
                f"new={new_median} old={old_median}"
            )
        if not math.isclose(new["tiny_ratio"], meta["stored_tiny_ratio"],
                            rel_tol=0.0, abs_tol=1e-6):
            raise RuntimeError(
                f"tiny_ratio复现失败 image_id={image_id}: "
                f"new={new['tiny_ratio']} old={meta['stored_tiny_ratio']}"
            )
        if new["tiny_count"] != meta["stored_tiny_count"]:
            raise RuntimeError(
                f"tiny_count复现失败 image_id={image_id}: "
                f"new={new['tiny_count']} old={meta['stored_tiny_count']}"
            )

        row = {
            "image_id": image_id,
            "file_name": meta["file_name"],
            "source": meta["source"],
            "sequence_id": meta["sequence_id"],
            "width": meta["width"],
            "height": meta["height"],
            "letterbox_scale": meta["letterbox_scale"],
            **new,
        }
        output_rows.append(row)
        audit_rows.append({
            "image_id": image_id,
            "source": meta["source"],
            "sequence_id": meta["sequence_id"],
            "original_prediction_count": len(pred_by_id.get(image_id, [])),
            "kept_conf025_count": new["n_det_conf025"],
            "base_feature_reproduction": "passed",
        })

    vis_rows = [r for r in output_rows if r["source"] == "visdrone"]
    groups = sorted({r["sequence_id"] for r in vis_rows})
    if len(vis_rows) != 247 or len(groups) != 94:
        raise RuntimeError(f"VisDrone数量/序列数异常：images={len(vis_rows)}, groups={len(groups)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv_atomic(OUT_FEATURES, output_rows)
    write_csv_atomic(OUT_DIR / "feature_image_audit.csv", audit_rows)

    correlation = spearman_matrix(vis_rows, EXPANDED_FEATURES)
    definitions = {
        "F0_original_3_features": BASE_FEATURES,
        "F1_expanded_8_features": EXPANDED_FEATURES,
        "feature_definitions": {
            "n_det_conf025": "Number of original-image detections with score >= 0.25.",
            "median_box_area_640": "Median detection box area after letterbox scale to 640; same definition as Stage 6.3C.",
            "tiny_ratio": "Fraction of detections whose 640-scale area is < 32^2; same definition as Stage 6.3C.",
            "tiny_count": "Absolute count of detections whose 640-scale area is < 32^2.",
            "det_density_per_mp": "Number of score>=0.25 detections divided by original image megapixels.",
            "mean_confidence_025": "Mean confidence of detections with score >= 0.25.",
            "box_area_cv_640": "Population standard deviation / mean of 640-scale detection box areas.",
            "grid_occupancy_4x4": "Number of occupied cells among a 4x4 grid using normalized original-image box centers, divided by 16.",
        },
        "calculation": {
            "source_predictions": "A_Original only",
            "feature_confidence_threshold": FEATURE_CONF,
            "feature_imgsz": FEATURE_IMGSZ,
            "tiny_area_threshold_640": TINY_AREA_THRESHOLD,
            "coordinate_convention": "COCO xywh in original-image pixel coordinates",
            "empty_detection_handling": {
                "median_box_area_640": "empty CSV cell; converted to NaN by the model pipeline",
                "other_numeric_features": 0,
            },
        },
    }
    write_json_atomic(OUT_DEFINITIONS, definitions)

    audit = {
        "experiment": "Stage6.3E-B_Expanded_Scene_Feature_Extraction",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "completed",
        "no_yolo_training": True,
        "no_yolo_inference": True,
        "utility_labels_read": False,
        "source_counts": dict(Counter(r["source"] for r in output_rows)),
        "visdrone_images": len(vis_rows),
        "visdrone_sequence_groups": len(groups),
        "base_feature_reproduction": {
            "checked_images": len(output_rows),
            "n_det_conf025": "passed",
            "median_box_area_640": "passed",
            "tiny_ratio": "passed",
            "tiny_count": "passed",
            "letterbox_scale": "passed",
        },
        "input_sha256": {
            "corrected_scene_features.csv": sha256_file(FEATURES_CSV),
            "all_original_predictions.json": sha256_file(ORIGINAL_JSON),
        },
        "output_sha256": {
            "expanded_scene_features.csv": sha256_file(OUT_FEATURES),
        },
        "spearman_correlation_visdrone": correlation,
        "outputs": {
            "features": str(OUT_FEATURES.relative_to(ROOT)),
            "image_audit": str((OUT_DIR / "feature_image_audit.csv").relative_to(ROOT)),
            "definitions": str(OUT_DEFINITIONS.relative_to(ROOT)),
            "audit": str(OUT_AUDIT.relative_to(ROOT)),
        },
    }
    write_json_atomic(OUT_AUDIT, audit)

    print("\n[PASS] Existing three features reproduced exactly within declared tolerances.")
    print("Images:", len(output_rows), "| VisDrone:", len(vis_rows), "| sequences:", len(groups))
    print("Output:", OUT_DIR)
    print("New F1 features:", ", ".join(EXPANDED_FEATURES))
    print("=" * 84)


if __name__ == "__main__":
    main()
