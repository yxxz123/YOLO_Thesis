#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 6.4B: final independent test for frozen M3 Logistic-F0.

Policies:
A_Original:
    Original YOLO11s-DySample inference for every independent-test image.
B_C2_area2048:
    UAVDT original-only; VisDrone full 2x2 tile inference + area-filtered fusion.
C_M3_selective:
    UAVDT original-only; VisDrone uses the frozen M3 selector. Only selected
    VisDrone images receive C2_area2048 tile inference/fusion.

No YOLO training. No selector refitting. No threshold tuning on test.
The independent test manifest must be produced by Stage 6.4A2.
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
from collections import Counter
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from faster_coco_eval import COCO, COCOeval_faster

ROOT = Path(__file__).resolve().parent
DATASET_ROOT_DEFAULT = ROOT / "datasets/HighwayVehicle-v1.0"
FILTER_DIR = ROOT / "runs/detect/stage6_scene_selector/stage6_4_preflight"
MANIFEST_DEFAULT = FILTER_DIR / "independent_test_manifest.csv"
FILTER_SUMMARY = FILTER_DIR / "sequence_filter_summary.json"
AUDIT_JSON = FILTER_DIR / "dataset_audit.json"

WEIGHT_DEFAULT = ROOT / "runs/detect/stage2_screening/YOLO11s-DySample/weights/best.pt"
SELECTOR_DIR = ROOT / "runs/detect/stage6_scene_selector/stage6_3f_c_runtime"
SELECTOR_PATH = SELECTOR_DIR / "final_selector_m3_logistic_f0.joblib"
SELECTOR_META_PATH = SELECTOR_DIR / "final_selector_metadata.json"
BASELINE_VAL_GT = ROOT / "runs/detect/source_validation/Stage5-Baseline-Unified/annotations_auto_detect_val.json"

OUT_DIR_DEFAULT = ROOT / "runs/detect/stage6_scene_selector/stage6_4b_independent_test"

EXPECTED_WEIGHT_SHA256 = "1a63af9a02c3970a71a2971890b84773851875971620c6eb9336becb090030bb"
EXPECTED_SELECTOR_SHA256 = "cbc403801df42769fac96ae30cf5751f83ffd59513d7c5f37733b452f76affda"
EXPECTED_NUMPY = "1.26.4"
EXPECTED_SKLEARN = "1.5.2"
EXPECTED_EVALUATOR = "1.8.0"

EXPECTED_TOTAL = 1372
EXPECTED_SOURCE_COUNTS = {"uavdt": 1288, "visdrone": 84}
EXPECTED_GROUPS = 89

# Fixed inference and fusion protocol: identical to Stage 6.1 / Stage 6.3F-C.
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

FEATURE_NAMES = ["n_det_conf025", "median_box_area_640", "tiny_ratio"]
SELECTOR_THRESHOLD = 0.5071783812207581
CATEGORY_ID = 1  # overwritten from the repository COCO category schema at runtime
CATEGORY_NAME = "vehicle"

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
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    tmp.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"拒绝写入空CSV：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def normalize_name(value: str) -> str:
    return str(value).replace("\\", "/").lstrip("./")


def sequence_id_from_name(filename: str) -> str:
    stem = Path(str(filename)).stem
    parts = stem.rsplit("_", 1)
    return parts[0] if len(parts) == 2 else stem


def check_environment() -> dict[str, str]:
    env = {
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "faster_coco_eval": version("faster-coco-eval"),
    }
    if env["numpy"] != EXPECTED_NUMPY:
        raise RuntimeError(f"NumPy版本不匹配：{env['numpy']}，要求{EXPECTED_NUMPY}")
    if env["scikit_learn"] != EXPECTED_SKLEARN:
        raise RuntimeError(
            f"scikit-learn版本不匹配：{env['scikit_learn']}，要求{EXPECTED_SKLEARN}"
        )
    if env["faster_coco_eval"] != EXPECTED_EVALUATOR:
        raise RuntimeError(
            f"faster-coco-eval版本不匹配：{env['faster_coco_eval']}，"
            f"要求{EXPECTED_EVALUATOR}"
        )
    return env


def validate_inputs(
    dataset_root: Path,
    manifest_path: Path,
    weights: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], Any, dict[str, str]]:
    required = [
        manifest_path, FILTER_SUMMARY, AUDIT_JSON,
        SELECTOR_PATH, SELECTOR_META_PATH, BASELINE_VAL_GT, weights,
    ]
    missing = [p for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("缺少Stage 6.4B输入：\n" + "\n".join(map(str, missing)))

    filter_summary = load_json(FILTER_SUMMARY)
    audit = load_json(AUDIT_JSON)
    selector_meta = load_json(SELECTOR_META_PATH)
    expected_manifest_sha = filter_summary.get("output_sha256", {}).get("independent_test_manifest.csv")
    if expected_manifest_sha and sha256_file(manifest_path) != expected_manifest_sha:
        raise RuntimeError("independent_test_manifest.csv与Stage 6.4A2摘要SHA256不一致")

    if filter_summary.get("status") != "READY_FOR_STAGE6_4B":
        raise RuntimeError(
            "Stage 6.4A2未通过，禁止运行最终测试："
            + str(filter_summary.get("status"))
        )
    if selector_meta.get("status") != "completed":
        raise RuntimeError("Stage 6.3F-C最终选择器元数据状态不是completed")

    m3_meta = selector_meta["models"]["M3_Logistic_F0"]
    if m3_meta.get("feature_names") != FEATURE_NAMES:
        raise RuntimeError("M3特征列表与冻结协议不一致")
    if m3_meta.get("final_config") != {"C": 0.01, "class_weight": "balanced"}:
        raise RuntimeError("M3 LogisticRegression配置与冻结协议不一致")
    if not math.isclose(
        float(m3_meta.get("probability_threshold", -1)),
        SELECTOR_THRESHOLD,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("M3概率阈值与冻结协议不一致")

    weight_hash = sha256_file(weights)
    selector_hash = sha256_file(SELECTOR_PATH)
    if weight_hash != EXPECTED_WEIGHT_SHA256:
        raise RuntimeError(
            f"YOLO权重SHA256不匹配：{weight_hash}，"
            f"预期{EXPECTED_WEIGHT_SHA256}"
        )
    if selector_hash != EXPECTED_SELECTOR_SHA256:
        raise RuntimeError(
            f"M3选择器SHA256不匹配：{selector_hash}，"
            f"预期{EXPECTED_SELECTOR_SHA256}"
        )
    if m3_meta.get("model_sha256") != selector_hash:
        raise RuntimeError("joblib模型SHA256与最终选择器元数据不一致")

    model = joblib.load(SELECTOR_PATH)

    rows_raw = read_csv(manifest_path)
    if len(rows_raw) != EXPECTED_TOTAL:
        raise RuntimeError(
            f"独立测试manifest应有{EXPECTED_TOTAL}行，实际{len(rows_raw)}"
        )

    rows: list[dict[str, Any]] = []
    image_ids = set()
    group_ids = set()
    source_counts = Counter()

    for raw in rows_raw:
        image_id = int(raw["image_id"])
        if image_id in image_ids:
            raise RuntimeError(f"独立测试manifest存在重复image_id={image_id}")
        image_ids.add(image_id)

        source = raw["source"].strip().lower()
        if source not in EXPECTED_SOURCE_COUNTS:
            raise RuntimeError(f"未知数据来源：{source}")

        sequence_id = raw["sequence_id"].strip()
        group_id = raw["group_id"].strip()
        if group_id != f"{source}:{sequence_id}":
            raise RuntimeError(f"image_id={image_id} group_id不一致")
        if sequence_id != sequence_id_from_name(raw["file_name"]):
            raise RuntimeError(f"image_id={image_id} sequence_id解析不一致")

        relative = Path(raw["relative_path"])
        image_path = dataset_root / "images" / "test" / relative
        label_path = dataset_root / "labels" / "test" / relative.with_suffix(".txt")
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        if not label_path.is_file():
            raise FileNotFoundError(label_path)
        if raw["label_errors"].strip() or raw["label_exists"] != "1":
            raise RuntimeError(f"标签审计未通过：image_id={image_id}")

        width, height = int(raw["width"]), int(raw["height"])
        if width <= 0 or height <= 0:
            raise RuntimeError(f"图片尺寸非法：image_id={image_id}")

        rows.append({
            "image_id": image_id,
            "file_name": raw["file_name"],
            "relative_path": str(relative).replace("\\", "/"),
            "source": source,
            "sequence_id": sequence_id,
            "group_id": group_id,
            "width": width,
            "height": height,
            "image_path": image_path,
            "label_path": label_path,
        })
        group_ids.add(group_id)
        source_counts[source] += 1

    if dict(source_counts) != EXPECTED_SOURCE_COUNTS:
        raise RuntimeError(f"独立测试来源数量异常：{dict(source_counts)}")
    if len(group_ids) != EXPECTED_GROUPS:
        raise RuntimeError(f"独立测试序列组数异常：{len(group_ids)}")

    excluded_groups = set(audit["sequence_overlap"]["train_vs_test"]["overlap_preview"])
    excluded_groups |= set(audit["sequence_overlap"]["val_vs_test"]["overlap_preview"])
    if group_ids & excluded_groups:
        raise RuntimeError("独立测试manifest仍包含与train/val交叉的序列")

    vis_groups = {r["group_id"] for r in rows if r["source"] == "visdrone"}
    uav_groups = {r["group_id"] for r in rows if r["source"] == "uavdt"}
    if len(vis_groups) != 84 or len(uav_groups) != 5:
        raise RuntimeError(
            f"独立测试序列分布异常：VisDrone={len(vis_groups)}, UAVDT={len(uav_groups)}"
        )

    hashes = {
        "independent_test_manifest": sha256_file(manifest_path),
        "sequence_filter_summary": sha256_file(FILTER_SUMMARY),
        "dataset_audit": sha256_file(AUDIT_JSON),
        "yolo_weights": weight_hash,
        "m3_selector": selector_hash,
        "selector_metadata": sha256_file(SELECTOR_META_PATH),
        "baseline_coco_category_schema": sha256_file(BASELINE_VAL_GT),
    }
    return rows, filter_summary, selector_meta, model, hashes


def parse_yolo_labels(label_path: Path, image_id: int, width: int, height: int,
                      ann_start_id: int) -> tuple[list[dict[str, Any]], int]:
    annotations = []
    ann_id = ann_start_id
    for line_no, line in enumerate(label_path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise RuntimeError(f"{label_path}:{line_no}不是5列YOLO检测标签")
        cls_float = float(parts[0])
        cls = int(cls_float)
        if cls_float != cls or cls != 0:
            raise RuntimeError(
                f"{label_path}:{line_no}类别应为0，实际为{parts[0]}"
            )
        cx, cy, bw, bh = map(float, parts[1:])
        if not all(math.isfinite(v) for v in (cx, cy, bw, bh)):
            raise RuntimeError(f"{label_path}:{line_no}包含非有限坐标")
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
            raise RuntimeError(f"{label_path}:{line_no}归一化坐标超出范围")

        box_w = bw * width
        box_h = bh * height
        x = (cx - bw / 2.0) * width
        y = (cy - bh / 2.0) * height
        # Preserve the YOLO annotation geometry exactly. Do not clip boxes:
        # the source labels may legitimately touch/cross image boundaries.
        if box_w <= 0 or box_h <= 0:
            raise RuntimeError(f"{label_path}:{line_no}目标框宽高非正")

        annotations.append({
            "id": ann_id,
            "image_id": int(image_id),
            "category_id": CATEGORY_ID,
            "bbox": [float(x), float(y), float(box_w), float(box_h)],
            "area": float(box_w * box_h),
            "iscrowd": 0,
        })
        ann_id += 1
    return annotations, ann_id


def build_coco_gt(rows: list[dict[str, Any]], out_path: Path) -> dict[str, Any]:
    images = []
    annotations = []
    next_ann_id = 1
    for row in rows:
        images.append({
            "id": row["image_id"],
            "file_name": row["file_name"],
            "width": row["width"],
            "height": row["height"],
        })
        new_anns, next_ann_id = parse_yolo_labels(
            row["label_path"], row["image_id"], row["width"], row["height"], next_ann_id
        )
        annotations.extend(new_anns)

    coco = {
        "info": {
            "description": "Stage 6.4B sequence-independent HighwayVehicle test set",
            "version": "1.0",
            "year": 2026,
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": CATEGORY_ID, "name": CATEGORY_NAME, "supercategory": CATEGORY_NAME}],
    }
    write_json_atomic(out_path, coco)
    return {
        "images": len(images),
        "annotations": len(annotations),
        "categories": coco["categories"],
        "gt_sha256": sha256_file(out_path),
    }


def empty_result():
    return (
        np.empty((0, 4), dtype=np.float32),
        np.empty((0,), dtype=np.float32),
        np.empty((0,), dtype=np.int64),
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


def predict_original(model, image, pred_args, torch, batched_nms):
    results = model.predict(source=image, batch=1, **pred_args)
    boxes, scores, classes = extract_result(results[0])
    height, width = image.shape[:2]
    return merge_nms(boxes, scores, classes, width, height, torch, batched_nms)


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


def predict_tiled(model, image, pred_args, torch, batched_nms):
    tiles, offsets = make_tiles(image)
    results = model.predict(source=tiles, batch=len(tiles), **pred_args)
    if len(results) != len(tiles) or len(tiles) != 4:
        raise RuntimeError(
            f"切片数与结果数异常：tiles={len(tiles)}, results={len(results)}"
        )

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
        width, height, torch, batched_nms,
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


def compute_scene_features(result, width: int, height: int) -> dict[str, Any]:
    boxes, scores, _classes = result
    keep = np.asarray(scores, dtype=np.float32) >= FEATURE_CONF
    boxes = np.asarray(boxes, dtype=np.float32)[keep]
    n_det = int(len(boxes))

    if n_det == 0:
        return {
            "n_det_conf025": 0,
            "median_box_area_640": None,
            "tiny_ratio": 0.0,
        }

    scale = min(FEATURE_IMGSZ / float(width), FEATURE_IMGSZ / float(height))
    box_widths = boxes[:, 2] - boxes[:, 0]
    box_heights = boxes[:, 3] - boxes[:, 1]
    if np.any(box_widths <= 0) or np.any(box_heights <= 0):
        raise RuntimeError("原图预测框存在非正宽高")
    areas = (box_widths * scale) * (box_heights * scale)
    if not np.all(np.isfinite(areas)) or np.any(areas <= 0):
        raise RuntimeError("场景特征面积出现非法值")
    tiny_count = int(np.sum(areas < TINY_AREA_THRESHOLD))
    return {
        "n_det_conf025": n_det,
        "median_box_area_640": round(float(np.median(areas)), 4),
        "tiny_ratio": round(float(tiny_count / n_det), 6),
    }


def selector_probability(selector, feature_row: dict[str, Any]) -> float:
    X = np.asarray([[
        float(feature_row["n_det_conf025"]),
        np.nan if feature_row["median_box_area_640"] is None
        else float(feature_row["median_box_area_640"]),
        float(feature_row["tiny_ratio"]),
    ]], dtype=np.float64)
    probabilities = selector.predict_proba(X)
    classes = list(selector.named_steps["classifier"].classes_)
    if 1 not in classes:
        raise RuntimeError(f"M3选择器缺少beneficial正类1：{classes}")
    return float(probabilities[0, classes.index(1)])


def prediction_to_coco(image_id: int, result) -> list[dict[str, Any]]:
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
            "category_id": CATEGORY_ID,
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "score": float(score),
        })
    return output


def prediction_digest(predictions: list[dict[str, Any]]) -> str:
    normalized = []
    for p in predictions:
        normalized.append((
            int(p["image_id"]),
            int(p["category_id"]),
            round(float(p["score"]), 5),
            tuple(round(float(v), 4) for v in p["bbox"]),
        ))
    normalized.sort()
    payload = json.dumps(normalized, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def synchronize(torch, device: str) -> None:
    if torch.cuda.is_available() and str(device).lower() not in {"cpu", "-1"}:
        torch.cuda.synchronize()


def run_repeat(
    repeat: int,
    rows: list[dict[str, Any]],
    model,
    selector,
    cv2,
    torch,
    batched_nms,
    device: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, list[dict[str, Any]]],
           list[dict[str, Any]]]:
    pred_args = prediction_args(device)
    predictions = {"A_Original": [], "B_C2_area2048": [], "C_M3_selective": []}
    per_image = []
    decisions = []
    read_t, original_t = {}, {}
    feature_t, selector_t, tile_t, fusion_t = {}, {}, {}, {}
    selected_ids = set()

    start_wall = time.perf_counter()

    for idx, row in enumerate(rows, 1):
        image_id = row["image_id"]
        t0 = time.perf_counter()
        image = cv2.imread(str(row["image_path"]))
        read_t[image_id] = time.perf_counter() - t0
        if image is None:
            raise RuntimeError(f"图片读取失败：{row['image_path']}")

        height, width = image.shape[:2]
        if (width, height) != (row["width"], row["height"]):
            raise RuntimeError(
                f"图片尺寸不一致：{row['file_name']} "
                f"actual={width}x{height}, expected={row['width']}x{row['height']}"
            )

        synchronize(torch, device)
        t0 = time.perf_counter()
        original = predict_original(model, image, pred_args, torch, batched_nms)
        synchronize(torch, device)
        original_t[image_id] = time.perf_counter() - t0

        original_coco = prediction_to_coco(image_id, original)
        predictions["A_Original"].extend(original_coco)

        f_sec = s_sec = t_sec = fu_sec = 0.0
        selector_score = None
        selected = False

        if row["source"] == "visdrone":
            t0 = time.perf_counter()
            features = compute_scene_features(original, width, height)
            f_sec = time.perf_counter() - t0
            feature_t[image_id] = f_sec

            t0 = time.perf_counter()
            selector_score = selector_probability(selector, features)
            s_sec = time.perf_counter() - t0
            selector_t[image_id] = s_sec
            selected = selector_score >= SELECTOR_THRESHOLD
            if selected:
                selected_ids.add(image_id)

            synchronize(torch, device)
            t0 = time.perf_counter()
            tiled, tile_count = predict_tiled(
                model, image, pred_args, torch, batched_nms
            )
            synchronize(torch, device)
            t_sec = time.perf_counter() - t0
            if tile_count != 4:
                raise RuntimeError(f"image_id={image_id}切片数不是4")
            tile_t[image_id] = t_sec

            t0 = time.perf_counter()
            c2_result = fuse_c2_area2048(
                original, tiled, width, height, torch, batched_nms
            )
            fu_sec = time.perf_counter() - t0
            fusion_t[image_id] = fu_sec

            c2_coco = prediction_to_coco(image_id, c2_result)
            predictions["B_C2_area2048"].extend(c2_coco)
            predictions["C_M3_selective"].extend(
                c2_coco if selected else original_coco
            )

            decisions.append({
                "image_id": image_id,
                "file_name": row["file_name"],
                "sequence_id": row["sequence_id"],
                "selector_score": selector_score,
                "threshold": SELECTOR_THRESHOLD,
                "selected_for_tiling": int(selected),
                "n_det_conf025": features["n_det_conf025"],
                "median_box_area_640": (
                    "" if features["median_box_area_640"] is None
                    else features["median_box_area_640"]
                ),
                "tiny_ratio": features["tiny_ratio"],
            })
        else:
            # The fixed source-specific protocol is UAVDT original-only.
            predictions["B_C2_area2048"].extend(original_coco)
            predictions["C_M3_selective"].extend(original_coco)

        base_sec = read_t[image_id] + original_t[image_id]
        a_ms = base_sec * 1000.0
        b_ms = (
            base_sec + (t_sec + fu_sec if row["source"] == "visdrone" else 0.0)
        ) * 1000.0
        c_ms = (
            base_sec
            + (f_sec + s_sec if row["source"] == "visdrone" else 0.0)
            + (t_sec + fu_sec if selected else 0.0)
        ) * 1000.0

        per_image.append({
            "repeat": repeat,
            "image_id": image_id,
            "source": row["source"],
            "sequence_id": row["sequence_id"],
            "selected_for_tiling": int(selected),
            "selector_score": "" if selector_score is None else selector_score,
            "read_ms": read_t[image_id] * 1000.0,
            "original_ms": original_t[image_id] * 1000.0,
            "feature_ms": f_sec * 1000.0,
            "selector_ms": s_sec * 1000.0,
            "tile_ms": t_sec * 1000.0,
            "fusion_ms": fu_sec * 1000.0,
            "A_Original_image_ms": a_ms,
            "B_C2_area2048_image_ms": b_ms,
            "C_M3_selective_image_ms": c_ms,
        })

        if idx % 50 == 0 or idx == len(rows):
            print(
                f"[repeat {repeat}] {idx}/{len(rows)} | "
                f"VisDrone selected={len(selected_ids)}",
                flush=True,
            )

    total_read = float(sum(read_t.values()))
    total_original = float(sum(original_t.values()))
    total_feature = float(sum(feature_t.values()))
    total_selector = float(sum(selector_t.values()))
    total_tile_all_vis = float(sum(tile_t.values()))
    total_fusion_all_vis = float(sum(fusion_t.values()))
    total_tile_selected = float(sum(tile_t[i] for i in selected_ids))
    total_fusion_selected = float(sum(fusion_t[i] for i in selected_ids))

    a_total = total_read + total_original
    b_total = a_total + total_tile_all_vis + total_fusion_all_vis
    c_total = (
        a_total + total_feature + total_selector
        + total_tile_selected + total_fusion_selected
    )

    lat_a = np.asarray([x["A_Original_image_ms"] for x in per_image], dtype=np.float64)
    lat_b = np.asarray([x["B_C2_area2048_image_ms"] for x in per_image], dtype=np.float64)
    lat_c = np.asarray([x["C_M3_selective_image_ms"] for x in per_image], dtype=np.float64)

    def policy_metrics(total: float, lat: np.ndarray) -> dict[str, float]:
        return {
            "total_seconds": float(total),
            "images_per_second": float(len(rows) / total) if total > 0 else 0.0,
            "mean_image_ms": float(np.mean(lat)),
            "p95_image_ms": float(np.percentile(lat, 95)),
        }

    result = {
        "repeat": repeat,
        "status": "completed",
        "wall_seconds": float(time.perf_counter() - start_wall),
        "images": len(rows),
        "visdrone_images": sum(r["source"] == "visdrone" for r in rows),
        "selected_visdrone_images": len(selected_ids),
        "trigger_rate_visdrone": (
            len(selected_ids) / sum(r["source"] == "visdrone" for r in rows)
        ),
        "tile_crops_selected": len(selected_ids) * 4,
        "component_seconds": {
            "read": total_read,
            "original": total_original,
            "feature": total_feature,
            "selector": total_selector,
            "tile_all_visdrone": total_tile_all_vis,
            "fusion_all_visdrone": total_fusion_all_vis,
            "tile_selected_m3": total_tile_selected,
            "fusion_selected_m3": total_fusion_selected,
        },
        "policies": {
            "A_Original": policy_metrics(a_total, lat_a),
            "B_C2_area2048": policy_metrics(b_total, lat_b),
            "C_M3_selective": policy_metrics(c_total, lat_c),
        },
        "prediction_sha256": {
            name: prediction_digest(preds) for name, preds in predictions.items()
        },
    }
    return result, per_image, predictions, decisions


def evaluate_predictions(gt_path: Path, prediction_paths: dict[str, Path],
                         rows: list[dict[str, Any]]) -> dict[str, Any]:
    coco_gt = COCO(str(gt_path))
    category_ids = sorted(int(x) for x in coco_gt.getCatIds())
    if category_ids != [CATEGORY_ID]:
        raise RuntimeError(f"COCO类别ID应为[{CATEGORY_ID}]，实际{category_ids}")

    scopes = {
        "full": sorted(int(r["image_id"]) for r in rows),
        "uavdt": sorted(int(r["image_id"]) for r in rows if r["source"] == "uavdt"),
        "visdrone": sorted(int(r["image_id"]) for r in rows if r["source"] == "visdrone"),
    }

    results: dict[str, Any] = {}
    for policy, pred_path in prediction_paths.items():
        print("\n" + "=" * 72)
        print("COCO evaluation:", policy)
        coco_dt = coco_gt.loadRes(str(pred_path))
        results[policy] = {}

        for scope_name, image_ids in scopes.items():
            print(f"\n[{policy}] scope={scope_name}, images={len(image_ids)}")
            evaluator = COCOeval_faster(
                coco_gt, coco_dt, iouType="bbox", lvis_style=False, print_function=print
            )
            evaluator.params.imgIds = image_ids
            evaluator.params.catIds = category_ids
            evaluator.params.iouThrs = EVAL_IOU_THRESHOLDS.copy()
            evaluator.params.recThrs = EVAL_RECALL_THRESHOLDS.copy()
            evaluator.params.maxDets = EVAL_MAX_DETS.copy()
            evaluator.params.areaRng = [x.copy() for x in EVAL_AREA_RANGES]
            evaluator.params.areaRngLbl = EVAL_AREA_LABELS.copy()
            evaluator.params.useCats = 1
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
            stats = evaluator.stats_as_dict

            required = ["AP_all", "AP_50", "AP_small", "AP_medium", "AP_large"]
            for key in required:
                if key not in stats:
                    raise RuntimeError(f"评价器缺少指标：{key}")

            results[policy][scope_name] = {
                "images": len(image_ids),
                "AP50-95": float(stats["AP_all"]),
                "AP50": float(stats["AP_50"]),
                "APs": float(stats["AP_small"]),
                "APm": float(stats["AP_medium"]),
                "APl": float(stats["AP_large"]),
            }

    return results


def build_deltas(results: dict[str, Any], baseline_policy: str = "A_Original") -> dict[str, Any]:
    out = {}
    base = results[baseline_policy]
    for policy in results:
        out[policy] = {}
        for scope in base:
            out[policy][scope] = {}
            for metric in ("AP50-95", "AP50", "APs", "APm", "APl"):
                out[policy][scope][metric] = round(
                    float(results[policy][scope][metric] - base[scope][metric]), 8
                )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 6.4B frozen M3 independent test"
    )
    parser.add_argument("--dataset-root", type=str, default=str(DATASET_ROOT_DEFAULT))
    parser.add_argument("--manifest", type=str, default=str(MANIFEST_DEFAULT))
    parser.add_argument("--weights", type=str, default=str(WEIGHT_DEFAULT))
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.resume and args.overwrite:
        parser.error("--resume与--overwrite不能同时使用")
    if args.repeats < 1 or args.warmup < 0:
        parser.error("repeats必须>=1，warmup必须>=0")

    env = check_environment()
    dataset_root = Path(args.dataset_root).expanduser()
    manifest_path = Path(args.manifest).expanduser()
    weights = Path(args.weights).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = ROOT / dataset_root
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    if not weights.is_absolute():
        weights = ROOT / weights
    out_dir = OUT_DIR_DEFAULT

    if out_dir.exists() and any(out_dir.iterdir()):
        if args.overwrite:
            shutil.rmtree(out_dir)
        elif not args.resume:
            raise FileExistsError(
                f"{out_dir}已有内容；续跑使用--resume，确认重做使用--overwrite"
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    repeat_dir = out_dir / "repeats"
    repeat_dir.mkdir(parents=True, exist_ok=True)

    rows, filter_summary, selector_meta, selector, input_hashes = validate_inputs(
        dataset_root, manifest_path, weights
    )

    # Reuse the exact single-class category ID/name from the Stage 5 COCO GT.
    global CATEGORY_ID, CATEGORY_NAME
    category_schema = load_json(BASELINE_VAL_GT)
    categories = category_schema.get("categories", [])
    if len(categories) != 1:
        raise RuntimeError(f"Stage 5 COCO类别数应为1，实际{len(categories)}")
    CATEGORY_ID = int(categories[0]["id"])
    CATEGORY_NAME = str(categories[0]["name"])

    gt_path = out_dir / "test_gt_coco.json"
    gt_info = build_coco_gt(rows, gt_path)

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
    print("Stage 6.4B | Frozen M3 independent test")
    print("Test images:", len(rows))
    print("Source counts:", dict(Counter(r["source"] for r in rows)))
    print("Sequence groups:", len({r["group_id"] for r in rows}))
    print("Selector evaluation images/sequences: 84 VisDrone")
    print("YOLO weights:", weights)
    print("YOLO SHA256:", input_hashes["yolo_weights"])
    print("M3 selector SHA256:", input_hashes["m3_selector"])
    print("M3 threshold:", SELECTOR_THRESHOLD)
    print("Protocol: imgsz=640, conf=0.001, NMS IoU=0.7, overlap=0.15")
    print("UAVDT=original-only; VisDrone=M3-gated C2_area2048")
    print("No YOLO training; no selector refitting; no test threshold tuning.")
    print("Environment:", env)
    print("GT:", gt_info)
    print("=" * 88, flush=True)

    model = YOLO(str(weights))
    if len(model.names) != 1:
        raise RuntimeError(f"模型类别数应为1，实际{len(model.names)}")

    # Warm-up is excluded from timing.
    first_image = cv2.imread(str(rows[0]["image_path"]))
    first_vis = next(r for r in rows if r["source"] == "visdrone")
    first_vis_image = cv2.imread(str(first_vis["image_path"]))
    if first_image is None or first_vis_image is None:
        raise RuntimeError("Warm-up图片读取失败")
    warm_tiles, _ = make_tiles(first_vis_image)
    pred_args = prediction_args(args.device)
    for _ in range(args.warmup):
        synchronize(torch, args.device)
        model.predict(source=first_image, batch=1, **pred_args)
        synchronize(torch, args.device)
        model.predict(source=warm_tiles, batch=len(warm_tiles), **pred_args)
        synchronize(torch, args.device)
    del first_image, first_vis_image, warm_tiles
    gc.collect()

    repeat_summaries = []
    all_per_image = []
    baseline_predictions = None
    baseline_digests = None
    baseline_decisions = None

    for repeat in range(1, args.repeats + 1):
        summary_path = repeat_dir / f"repeat_{repeat}.json"
        per_image_path = repeat_dir / f"repeat_{repeat}_per_image.csv"
        decisions_path = repeat_dir / f"repeat_{repeat}_selector_decisions.csv"
        pred_paths = {
            "A_Original": repeat_dir / f"repeat_{repeat}_predictions_A.json",
            "B_C2_area2048": repeat_dir / f"repeat_{repeat}_predictions_B.json",
            "C_M3_selective": repeat_dir / f"repeat_{repeat}_predictions_C.json",
        }

        complete = (
            summary_path.is_file() and per_image_path.is_file() and decisions_path.is_file()
            and all(p.is_file() for p in pred_paths.values())
        )
        if args.resume and complete:
            cached = load_json(summary_path)
            if cached.get("input_sha256") != input_hashes:
                raise RuntimeError(f"repeat {repeat}输入SHA256不一致")
            per_rows = read_csv(per_image_path)
            decisions = read_csv(decisions_path)
            preds = {k: load_json(v) for k, v in pred_paths.items()}
            if repeat == 1:
                baseline_predictions = preds
                baseline_digests = cached["result"]["prediction_sha256"]
                baseline_decisions = decisions
            else:
                for name in preds:
                    if prediction_digest(preds[name]) != baseline_digests[name]:
                        raise RuntimeError(f"resume时repeat {repeat}预测与repeat 1不一致：{name}")
            repeat_summaries.append(cached["result"])
            all_per_image.extend(per_rows)
            print(f"[RESUME] repeat={repeat} checkpoint loaded", flush=True)
            continue

        print(f"\n[REPEAT {repeat}/{args.repeats}]", flush=True)
        result, per_rows, preds, decisions = run_repeat(
            repeat, rows, model, selector, cv2, torch, batched_nms, args.device
        )

        if repeat == 1:
            baseline_predictions = preds
            baseline_digests = result["prediction_sha256"]
            baseline_decisions = decisions
        else:
            for name in preds:
                if prediction_digest(preds[name]) != baseline_digests[name]:
                    raise RuntimeError(
                        f"重复运行预测不一致：repeat={repeat}, policy={name}。"
                        "停止最终评价，先检查推理确定性。"
                    )

        for name, path in pred_paths.items():
            write_json_atomic(path, preds[name])
        write_csv_atomic(per_image_path, per_rows)
        write_csv_atomic(decisions_path, decisions)
        write_json_atomic(summary_path, {
            "status": "completed",
            "input_sha256": input_hashes,
            "result": result,
        })

        repeat_summaries.append(result)
        all_per_image.extend(per_rows)

    if baseline_predictions is None:
        raise RuntimeError("没有可用于COCO评价的repeat 1预测")

    prediction_paths = {
        "A_Original": repeat_dir / "repeat_1_predictions_A.json",
        "B_C2_area2048": repeat_dir / "repeat_1_predictions_B.json",
        "C_M3_selective": repeat_dir / "repeat_1_predictions_C.json",
    }
    # Publish stable top-level copies for downstream analysis.
    final_pred_paths = {
        "A_Original": out_dir / "test_predictions_A_original.json",
        "B_C2_area2048": out_dir / "test_predictions_B_C2_area2048.json",
        "C_M3_selective": out_dir / "test_predictions_C_M3_selective.json",
    }
    for name, source in prediction_paths.items():
        shutil.copy2(source, final_pred_paths[name])

    if baseline_decisions:
        write_csv_atomic(out_dir / "selector_decisions.csv", baseline_decisions)
    else:
        raise RuntimeError("缺少repeat 1的selector_decisions检查点")
    write_csv_atomic(out_dir / "runtime_per_image.csv", all_per_image)

    repeat_metric_rows = []
    for rep in repeat_summaries:
        for policy, values in rep["policies"].items():
            repeat_metric_rows.append({
                "repeat": rep["repeat"],
                "policy": policy,
                "images": rep["images"],
                "selected_visdrone_images": rep["selected_visdrone_images"],
                "trigger_rate_visdrone": rep["trigger_rate_visdrone"],
                "tile_crops_selected": rep["tile_crops_selected"],
                **values,
            })
    write_csv_atomic(out_dir / "runtime_repeat_metrics.csv", repeat_metric_rows)

    # COCO metrics use only repeat 1 predictions. Repeated runs must have
    # identical prediction digests, so accuracy is not averaged across repeats.
    results = evaluate_predictions(gt_path, final_pred_paths, rows)
    deltas = build_deltas(results, "A_Original")
    deltas_vs_c2 = build_deltas(results, "B_C2_area2048")

    runtime_summary = {}
    for policy in ("A_Original", "B_C2_area2048", "C_M3_selective"):
        totals = np.asarray(
            [r["policies"][policy]["total_seconds"] for r in repeat_summaries],
            dtype=np.float64,
        )
        fps = np.asarray(
            [r["policies"][policy]["images_per_second"] for r in repeat_summaries],
            dtype=np.float64,
        )
        p95 = np.asarray(
            [r["policies"][policy]["p95_image_ms"] for r in repeat_summaries],
            dtype=np.float64,
        )
        runtime_summary[policy] = {
            "total_seconds_mean": float(np.mean(totals)),
            "total_seconds_std": float(np.std(totals, ddof=1)) if len(totals) > 1 else 0.0,
            "images_per_second_mean": float(np.mean(fps)),
            "p95_image_ms_mean": float(np.mean(p95)),
        }

    selected_counts = [int(r["selected_visdrone_images"]) for r in repeat_summaries]
    if len(set(selected_counts)) != 1:
        raise RuntimeError(f"重复运行的M3触发数量不一致：{selected_counts}")

    summary = {
        "experiment": "Stage6.4B_Frozen_M3_Independent_Test",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "completed",
        "protocol": {
            "imgsz": IMGSZ,
            "conf": CONF,
            "nms_iou": NMS_IOU,
            "max_det": MAX_DET,
            "overlap": OVERLAP,
            "tile_layout": "2x2",
            "tile_area_filter": "tile box area < 2048 original-image pixels squared",
            "half": HALF,
            "augment": AUGMENT,
            "feature_conf": FEATURE_CONF,
            "feature_imgsz": FEATURE_IMGSZ,
            "tiny_area_threshold": TINY_AREA_THRESHOLD,
            "selector": "frozen M3 LogisticRegression F0",
            "selector_features": FEATURE_NAMES,
            "selector_config": {"C": 0.01, "class_weight": "balanced"},
            "selector_threshold": SELECTOR_THRESHOLD,
            "coco_category_id": CATEGORY_ID,
            "coco_category_name": CATEGORY_NAME,
            "source_policy": "UAVDT original-only; VisDrone M3-gated C2_area2048",
            "repeats": len(repeat_summaries),
            "accuracy_evaluation_repeat": 1,
            "evaluator": "faster_coco_eval",
            "evaluator_version": version("faster-coco-eval"),
            "coco_iou_thresholds": EVAL_IOU_THRESHOLDS.tolist(),
            "coco_recall_thresholds": EVAL_RECALL_THRESHOLDS.tolist(),
            "coco_max_dets": EVAL_MAX_DETS,
            "coco_area_ranges": EVAL_AREA_RANGES,
        },
        "test_set": {
            "images": len(rows),
            "source_counts": dict(Counter(r["source"] for r in rows)),
            "sequence_groups": len({r["group_id"] for r in rows}),
            "uavdt_sequences": len({r["group_id"] for r in rows if r["source"] == "uavdt"}),
            "visdrone_sequences": len({r["group_id"] for r in rows if r["source"] == "visdrone"}),
            "visdrone_selector_test_images": sum(r["source"] == "visdrone" for r in rows),
            "ground_truth_annotations": gt_info["annotations"],
        },
        "selector": {
            "selected_visdrone_images": selected_counts[0],
            "trigger_rate_visdrone": selected_counts[0] / 84.0,
            "tile_crops_selected": selected_counts[0] * 4,
            "decision_file": "selector_decisions.csv",
        },
        "runtime": runtime_summary,
        "runtime_repeats": repeat_summaries,
        "metrics": results,
        "delta_vs_original": deltas,
        "delta_vs_c2_area2048": deltas_vs_c2,
        "input_sha256": input_hashes,
        "ground_truth_sha256": gt_info["gt_sha256"],
        "prediction_sha256": {
            name: sha256_file(path) for name, path in final_pred_paths.items()
        },
        "interpretation_limit": (
            "The independent test has 1,288 UAVDT images from 5 sequences and "
            "84 VisDrone images from 84 sequences. The selector is only applied "
            "to VisDrone; therefore the independent sample for evaluating M3 "
            "selection is 84 images/sequences. Report source-specific VisDrone "
            "metrics and this sample size explicitly. Do not claim that all "
            "1,372 images independently validate selector decisions."
        ),
    }
    write_json_atomic(out_dir / "metrics_summary.json", summary)

    print("\n" + "=" * 88)
    print("Stage 6.4B completed")
    print("Test:", summary["test_set"])
    print("M3 selected:", summary["selector"])
    print("Runtime:", runtime_summary)
    print("Metrics:", json.dumps(results, ensure_ascii=False, indent=2))
    print("Delta vs Original:", json.dumps(deltas, ensure_ascii=False, indent=2))
    print("Delta vs C2_area2048:", json.dumps(deltas_vs_c2, ensure_ascii=False, indent=2))
    print("Output:", out_dir)
    print("=" * 88)


if __name__ == "__main__":
    main()
