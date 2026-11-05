#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 6.3D-B corrected live-selector runtime benchmark.

Stage 6.3D-A bootstrap is already completed. This replacement is for the
runtime benchmark only. It computes scene features and fold-specific selector
rules online; selector_oof_assignments.csv is audit-only, never an inference
input. Run from the YOLO_Thesis repository root.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Fixed paths and protocol (must match Stage 6.3A/6.3C/Stage 6.1)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
SCENE_DIR = ROOT / "runs/detect/stage6_scene_selector"
ORACLE_DIR = ROOT / "runs/detect/stage6_selective_oracle"
C2_DIR = ROOT / "runs/detect/stage6_size_aware_fusion_iou07"
FEATURE_CSV = SCENE_DIR / "corrected_scene_features.csv"
ASSIGNMENTS_CSV = SCENE_DIR / "selector_oof_assignments.csv"
FOLD_RULES_JSON = SCENE_DIR / "selector_fold_rules.json"
OOF_SUMMARY_JSON = SCENE_DIR / "selector_oof_summary.json"
GT_JSON = ROOT / "runs/detect/source_validation/Stage5-Baseline-Unified/annotations_auto_detect_val.json"
A_PRED_JSON = ORACLE_DIR / "all_original_predictions.json"
C2_PRED_JSON = C2_DIR / "C2_area2048_predictions.json"
REFERENCE_JSON = C2_DIR / "faster_eval_metrics.json"
WEIGHT_DEFAULT = ROOT / "runs/detect/stage2_screening/YOLO11s-DySample/weights/best.pt"
IMAGE_DIR_DEFAULT = ROOT / "datasets/HighwayVehicle-v1.0/images/val"
OUT_SUMMARY = SCENE_DIR / "stage6_3d_runtime_summary.json"
OUT_PER_IMAGE = SCENE_DIR / "stage6_3d_runtime_per_image.csv"

EXPECTED_SOURCE_COUNTS = {"uavdt": 1320, "visdrone": 247}
EXPECTED_TOTAL = 1567
EXPECTED_VIS_GROUPS = 94
EXPECTED_SELECTED_COUNTS = {"Loose": 154, "Medium": 120}

IMGSZ = 640
CONF = 0.001
NMS_IOU = 0.7
MAX_DET = 300
OVERLAP = 0.15
HALF = False
AUGMENT = False
EXPECTED_EVALUATOR_VERSION = "1.8.0"
FEATURE_CONF = 0.25
FEATURE_IMGSZ = 640
TINY_AREA_THRESHOLD = 32 ** 2
AREA_THRESHOLD = 2048.0


# ---------------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------------
def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require_files(paths: list[Path]) -> None:
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("缺少必要文件：\n" + "\n".join(missing))


def normalize_name(value: str) -> str:
    return str(value).replace("\\", "/").lstrip("./")


def sequence_id_from_name(filename: str) -> str:
    stem = Path(str(filename)).stem
    parts = stem.rsplit("_", 1)
    return parts[0] if len(parts) == 2 else stem


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"拒绝写入空CSV：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def check_evaluator_version() -> str:
    try:
        current = version("faster-coco-eval")
    except PackageNotFoundError as exc:
        raise RuntimeError("未安装faster-coco-eval 1.8.0。") from exc
    if current != EXPECTED_EVALUATOR_VERSION:
        raise RuntimeError(f"faster-coco-eval版本不匹配：{current}")
    return current


# ---------------------------------------------------------------------------
# Load and validate Stage 6 inputs
# ---------------------------------------------------------------------------
def read_feature_rows() -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with FEATURE_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"image_id", "file_name", "source", "sequence_id", "width", "height",
                    "n_det_conf025", "median_box_area_640", "tiny_ratio"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"corrected_scene_features.csv缺少字段：{sorted(missing)}")
        for raw in reader:
            image_id = int(raw["image_id"])
            if image_id in rows:
                raise RuntimeError(f"特征CSV重复image_id={image_id}")
            median = raw["median_box_area_640"].strip()
            rows[image_id] = {
                **raw,
                "image_id": image_id,
                "source": raw["source"].strip().lower(),
                "sequence_id": raw["sequence_id"].strip(),
                "width": int(raw["width"]),
                "height": int(raw["height"]),
                "n_det_conf025": int(raw["n_det_conf025"]),
                "median_box_area_640": None if median == "" else float(median),
                "tiny_ratio": float(raw["tiny_ratio"]),
            }
    return rows


def read_assignments(vis_ids: set[int]) -> dict[str, dict[int, dict[str, Any]]]:
    result: dict[str, dict[int, dict[str, Any]]] = {"Loose": {}, "Medium": {}}
    with ASSIGNMENTS_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            policy = raw["policy"].strip()
            if policy not in result:
                continue
            image_id = int(raw["image_id"])
            if image_id in result[policy]:
                raise RuntimeError(f"{policy} assignment重复image_id={image_id}")
            flag = raw["selected_for_c2_2048"].strip()
            if flag not in {"0", "1"}:
                raise RuntimeError(f"非法选择标记：{policy}/{image_id}/{flag}")
            result[policy][image_id] = {
                **raw,
                "image_id": image_id,
                "sequence_id": raw["sequence_id"].strip(),
                "selected_for_c2_2048": flag == "1",
            }
    for policy in ("Loose", "Medium"):
        if set(result[policy]) != vis_ids:
            raise RuntimeError(f"{policy} assignment没有且仅有覆盖247张VisDrone图片")
        n = sum(x["selected_for_c2_2048"] for x in result[policy].values())
        if n != EXPECTED_SELECTED_COUNTS[policy]:
            raise RuntimeError(f"{policy}选择数异常：预期{EXPECTED_SELECTED_COUNTS[policy]}，实际{n}")
    return result


def read_fold_rule_maps(vis_groups: set[str]) -> dict[str, dict[str, dict[str, Any]]]:
    data = load_json(FOLD_RULES_JSON)
    protocol = data.get("protocol", {})
    if protocol.get("group_key") != "sequence_id" or int(protocol.get("n_splits", -1)) != 5:
        raise RuntimeError("selector_fold_rules.json不是5折sequence_id GroupKFold规则")
    maps: dict[str, dict[str, dict[str, Any]]] = {}
    for policy in ("Loose", "Medium"):
        group_map: dict[str, dict[str, Any]] = {}
        for fold in data.get("fold_rules", {}).get(policy, []):
            rule = fold.get("rule", {})
            for group in fold.get("test_groups", []):
                group = str(group)
                if group in group_map:
                    raise RuntimeError(f"{policy}规则重复覆盖序列：{group}")
                group_map[group] = rule
        if len(group_map) != EXPECTED_VIS_GROUPS or set(group_map) != vis_groups:
            raise RuntimeError(f"{policy}折外规则没有准确覆盖94个VisDrone序列")
        maps[policy] = group_map
    return maps


def check_common_inputs() -> dict[str, Any]:
    require_files([FEATURE_CSV, ASSIGNMENTS_CSV, FOLD_RULES_JSON, OOF_SUMMARY_JSON,
                   GT_JSON, A_PRED_JSON, C2_PRED_JSON, REFERENCE_JSON])
    summary = load_json(OOF_SUMMARY_JSON)
    if summary.get("status") != "completed":
        raise RuntimeError("Stage 6.3C summary不是completed状态")
    protocol = summary.get("protocol", {})
    expected = {"n_images": EXPECTED_TOTAL, "n_splits": 5, "group_key": "sequence_id",
                "imgsz": IMGSZ, "conf": CONF, "nms_iou": NMS_IOU}
    for key, value in expected.items():
        actual = protocol.get(key)
        ok = (actual is not None and math.isclose(float(actual), value, abs_tol=1e-9)) if isinstance(value, float) else actual == value
        if not ok:
            raise RuntimeError(f"Stage 6.3C协议不一致：{key} expected={value}, actual={actual}")

    reference = load_json(REFERENCE_JSON)
    rp = reference.get("protocol", {})
    ref_expected = {"evaluator_version": EXPECTED_EVALUATOR_VERSION, "imgsz": IMGSZ,
                    "conf": CONF, "nms_iou": NMS_IOU, "nms_max_det": MAX_DET,
                    "overlap": OVERLAP, "half": HALF, "augment": AUGMENT}
    for key, value in ref_expected.items():
        actual = rp.get(key)
        ok = (actual is not None and math.isclose(float(actual), value, abs_tol=1e-9)) if isinstance(value, float) else actual == value
        if not ok:
            raise RuntimeError(f"faster_eval_metrics.json协议不一致：{key} expected={value}, actual={actual}")

    features = read_feature_rows()
    if len(features) != EXPECTED_TOTAL:
        raise RuntimeError(f"特征表图片数应为{EXPECTED_TOTAL}，实际{len(features)}")
    counts = Counter(x["source"] for x in features.values())
    if dict(counts) != EXPECTED_SOURCE_COUNTS:
        raise RuntimeError(f"数据来源数量异常：{dict(counts)}")

    gt = load_json(GT_JSON)
    gt_images = {int(x["id"]): x for x in gt["images"]}
    if set(gt_images) != set(features):
        raise RuntimeError("GT image_id与corrected_scene_features.csv不一致")
    for image_id, row in features.items():
        info = gt_images[image_id]
        if normalize_name(row["file_name"]) != normalize_name(info["file_name"]):
            raise RuntimeError(f"image_id={image_id} file_name映射不一致")
        if row["width"] != int(info["width"]) or row["height"] != int(info["height"]):
            raise RuntimeError(f"image_id={image_id}图片尺寸与GT不一致")
        if row["sequence_id"] != sequence_id_from_name(info["file_name"]):
            raise RuntimeError(f"image_id={image_id} sequence_id与Stage 6.3A规则不一致")

    vis_ids = {i for i, x in features.items() if x["source"] == "visdrone"}
    vis_groups = {features[i]["sequence_id"] for i in vis_ids}
    if len(vis_groups) != EXPECTED_VIS_GROUPS:
        raise RuntimeError(f"VisDrone序列数应为{EXPECTED_VIS_GROUPS}，实际{len(vis_groups)}")
    assignments = read_assignments(vis_ids)
    rule_maps = read_fold_rule_maps(vis_groups)

    hashes = {"features": sha256_file(FEATURE_CSV), "original": sha256_file(A_PRED_JSON),
              "c2": sha256_file(C2_PRED_JSON), "gt": sha256_file(GT_JSON)}
    for key, actual in hashes.items():
        expected_hash = summary.get("input_sha256", {}).get(key)
        if expected_hash and expected_hash != actual:
            raise RuntimeError(f"Stage 6.3C输入哈希不一致：{key}")
    ref_hash = reference.get("prediction_sha256", {})
    if ref_hash.get("A_Original") != hashes["original"] or ref_hash.get("C2_area2048") != hashes["c2"]:
        raise RuntimeError("原图/C2预测文件哈希与faster_eval_metrics.json不一致")
    if reference.get("ground_truth", {}).get("sha256") != hashes["gt"]:
        raise RuntimeError("GT哈希与faster_eval_metrics.json不一致")

    return {"features": features, "gt": gt, "gt_images": gt_images,
            "vis_ids": vis_ids, "assignments": assignments, "rule_maps": rule_maps,
            "source_counts": dict(counts), "input_sha256": hashes}


# ---------------------------------------------------------------------------
# Scene features and rule execution (live inference path)
# ---------------------------------------------------------------------------
def compute_scene_features(result: tuple[np.ndarray, np.ndarray, np.ndarray],
                           width: int, height: int) -> dict[str, Any]:
    boxes, scores, _classes = result
    keep = np.asarray(scores, dtype=np.float32) >= FEATURE_CONF
    boxes = np.asarray(boxes, dtype=np.float32)[keep]
    n_det = int(len(boxes))
    if n_det == 0:
        return {"n_det_conf025": 0, "median_box_area_640": None, "tiny_ratio": 0.0}
    scale = min(FEATURE_IMGSZ / float(width), FEATURE_IMGSZ / float(height))
    bw = boxes[:, 2] - boxes[:, 0]
    bh = boxes[:, 3] - boxes[:, 1]
    if np.any(bw <= 0) or np.any(bh <= 0):
        raise RuntimeError("预测框存在非正宽高，无法计算场景特征")
    areas = (bw * scale) * (bh * scale)
    if not np.all(np.isfinite(areas)):
        raise RuntimeError("预测框面积出现NaN/Inf")
    tiny_ratio = float(np.sum(areas < TINY_AREA_THRESHOLD)) / n_det
    return {"n_det_conf025": n_det,
            "median_box_area_640": round(float(np.median(areas)), 4),
            "tiny_ratio": round(tiny_ratio, 6)}


def compare_live_feature(image_id: int, live: dict[str, Any], saved: dict[str, Any]) -> None:
    if live["n_det_conf025"] != saved["n_det_conf025"]:
        raise RuntimeError(f"image_id={image_id} n_det_conf025不一致：live={live['n_det_conf025']} saved={saved['n_det_conf025']}")
    lm, sm = live["median_box_area_640"], saved["median_box_area_640"]
    if lm is None or sm is None:
        if lm is not None or sm is not None:
            raise RuntimeError(f"image_id={image_id} median_box_area_640空值状态不一致")
    elif not math.isclose(float(lm), float(sm), rel_tol=0.0, abs_tol=1e-3):
        raise RuntimeError(f"image_id={image_id} median_box_area_640不一致：live={lm} saved={sm}")
    if not math.isclose(float(live["tiny_ratio"]), float(saved["tiny_ratio"]), rel_tol=0.0, abs_tol=1e-6):
        raise RuntimeError(f"image_id={image_id} tiny_ratio不一致：live={live['tiny_ratio']} saved={saved['tiny_ratio']}")


def apply_scene_rule(feature: dict[str, Any], rule: dict[str, Any]) -> bool:
    if rule.get("rule_type") == "none" or not rule.get("conditions"):
        return False
    for cond in rule["conditions"]:
        value = feature.get(cond["feature"])
        if value is None:
            value = float("inf")  # Same missing-median behavior as Stage 6.3C.
        value, threshold, op = float(value), float(cond["threshold"]), cond["op"]
        if op == ">=" and value < threshold:
            return False
        if op == "<=" and value > threshold:
            return False
        if op not in {">=", "<="}:
            raise RuntimeError(f"未知规则运算符：{op}")
    return True


# ---------------------------------------------------------------------------
# Image lookup, 2x2 inference and C2 fusion (same as slice_val_selective.py)
# ---------------------------------------------------------------------------
def build_image_index(image_dir: Path):
    files = [p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    if not files:
        raise RuntimeError(f"验证图片目录为空：{image_dir}")
    by_relative: dict[str, Path] = {}
    by_basename: dict[str, list[Path]] = defaultdict(list)
    for p in files:
        by_relative[normalize_name(p.relative_to(image_dir))] = p
        by_basename[p.name].append(p)
    return by_relative, by_basename


def resolve_image_path(filename: str, image_dir: Path, by_relative: dict[str, Path], by_basename: dict[str, list[Path]]) -> Path:
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
    return (np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int64))


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
    valid = ((boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1]) & np.isfinite(scores))
    boxes, scores, classes = boxes[valid], scores[valid], classes[valid]
    if len(boxes) == 0:
        return empty_result()
    keep = batched_nms(torch.from_numpy(boxes), torch.from_numpy(scores), torch.from_numpy(classes), NMS_IOU)[:MAX_DET].cpu().numpy()
    return boxes[keep], scores[keep], classes[keep]


def predict_original(model, image, pred_args, torch, batched_nms):
    result = model.predict(source=image, batch=1, **pred_args)[0]
    boxes, scores, classes = extract_result(result)
    h, w = image.shape[:2]
    return merge_nms(boxes, scores, classes, w, h, torch, batched_nms)


def predict_tiled(model, image, pred_args, torch, batched_nms):
    tiles, offsets = make_tiles(image)
    results = model.predict(source=tiles, batch=len(tiles), **pred_args)
    if len(results) != len(tiles):
        raise RuntimeError(f"切片数{len(tiles)}与返回结果数{len(results)}不一致")
    all_b, all_s, all_c = [], [], []
    for result, (ox, oy) in zip(results, offsets):
        b, s, c = extract_result(result, ox, oy)
        if len(b):
            all_b.append(b); all_s.append(s); all_c.append(c)
    if not all_b:
        return empty_result(), len(tiles)
    h, w = image.shape[:2]
    merged = merge_nms(np.concatenate(all_b), np.concatenate(all_s), np.concatenate(all_c), w, h, torch, batched_nms)
    return merged, len(tiles)


def fuse_c2_area2048(original, tiled, width, height, torch, batched_nms):
    b0, s0, c0 = original
    b1, s1, c1 = tiled
    if len(b1):
        areas = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
        keep = areas < AREA_THRESHOLD  # strict <, matching Stage 6.1
        b1, s1, c1 = b1[keep], s1[keep], c1[keep]
    if len(b1) == 0:
        return original
    if len(b0) == 0:
        boxes, scores, classes = b1, s1, c1
    else:
        boxes = np.concatenate([b0, b1]); scores = np.concatenate([s0, s1]); classes = np.concatenate([c0, c1])
    return merge_nms(boxes, scores, classes, width, height, torch, batched_nms)


def prediction_to_coco(image_id: int, result, category_id: int):
    boxes, scores, _classes = result
    output = []
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = map(float, box)
        output.append({"image_id": int(image_id), "category_id": int(category_id),
                       "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(score)})
    return output


def prediction_signature(records: list[dict[str, Any]]) -> Counter:
    return Counter((int(x["category_id"]), round(float(x["score"]), 7), tuple(round(float(v), 5) for v in x["bbox"])) for x in records)


def load_verify_predictions(path: Path, wanted_ids: set[int]):
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        for item in json.load(f):
            image_id = int(item["image_id"])
            if image_id in wanted_ids:
                grouped[image_id].append(item)
    return grouped


def synchronize_if_needed(torch, device: str):
    if torch.cuda.is_available() and str(device).lower() not in {"cpu", "-1"}:
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
def run_benchmark(args: argparse.Namespace) -> None:
    common = check_common_inputs()
    features, gt_images = common["features"], common["gt_images"]
    assignments, rule_maps, vis_ids = common["assignments"], common["rule_maps"], common["vis_ids"]
    evaluator_version = check_evaluator_version()

    try:
        import cv2
        import torch
        from torchvision.ops import batched_nms
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError("需要opencv-python、torch、torchvision和ultralytics") from exc

    weights = Path(args.weights)
    image_dir = Path(args.image_dir)
    if not weights.is_absolute(): weights = ROOT / weights
    if not image_dir.is_absolute(): image_dir = ROOT / image_dir
    require_files([weights])
    if not image_dir.is_dir():
        raise FileNotFoundError(image_dir)
    by_relative, by_basename = build_image_index(image_dir)

    records = []
    for image_id in sorted(features):
        info, feat = gt_images[image_id], features[image_id]
        records.append({"image_id": image_id, "source": feat["source"],
                        "file_name": info["file_name"], "sequence_id": feat["sequence_id"],
                        "image_path": resolve_image_path(info["file_name"], image_dir, by_relative, by_basename),
                        "width": int(info["width"]), "height": int(info["height"])})
    if len(records) != EXPECTED_TOTAL:
        raise RuntimeError(f"待测图片应为{EXPECTED_TOTAL}张，实际{len(records)}张")
    for rec in records:
        if not rec["image_path"].is_file(): raise FileNotFoundError(rec["image_path"])
    vis_records = [r for r in records if r["source"] == "visdrone"]
    uavdt_records = [r for r in records if r["source"] == "uavdt"]
    if len(vis_records) != 247 or len(uavdt_records) != 1320:
        raise RuntimeError("UAVDT/VisDrone待测图片数量不符合固定协议")

    device = str(args.device)
    pred_args = {"imgsz": IMGSZ, "conf": CONF, "iou": NMS_IOU, "max_det": MAX_DET,
                 "device": device, "half": HALF, "rect": False, "augment": AUGMENT,
                 "agnostic_nms": False, "verbose": False, "save": False}

    print("=" * 74)
    print("Stage 6.3D-B | corrected live-selector runtime benchmark")
    print("weights:", weights); print("image_dir:", image_dir); print("device:", device)
    print("images: 1567 (UAVDT=1320, VisDrone=247)")
    print("OOF selected: Loose=154, Medium=120")
    print("imgsz=640 conf=0.001 nms_iou=0.7 max_det=300 overlap=0.15")
    print("repeats:", args.repeats, "warmup:", args.warmup)
    print("=" * 74)

    model = YOLO(str(weights))
    if len(model.names) != 1:
        raise RuntimeError(f"模型类别数应为1，当前为{len(model.names)}")

    # Warm-up (excluded from timing).
    warm_image = cv2.imread(str(records[0]["image_path"]))
    warm_vis = cv2.imread(str(vis_records[0]["image_path"]))
    if warm_image is None or warm_vis is None:
        raise RuntimeError("Warm-up图片读取失败")
    warm_tiles, _ = make_tiles(warm_vis)
    for _ in range(args.warmup):
        synchronize_if_needed(torch, device)
        model.predict(source=warm_image, batch=1, **pred_args)
        synchronize_if_needed(torch, device)
        model.predict(source=warm_tiles, batch=len(warm_tiles), **pred_args)
        synchronize_if_needed(torch, device)
    del warm_image, warm_vis, warm_tiles

    # Prediction identity checks on a deterministic subset.
    verify_ids: set[int] = set()
    if not args.skip_verify and args.verify_samples > 0:
        verify_ids.update(r["image_id"] for r in uavdt_records[:args.verify_samples])
        verify_ids.update(r["image_id"] for r in vis_records[:args.verify_samples])
    verify_a = load_verify_predictions(A_PRED_JSON, verify_ids) if verify_ids else {}
    verify_c2 = load_verify_predictions(C2_PRED_JSON, verify_ids) if verify_ids else {}
    verified_a: set[int] = set(); verified_c2: set[int] = set()

    feature_verified: set[int] = set()
    live_selected_counts = {"Loose": None, "Medium": None}
    selector_matches = {"Loose": 0, "Medium": 0}
    selector_checks = {"Loose": 0, "Medium": 0}
    repeat_summaries: list[dict[str, Any]] = []
    per_image_rows: list[dict[str, Any]] = []
    category_id = int(common["gt"]["categories"][0]["id"])

    for repeat in range(args.repeats):
        print(f"\n[REPEAT {repeat + 1}/{args.repeats}]")
        read_t, original_t, feature_t, tile_t, fusion_t = {}, {}, {}, {}, {}
        rule_t = {"Loose": {}, "Medium": {}}
        live_selected = {"Loose": set(), "Medium": set()}

        for idx, rec in enumerate(records, 1):
            image_id = rec["image_id"]
            t0 = time.perf_counter(); image = cv2.imread(str(rec["image_path"]))
            read_t[image_id] = time.perf_counter() - t0
            if image is None: raise RuntimeError(f"图片读取失败：{rec['image_path']}")
            h, w = image.shape[:2]
            if (w, h) != (rec["width"], rec["height"]):
                raise RuntimeError(f"图片尺寸与GT不一致：{rec['file_name']} actual={w}x{h} GT={rec['width']}x{rec['height']}")

            synchronize_if_needed(torch, device); t0 = time.perf_counter()
            original = predict_original(model, image, pred_args, torch, batched_nms)
            synchronize_if_needed(torch, device); original_t[image_id] = time.perf_counter() - t0

            if image_id in verify_ids and image_id not in verified_a:
                actual = prediction_to_coco(image_id, original, category_id)
                if prediction_signature(actual) != prediction_signature(verify_a.get(image_id, [])):
                    raise RuntimeError(f"A_Original预测不一致：image_id={image_id}。检查权重、推理参数、Ultralytics版本及NMS。")
                verified_a.add(image_id)

            if rec["source"] == "visdrone":
                # Feature extraction time is measured; audit comparison is excluded.
                t0 = time.perf_counter()
                live_feature = compute_scene_features(original, w, h)
                feature_t[image_id] = time.perf_counter() - t0
                compare_live_feature(image_id, live_feature, features[image_id])
                feature_verified.add(image_id)

                seq = rec["sequence_id"]
                for policy in ("Loose", "Medium"):
                    t1 = time.perf_counter()
                    selected = apply_scene_rule(live_feature, rule_maps[policy][seq])
                    rule_t[policy][image_id] = time.perf_counter() - t1
                    if selected: live_selected[policy].add(image_id)
                    expected_selected = assignments[policy][image_id]["selected_for_c2_2048"]
                    selector_checks[policy] += 1
                    if selected != expected_selected:
                        raise RuntimeError(f"{policy}实时选择与OOF分配不一致：image_id={image_id}, sequence_id={seq}, live={selected}, OOF={expected_selected}")
                    selector_matches[policy] += 1

                # Measure tile/fusion components on all VisDrone images once;
                # each policy total includes them only for its selected images.
                synchronize_if_needed(torch, device); t0 = time.perf_counter()
                tiled, n_tiles = predict_tiled(model, image, pred_args, torch, batched_nms)
                synchronize_if_needed(torch, device); tile_t[image_id] = time.perf_counter() - t0
                if n_tiles != 4: raise RuntimeError(f"image_id={image_id}切片数应为4，实际{n_tiles}")
                t0 = time.perf_counter()
                c2_result = fuse_c2_area2048(original, tiled, w, h, torch, batched_nms)
                fusion_t[image_id] = time.perf_counter() - t0

                if image_id in verify_ids and image_id not in verified_c2:
                    actual = prediction_to_coco(image_id, c2_result, category_id)
                    if prediction_signature(actual) != prediction_signature(verify_c2.get(image_id, [])):
                        raise RuntimeError(f"C2_area2048预测不一致：image_id={image_id}。检查切片坐标、面积过滤及NMS。")
                    verified_c2.add(image_id)

            if idx % 100 == 0 or idx == len(records):
                print(f"processed {idx}/{len(records)} | live Loose={len(live_selected['Loose'])} | Medium={len(live_selected['Medium'])}")

        for policy in ("Loose", "Medium"):
            if len(live_selected[policy]) != EXPECTED_SELECTED_COUNTS[policy]:
                raise RuntimeError(f"{policy}实时选择数量异常：{len(live_selected[policy])}")
        if repeat == 0:
            live_selected_counts = {p: len(live_selected[p]) for p in ("Loose", "Medium")}

        selected_by_policy = {"A_Original": set(), "C2_area2048": set(vis_ids),
                              "Loose": live_selected["Loose"], "Medium": live_selected["Medium"]}
        repeat_row: dict[str, Any] = {"repeat": repeat + 1}
        image_rows = []

        for policy, selected_ids in selected_by_policy.items():
            total_read = float(sum(read_t.values()))
            total_original = float(sum(original_t.values()))
            total_feature = float(sum(feature_t.values())) if policy in {"Loose", "Medium"} else 0.0
            total_rule = float(sum(rule_t[policy].values())) if policy in {"Loose", "Medium"} else 0.0
            total_tile = float(sum(tile_t[i] for i in selected_ids if i in tile_t))
            total_fusion = float(sum(fusion_t[i] for i in selected_ids if i in fusion_t))
            latencies = []

            for rec in records:
                image_id = rec["image_id"]; is_vis = rec["source"] == "visdrone"
                selected = image_id in selected_ids
                latency = read_t[image_id] + original_t[image_id]
                f_ms = r_ms = t_ms = fu_ms = 0.0
                if policy in {"Loose", "Medium"} and is_vis:
                    f_ms = feature_t[image_id] * 1000.0
                    r_ms = rule_t[policy][image_id] * 1000.0
                    latency += feature_t[image_id] + rule_t[policy][image_id]
                if selected:
                    t_ms = tile_t[image_id] * 1000.0
                    fu_ms = fusion_t[image_id] * 1000.0
                    latency += tile_t[image_id] + fusion_t[image_id]
                latencies.append(latency)
                image_rows.append({"repeat": repeat + 1, "policy": policy, "image_id": image_id,
                                   "source": rec["source"], "sequence_id": rec["sequence_id"],
                                   "selected_for_tiling": int(selected),
                                   "read_ms": read_t[image_id] * 1000.0,
                                   "original_ms": original_t[image_id] * 1000.0,
                                   "feature_ms": f_ms, "rule_ms": r_ms,
                                   "tile_ms": t_ms, "fusion_ms": fu_ms,
                                   "total_image_ms": latency * 1000.0})

            total = total_read + total_original + total_feature + total_rule + total_tile + total_fusion
            if not math.isclose(total, sum(latencies), rel_tol=1e-8, abs_tol=1e-6):
                raise RuntimeError(f"{policy}耗时分解与逐图耗时之和不一致")
            repeat_row[f"{policy}_total_seconds"] = total
            repeat_row[f"{policy}_images_per_second"] = len(records) / total if total else 0.0
            repeat_row[f"{policy}_mean_image_ms"] = float(np.mean(latencies) * 1000.0)
            repeat_row[f"{policy}_p95_image_ms"] = float(np.percentile(latencies, 95) * 1000.0)
            repeat_row[f"{policy}_selected_visdrone_images"] = len(selected_ids)
            repeat_row[f"{policy}_trigger_rate_visdrone"] = len(selected_ids) / len(vis_ids)
            repeat_row[f"{policy}_tile_crops"] = len(selected_ids) * 4
            for name, value in {"read": total_read, "original": total_original,
                                "feature": total_feature, "rule": total_rule,
                                "tile": total_tile, "fusion": total_fusion}.items():
                repeat_row[f"{policy}_{name}_seconds"] = value

        repeat_summaries.append(repeat_row); per_image_rows.extend(image_rows)
        print("Reconstructed totals:")
        for policy in selected_by_policy:
            print(f"  {policy:14s}: {repeat_row[f'{policy}_total_seconds']:.3f}s")

    # Final checks: feature parity and live OOF selector parity.
    if len(feature_verified) != 247:
        raise RuntimeError(f"场景特征验证不完整：{len(feature_verified)}/247")
    for policy in ("Loose", "Medium"):
        expected_checks = 247 * args.repeats
        if selector_checks[policy] != expected_checks or selector_matches[policy] != expected_checks:
            raise RuntimeError(f"{policy}规则复现不完整：{selector_matches[policy]}/{expected_checks}")

    expected_a = 0 if args.skip_verify or args.verify_samples == 0 else min(args.verify_samples, 1320) + min(args.verify_samples, 247)
    expected_c2 = 0 if args.skip_verify or args.verify_samples == 0 else min(args.verify_samples, 247)
    if len(verified_a) != expected_a or len(verified_c2) != expected_c2:
        raise RuntimeError(f"预测复核数量不完整：A={len(verified_a)}/{expected_a}, C2={len(verified_c2)}/{expected_c2}")

    policies = ["A_Original", "C2_area2048", "Loose", "Medium"]
    summary: dict[str, Any] = {
        "experiment": "Stage6.3D_B_Corrected_Live_Selector_Runtime_Benchmark",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "completed", "weights": str(weights), "weights_sha256": sha256_file(weights),
        "image_dir": str(image_dir), "device": device,
        "hardware": {"torch": str(torch.__version__), "cuda": getattr(torch.version, "cuda", None),
                     "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() and device.lower() not in {"cpu", "-1"} else "CPU"},
        "protocol": {"images": 1567, "source_counts": common["source_counts"], "imgsz": IMGSZ,
                     "conf": CONF, "feature_conf": FEATURE_CONF, "nms_iou": NMS_IOU,
                     "max_det": MAX_DET, "feature_imgsz": FEATURE_IMGSZ,
                     "tiny_area_threshold": TINY_AREA_THRESHOLD, "overlap": OVERLAP,
                     "tile_layout": "2x2", "tile_area_filter": "tile box area < 2048 in original-image pixels",
                     "half": HALF, "augment": AUGMENT, "warmup": args.warmup, "repeats": args.repeats,
                     "timing_method": "sum of measured per-image components; adaptive policies include all original inference, all VisDrone feature/rule costs, and tile/fusion only for selected VisDrone images",
                     "selector_execution": "live feature calculation + fold-specific GroupKFold rule; OOF assignments are audit-only",
                     "evaluator_version": evaluator_version},
        "input_sha256": common["input_sha256"],
        "selector_verification": {"feature_images_verified": len(feature_verified),
            "feature_values_match_stage6_3a": True,
            "rules_match_oof_assignments": {p: {"matched_decisions": selector_matches[p],
                "total_decisions": selector_checks[p], "match_rate": selector_matches[p] / selector_checks[p],
                "selected_images": live_selected_counts[p], "expected_selected_images": EXPECTED_SELECTED_COUNTS[p]}
                for p in ("Loose", "Medium")}, "result": "passed"},
        "prediction_verification": {"enabled": not args.skip_verify and args.verify_samples > 0,
            "verified_original_images": len(verified_a), "verified_c2_visdrone_images": len(verified_c2),
            "result": "skipped" if args.skip_verify else ("skipped_no_samples" if args.verify_samples == 0 else "passed")},
        "policies": {}, "repeat_results": repeat_summaries,
        "per_image_csv": str(OUT_PER_IMAGE.relative_to(ROOT))}

    for policy in policies:
        totals = np.asarray([x[f"{policy}_total_seconds"] for x in repeat_summaries], dtype=np.float64)
        fps = np.asarray([x[f"{policy}_images_per_second"] for x in repeat_summaries], dtype=np.float64)
        means = np.asarray([x[f"{policy}_mean_image_ms"] for x in repeat_summaries], dtype=np.float64)
        p95 = np.asarray([x[f"{policy}_p95_image_ms"] for x in repeat_summaries], dtype=np.float64)
        summary["policies"][policy] = {
            "selected_visdrone_images": repeat_summaries[-1][f"{policy}_selected_visdrone_images"],
            "trigger_rate_visdrone": repeat_summaries[-1][f"{policy}_trigger_rate_visdrone"],
            "tile_crops": repeat_summaries[-1][f"{policy}_tile_crops"],
            "total_seconds_mean": float(np.mean(totals)),
            "total_seconds_std": float(np.std(totals, ddof=1)) if len(totals) > 1 else 0.0,
            "images_per_second_mean": float(np.mean(fps)),
            "mean_image_ms_mean": float(np.mean(means)),
            "p95_image_ms_mean": float(np.mean(p95)),
            "component_seconds_mean": {component: float(np.mean([x[f"{policy}_{component}_seconds"] for x in repeat_summaries]))
                for component in ("read", "original", "feature", "rule", "tile", "fusion")}}

    c2_time = summary["policies"]["C2_area2048"]["total_seconds_mean"]
    for policy in ("A_Original", "Loose", "Medium"):
        t = summary["policies"][policy]["total_seconds_mean"]
        summary["policies"][policy]["speedup_vs_C2"] = c2_time / t if t > 0 else None
        summary["policies"][policy]["time_reduction_vs_C2_percent"] = (c2_time - t) / c2_time * 100.0 if c2_time > 0 else None

    write_csv(OUT_PER_IMAGE, per_image_rows)
    write_json(OUT_SUMMARY, summary)
    print("\nStage 6.3D-B completed.")
    print("feature parity: 247/247; live selector parity: 100%")
    for policy in policies:
        p = summary["policies"][policy]
        print(f"{policy:14s} {p['total_seconds_mean']:.3f}s | {p['images_per_second_mean']:.3f} img/s | P95 {p['p95_image_ms_mean']:.3f} ms")
    print("per-image:", OUT_PER_IMAGE)
    print("summary:", OUT_SUMMARY)


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 6.3D-B corrected live-selector runtime benchmark")
    parser.add_argument("--mode", choices=["benchmark"], default="benchmark",
                        help="本修正版仅运行Stage 6.3D-B；Stage 6.3D-A已完成")
    parser.add_argument("--weights", type=str, default=str(WEIGHT_DEFAULT))
    parser.add_argument("--image-dir", type=str, default=str(IMAGE_DIR_DEFAULT))
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--verify-samples", type=int, default=20)
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0 or args.verify_samples < 0:
        parser.error("repeats必须>=1，warmup和verify-samples必须>=0")
    return args


def main():
    args = parse_args()
    if not (ROOT / "ultralytics").is_dir():
        raise RuntimeError("请将本文件放在YOLO_Thesis仓库根目录运行")
    SCENE_DIR.mkdir(parents=True, exist_ok=True)
    run_benchmark(args)


if __name__ == "__main__":
    main()
