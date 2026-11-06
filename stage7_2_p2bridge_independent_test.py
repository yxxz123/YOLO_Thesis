#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 7.2 — P2Bridge-MLCA-TD, frozen original vs 2x2+C2-2048 independent test.

Exactly two policies on the same sequence-filtered Stage 6.4A2 test manifest:
 A_original : YOLO11s-DySample-P2Bridge-MLCA-TD original prediction on all 1,372 images.
 B_C2_area2048 : same original predictions for UAVDT; original+full 2x2 tiled
                 predictions+C2-2048 global NMS on the 84 VisDrone images.

No training, M3, model selection, parameter tuning, or cross-model experiment.
Shares the Stage 6.4B inference and COCO metric implementation through import.
Uses an *append-only* checkpoint instead of rewriting cumulative predictions every 10 images.

Run from the YOLO_Thesis repository root, with stage6_4b_independent_test.py present.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import shutil
import time
from collections import Counter, defaultdict
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from faster_coco_eval import COCO, COCOeval_faster
import stage6_4b_independent_test as stage6

ROOT = Path(__file__).resolve().parent
PREFLIGHT = ROOT / "runs/detect/stage6_scene_selector/stage6_4_preflight"
BASELINE_TEST = ROOT / "runs/detect/stage6_scene_selector/stage6_4b_independent_test"
DEFAULT_WEIGHTS = ROOT / "runs/detect/stage3_fast_screen/YOLO11s-DySample-P2Bridge-MLCA-TD/weights/best.pt"
DEFAULT_DATASET = ROOT / "datasets/HighwayVehicle-v1.0"
DEFAULT_MANIFEST = PREFLIGHT / "independent_test_manifest.csv"
DEFAULT_FILTER_SUMMARY = PREFLIGHT / "sequence_filter_summary.json"
DEFAULT_AUDIT = PREFLIGHT / "dataset_audit.json"
DEFAULT_GT = BASELINE_TEST / "test_gt_coco.json"
DEFAULT_STAGE6_METRICS = BASELINE_TEST / "metrics_summary.json"
DEFAULT_OUT = ROOT / "runs/detect/stage7_p2bridge_slice/stage7_2_independent_test"
OPTIONAL_VAL = ROOT / "runs/detect/stage7_p2bridge_slice/stage7_1_validation/metrics_summary.json"

# Freeze *the exact new model best.pt* used in the uploaded Stage 7.1 evaluation.
EXPECTED_WEIGHT_SHA256 = "06f713f6c5513661280d7530a095f196151041c232bbe27ddd3d7b8c0b914a84"
EXPECTED_VAL_GT_SHA256 = "75098d09c3a02e8f8e32eab99df6c977474748dd177a97b1692f766bbdcf0de4"
EXPECTED_EVAL_VERSION = "1.8.0"
EXPECTED_IMAGES = 1372
EXPECTED_SOURCES = {"uavdt": 1288, "visdrone": 84}
EXPECTED_GROUPS = {"uavdt": 5, "visdrone": 84}
EXPECTED_PARAMS = {
    "IMGSZ": 640,
    "CONF": 0.001,
    "NMS_IOU": 0.7,
    "MAX_DET": 300,
    "OVERLAP": 0.15,
    "HALF": False,
    "AUGMENT": False,
    "TILE_AREA_THRESHOLD": 2048.0,
}
METRIC_MAP = {
    "AP50-95": "AP_all",
    "AP50": "AP_50",
    "AP75": "AP_75",
    "APs": "AP_small",
    "APm": "AP_medium",
    "APl": "AP_large",
}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(p: Path):
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(p: Path, x: Any, compact: bool = False):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        if compact:
            json.dump(x, f, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        else:
            json.dump(x, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    tmp.replace(p)


def write_csv(p: Path, rows: list[dict[str, Any]]):
    if not rows:
        raise RuntimeError("无法写入空的CSV记录")
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    tmp.replace(p)


def ensure_inside(base: Path, path: Path) -> Path:
    base = base.resolve()
    path = path.resolve()
    if not path.is_relative_to(base):
        raise RuntimeError(f"路径越界：{path} 不在 {base} 内")
    return path


def audit_inputs(args):
    paths = {
        "weights": Path(args.weights).expanduser().resolve(),
        "dataset_root": Path(args.dataset_root).expanduser().resolve(),
        "manifest": Path(args.manifest).expanduser().resolve(),
        "sequence_filter": Path(args.sequence_filter).expanduser().resolve(),
        "dataset_audit": Path(args.dataset_audit).expanduser().resolve(),
        "gt": Path(args.gt).expanduser().resolve(),
        "stage6_metrics": Path(args.stage6_metrics).expanduser().resolve(),
    }
    for key, p in paths.items():
        if key == "dataset_root":
            if not p.is_dir():
                raise FileNotFoundError(f"{key}不存在：{p}")
        elif not p.is_file():
            raise FileNotFoundError(f"{key}不存在：{p}")

    if version("faster-coco-eval") != EXPECTED_EVAL_VERSION:
        raise RuntimeError(f"faster-coco-eval版本应为{EXPECTED_EVAL_VERSION}")
    for key, expected in EXPECTED_PARAMS.items():
        found = getattr(stage6, key)
        if found != expected:
            raise RuntimeError(f"Stage 6.4B冻结参数已变化：{key}={found}, expected={expected}")

    weight_sha = sha256_file(paths["weights"])
    if weight_sha != EXPECTED_WEIGHT_SHA256:
        raise RuntimeError(
            f"权重SHA256不一致，可能使用错误的P2Bridge变体\n"
            f"actual={weight_sha}\nexpected={EXPECTED_WEIGHT_SHA256}"
        )

    filter_info = read_json(paths["sequence_filter"])
    audit = read_json(paths["dataset_audit"])
    stage6_summary = read_json(paths["stage6_metrics"])
    gt_info = read_json(paths["gt"])
    if filter_info.get("status") != "READY_FOR_STAGE6_4B":
        raise RuntimeError("Stage 6.4A2独立测试序列过滤未通过")
    if (filter_info.get("test_train_overlap_after_filter") != 0 or
            filter_info.get("test_val_overlap_after_filter") != 0):
        raise RuntimeError("独立测试集仍存在已识别的Train/Val序列交叉")
    if stage6_summary.get("status") != "completed":
        raise RuntimeError("Stage 6.4B的基准独立测试结果未完成")

    expected_manifest = filter_info.get("output_sha256", {}).get("independent_test_manifest.csv")
    manifest_sha = sha256_file(paths["manifest"])
    if expected_manifest != manifest_sha:
        raise RuntimeError("独立测试manifest不匹配Stage6.4A2冻结哈希")
    stage6_input_sha = stage6_summary.get("input_sha256", {}).get("independent_test_manifest")
    if stage6_input_sha != manifest_sha:
        raise RuntimeError("独立测试manifest不匹配Stage6.4B的输入哈希")
    gt_sha = sha256_file(paths["gt"])
    if gt_sha != stage6_summary.get("ground_truth_sha256"):
        raise RuntimeError("COCO GT哈希不同于Stage6.4B已验证的测试GT")
    if len(gt_info.get("images", [])) != EXPECTED_IMAGES:
        raise RuntimeError("独立测试GT图片数不符合1372")
    cats = gt_info.get("categories", [])
    if len(cats) != 1 or int(cats[0]["id"]) != int(stage6_summary["protocol"]["coco_category_id"]):
        raise RuntimeError("COCO单类别ID与Stage6.4B不一致")
    cat_id = int(cats[0]["id"])
    stage6.CATEGORY_ID = cat_id
    if cat_id != 1:
        raise RuntimeError(f"本研究的COCO类别ID应为1，实际为{cat_id}")

    # Additional safeguard: input data and model protocol must match the new model's
    # Stage 7.1 validation results if that file is accessible locally.
    if OPTIONAL_VAL.is_file():
        validation = read_json(OPTIONAL_VAL)
        if validation.get("status") != "completed":
            raise RuntimeError("Stage7.1验证实验状态未完成")
        if validation.get("weights_sha256") != weight_sha:
            raise RuntimeError("Stage7.1与Stage7.2权重不是同一份best.pt")
        if validation.get("dataset", {}).get("gt_sha256") != EXPECTED_VAL_GT_SHA256:
            raise RuntimeError("Stage7.1 GT与原验证集口径不一致")
        for field, fixed in (("imgsz", 640), ("conf", .001), ("nms_iou", .7),
                             ("max_det", 300), ("overlap", .15)):
            if validation["protocol"].get(field) != fixed:
                raise RuntimeError(f"Stage7.1配置不是冻结参数：{field}")

    with paths["manifest"].open("r", encoding="utf-8-sig", newline="") as f:
        raws = list(csv.DictReader(f))
    if len(raws) != EXPECTED_IMAGES:
        raise RuntimeError(f"独立测试manifest应有1372行，实际{len(raws)}")

    image_folder = paths["dataset_root"] / "images" / "test"
    label_folder = paths["dataset_root"] / "labels" / "test"
    gt_images = {int(x["id"]): x for x in gt_info["images"]}
    rows, source_counts, source_groups = [], Counter(), defaultdict(set)
    ids = set()
    for raw in raws:
        iid = int(raw["image_id"])
        if iid in ids:
            raise RuntimeError(f"manifest重复image_id={iid}")
        ids.add(iid)
        if raw["split"] != "test":
            raise RuntimeError(f"非测试图片image_id={iid}")
        source = raw["source"].lower().strip()
        if source not in EXPECTED_SOURCES:
            raise RuntimeError(f"未知来源{source}")
        group_id = raw["group_id"].strip()
        seq = raw["sequence_id"].strip()
        if group_id != f"{source}:{seq}":
            raise RuntimeError(f"group_id与source/sequence_id不一致：{group_id}")
        if seq != stage6.sequence_id_from_name(raw["file_name"]):
            raise RuntimeError(f"sequence_id解析错误：image_id={iid}")

        rel = Path(raw["relative_path"])
        if rel.is_absolute() or ".." in rel.parts:
            raise RuntimeError(f"manifest relative_path非法：{rel}")
        image_path = ensure_inside(image_folder, image_folder / rel)
        label_path = ensure_inside(label_folder, label_folder / rel.with_suffix(".txt"))
        if not image_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(f"缺少测试图片/标签：{image_path}, {label_path}")
        if raw["label_exists"] != "1" or raw["label_errors"].strip():
            raise RuntimeError(f"manifest标签审计未通过：image_id={iid}")
        info = gt_images.get(iid)
        if info is None:
            raise RuntimeError(f"GT缺少image_id={iid}")
        if Path(info["file_name"]).name != Path(raw["file_name"]).name:
            raise RuntimeError(f"GT filename不匹配：image_id={iid}")
        width, height = int(raw["width"]), int(raw["height"])
        if width != int(info["width"]) or height != int(info["height"]):
            raise RuntimeError(f"GT尺寸不匹配：image_id={iid}")
        rows.append({"image_id": iid, "file_name": raw["file_name"],
                     "source": source, "group_id": group_id, "sequence_id": seq,
                     "width": width, "height": height, "image_path": image_path})
        source_counts[source] += 1
        source_groups[source].add(group_id)
    if ids != set(gt_images):
        raise RuntimeError("COCO GT与独立manifest image_id集合不一致")
    if dict(source_counts) != EXPECTED_SOURCES:
        raise RuntimeError(f"测试来源图片数错误：{dict(source_counts)}")
    group_counts = {k: len(v) for k, v in source_groups.items()}
    if group_counts != EXPECTED_GROUPS:
        raise RuntimeError(f"独立序列组数量错误：{group_counts}")

    overlap_groups = set(audit["sequence_overlap"]["train_vs_test"]["overlap_preview"])
    overlap_groups |= set(audit["sequence_overlap"]["val_vs_test"]["overlap_preview"])
    if any(r["group_id"] in overlap_groups for r in rows):
        raise RuntimeError("发现与Train/Val交叉的图片序列")

    fingerprints = {
        "weights": weight_sha, "manifest": manifest_sha,
        "filter_summary": sha256_file(paths["sequence_filter"]),
        "dataset_audit": sha256_file(paths["dataset_audit"]),
        "test_gt": gt_sha,
        "stage6_metrics": sha256_file(paths["stage6_metrics"]),
        "source_script": sha256_file(Path(stage6.__file__)),
    }
    return rows, gt_info, paths, fingerprints, group_counts


def read_checkpoint(path: Path, rows, manifest_hash):
    if not path.exists():
        return []
    by_id = {r["image_id"]: r for r in rows}
    completed = []
    last_good_byte = 0
    with path.open("rb") as f:
        for line in f:
            # Require complete newline-terminated records; discard a partial last line.
            if not line.endswith(b"\n"):
                break
            try:
                rec = json.loads(line.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                break
            iid = int(rec["image_id"])
            if iid not in by_id or iid in {x["image_id"] for x in completed}:
                raise RuntimeError(f"checkpoint存在重复或未知image_id={iid}")
            if rec["source"] != by_id[iid]["source"]:
                raise RuntimeError("checkpoint来源与manifest不一致")
            if rec.get("manifest_sha256") != manifest_hash:
                raise RuntimeError("checkpoint与本次manifest哈希不一致")
            if not isinstance(rec.get("A"), list) or not isinstance(rec.get("B"), list):
                raise RuntimeError("checkpoint缺少A/B预测列表")
            if not isinstance(rec.get("timing"), dict):
                raise RuntimeError("checkpoint缺少计时信息")
            completed.append(rec)
            last_good_byte = f.tell()
    if path.stat().st_size != last_good_byte:
        with path.open("r+b") as f:
            f.truncate(last_good_byte)
        print("[RESUME] 已丢弃未完整写入的最后一条checkpoint记录", flush=True)
    return completed


def evaluate(gt_path: Path, pred_paths: dict[str, Path], rows):
    coco_gt = COCO(str(gt_path))
    cats = sorted(int(x) for x in coco_gt.getCatIds())
    if cats != [1]:
        raise RuntimeError(f"测试GT类别异常：{cats}")
    scopes = {
        "full": sorted(r["image_id"] for r in rows),
        "uavdt": sorted(r["image_id"] for r in rows if r["source"] == "uavdt"),
        "visdrone": sorted(r["image_id"] for r in rows if r["source"] == "visdrone"),
    }
    result = {}
    for policy, pred_path in pred_paths.items():
        result[policy] = {}
        coco_dt = coco_gt.loadRes(str(pred_path))
        for name, image_ids in scopes.items():
            ev = COCOeval_faster(
                coco_gt, coco_dt, iouType="bbox", lvis_style=False,
                print_function=lambda *a, **k: None,
            )
            ev.params.imgIds = image_ids
            ev.params.catIds = cats
            ev.params.iouThrs = stage6.EVAL_IOU_THRESHOLDS.copy()
            ev.params.recThrs = stage6.EVAL_RECALL_THRESHOLDS.copy()
            ev.params.maxDets = stage6.EVAL_MAX_DETS.copy()
            ev.params.areaRng = [x.copy() for x in stage6.EVAL_AREA_RANGES]
            ev.params.areaRngLbl = stage6.EVAL_AREA_LABELS.copy()
            ev.params.useCats = 1
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
            stats = ev.stats_as_dict
            result[policy][name] = {"images": len(image_ids), **{
                label: float(stats[key]) for label, key in METRIC_MAP.items()
            }}
            print(f"[COCO] {policy} / {name}: "
                  f"AP50-95={result[policy][name]['AP50-95']:.6f} "
                  f"APs={result[policy][name]['APs']:.6f}", flush=True)
            del ev
        del coco_dt
    return result


def main():
    ap = argparse.ArgumentParser(
        description="Stage 7.2 frozen P2Bridge original vs C2-2048 independent test"
    )
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--dataset-root", default=str(DEFAULT_DATASET))
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--sequence-filter", default=str(DEFAULT_FILTER_SUMMARY))
    ap.add_argument("--dataset-audit", default=str(DEFAULT_AUDIT))
    ap.add_argument("--gt", default=str(DEFAULT_GT))
    ap.add_argument("--stage6-metrics", default=str(DEFAULT_STAGE6_METRICS))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--device", default="0")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--checkpoint-every", type=int, default=10)
    ap.add_argument("--preflight", action="store_true", help="仅验证权重/数据/冻结协议，不进行推理")
    ap.add_argument("--resume", action="store_true", help="根据追加式checkpoint续跑")
    ap.add_argument("--overwrite", action="store_true", help="明确重做时删除已有输出")
    args = ap.parse_args()
    if args.warmup < 0 or args.checkpoint_every < 1:
        ap.error("warmup必须>=0，checkpoint-every必须>=1")
    if args.resume and args.overwrite:
        ap.error("--resume与--overwrite不能同时使用")

    rows, gt, paths, fingerprints, group_counts = audit_inputs(args)
    out_dir = Path(args.out_dir).expanduser().resolve()
    print("=" * 78)
    print("Stage 7.2 — P2Bridge-MLCA-TD + frozen C2-2048 independent test")
    print("images", len(rows), "source_counts", EXPECTED_SOURCES)
    print("groups", group_counts, "weight_sha256", fingerprints["weights"])
    print("preflight=PASS; NO YOLO training, NO M3, NO threshold tuning", flush=True)
    print("=" * 78)
    if args.preflight:
        return

    metadata = {
        "experiment": "Stage7.2_P2Bridge_MLCA_TD_Independent_Test",
        "protocol": {
            "imgsz": 640, "conf": 0.001, "nms_iou": 0.7, "max_det": 300,
            "overlap": 0.15, "tile_layout": "2x2", "area_filter": "mapped tile area < 2048 px^2",
            "half": False, "augment": False,
            "source_policy": "UAVDT original-only; VisDrone original+full tiled C2-2048",
            "evaluator": "faster_coco_eval", "evaluator_version": EXPECTED_EVAL_VERSION,
            "coco_max_dets": stage6.EVAL_MAX_DETS,
            "coco_area_ranges": stage6.EVAL_AREA_RANGES,
            "no_m3": True,
        },
        "inputs": fingerprints,
        "manifest_images": len(rows),
        "groups": group_counts,
        "device": str(args.device),
    }
    meta_path = out_dir / "checkpoint_metadata.json"
    checkpoint_path = out_dir / "checkpoint_records.jsonl"
    if out_dir.exists() and any(out_dir.iterdir()):
        if args.overwrite:
            shutil.rmtree(out_dir)
        elif not args.resume:
            raise FileExistsError(f"输出目录已有内容：{out_dir}\n中断续跑用--resume，确认重做才用--overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.resume:
        if not meta_path.is_file():
            raise RuntimeError("--resume需要checkpoint_metadata.json")
        old = read_json(meta_path)
        if old != metadata:
            raise RuntimeError("续跑时输入文件哈希、冻结协议或设备参数已改变")
    else:
        write_json(meta_path, metadata)

    records = read_checkpoint(checkpoint_path, rows, fingerprints["manifest"]) if args.resume else []
    completed = {int(x["image_id"]) for x in records}
    print(f"[RESUME] 已完成{len(completed)} / {len(rows)}张图片", flush=True)

    if len(completed) < len(rows):
        import cv2
        import torch
        from torchvision.ops import batched_nms
        from ultralytics import YOLO
        model = YOLO(str(paths["weights"]))
        if len(model.names) != 1:
            raise RuntimeError(f"模型类别数应为1，当前{model.names}")
        pred_args = stage6.prediction_args(args.device)
        first_row = next(x for x in rows if x["image_id"] not in completed)
        first_image = cv2.imread(str(first_row["image_path"]))
        first_vis = next(x for x in rows if x["source"] == "visdrone")
        vis_image = cv2.imread(str(first_vis["image_path"]))
        if first_image is None or vis_image is None:
            raise RuntimeError("Warm-up图片读取失败")
        tiles, offsets = stage6.make_tiles(vis_image)
        if len(tiles) != 4:
            raise RuntimeError("历史2x2切片协议无法产生4张切片")
        for _ in range(args.warmup):
            stage6.synchronize(torch, args.device)
            model.predict(source=first_image, batch=1, **pred_args)
            stage6.synchronize(torch, args.device)
            model.predict(source=tiles, batch=4, **pred_args)
            stage6.synchronize(torch, args.device)
        del first_image, vis_image, tiles, offsets
        gc.collect()

        start_loop = time.perf_counter()
        with checkpoint_path.open("a", encoding="utf-8", buffering=1) as f:
            for row in rows:
                iid = int(row["image_id"])
                if iid in completed:
                    continue

                t = time.perf_counter()
                img = cv2.imread(str(row["image_path"]))
                read_ms = (time.perf_counter() - t) * 1000.0
                if img is None:
                    raise RuntimeError(f"无法读取测试图片：{row['image_path']}")
                height, width = img.shape[:2]
                if (width, height) != (row["width"], row["height"]):
                    raise RuntimeError(f"图片与GT尺寸不一致：image_id={iid}")

                stage6.synchronize(torch, args.device)
                t = time.perf_counter()
                original = stage6.predict_original(model, img, pred_args, torch, batched_nms)
                stage6.synchronize(torch, args.device)
                original_ms = (time.perf_counter() - t) * 1000.0
                pred_A = stage6.prediction_to_coco(iid, original)

                tile_ms, fuse_ms, tile_count = 0.0, 0.0, 0
                if row["source"] == "visdrone":
                    stage6.synchronize(torch, args.device)
                    t = time.perf_counter()
                    tiled, tile_count = stage6.predict_tiled(model, img, pred_args, torch, batched_nms)
                    stage6.synchronize(torch, args.device)
                    tile_ms = (time.perf_counter() - t) * 1000.0
                    if tile_count != 4:
                        raise RuntimeError(f"切片数量不是4：image_id={iid}")
                    t = time.perf_counter()
                    fused = stage6.fuse_c2_area2048(
                        original, tiled, width, height, torch, batched_nms
                    )
                    fuse_ms = (time.perf_counter() - t) * 1000.0
                else:
                    fused = original
                pred_B = stage6.prediction_to_coco(iid, fused)
                if row["source"] == "uavdt" and pred_A != pred_B:
                    raise RuntimeError(f"UAVDT原图与融合策略输出不一致：{iid}")

                one = {
                    "image_id": iid, "source": row["source"],
                    "group_id": row["group_id"],
                    "manifest_sha256": fingerprints["manifest"],
                    "A": pred_A,
                    "B": pred_B,
                    "timing": {
                        "image_id": iid, "file_name": row["file_name"],
                        "source": row["source"], "group_id": row["group_id"],
                        "a_boxes": len(pred_A), "b_boxes": len(pred_B),
                        "tile_count": tile_count,
                        "read_ms": read_ms, "original_ms": original_ms,
                        "tile_ms": tile_ms, "fusion_ms": fuse_ms,
                    },
                }
                f.write(json.dumps(one, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
                completed.add(iid)
                records.append(one)
                if len(completed) % args.checkpoint_every == 0 or len(completed) == len(rows):
                    f.flush()
                    os.fsync(f.fileno())
                    print(f"[CHECKPOINT] {len(completed)}/{len(rows)} saved, "
                          f"elapsed={time.perf_counter()-start_loop:.1f}s", flush=True)
        del model
        gc.collect()

    if len(records) != len(rows) or len({int(x["image_id"]) for x in records}) != len(rows):
        raise RuntimeError(f"未覆盖全部1372张独立测试图片：{len(records)}")
    records_by_id = {int(r["image_id"]): r for r in records}
    if set(records_by_id) != {r["image_id"] for r in rows}:
        raise RuntimeError("checkpoint预测image_id集合与manifest不一致")

    pred_A, pred_B, runtimes = [], [], []
    for row in rows:
        rec = records_by_id[row["image_id"]]
        pred_A.extend(rec["A"])
        pred_B.extend(rec["B"])
        runtimes.append(rec["timing"])
    if not pred_A or not pred_B:
        raise RuntimeError("两组预测列表中至少一个为空，无法进行COCO评价")

    p_A = out_dir / "test_predictions_A_original.json"
    p_B = out_dir / "test_predictions_B_C2_area2048.json"
    write_json(p_A, pred_A, compact=True)
    write_json(p_B, pred_B, compact=True)
    write_csv(out_dir / "runtime_per_image.csv", runtimes)

    results = evaluate(paths["gt"], {"A_original": p_A, "B_C2_area2048": p_B}, rows)
    delta = {scope: {
        metric: results["B_C2_area2048"][scope][metric] - results["A_original"][scope][metric]
        for metric in METRIC_MAP
    } for scope in ("full", "uavdt", "visdrone")}
    if any(abs(v) > 1e-12 for v in delta["uavdt"].values()):
        raise RuntimeError("UAVDT应该没有切片增益：发现两组预测差异")

    runtime = {}
    for policy in ("A_original", "B_C2_area2048"):
        totals = []
        for item in runtimes:
            elapsed = float(item["read_ms"]) + float(item["original_ms"])
            if policy == "B_C2_area2048":
                elapsed += float(item["tile_ms"]) + float(item["fusion_ms"])
            totals.append(elapsed)
        runtime[policy] = {
            "component_sum_seconds": float(sum(totals) / 1000.0),
            "mean_component_ms_per_image": float(np.mean(totals)),
            "p95_component_ms_per_image": float(np.percentile(totals, 95)),
            "images_per_component_second": float(len(rows) / (sum(totals) / 1000.0)),
        }

    full_summary = {
        "experiment": "Stage7.2_P2Bridge_MLCA_TD_Independent_Test",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "completed",
        "dataset": {
            "split": "independent_test_sequence_filtered_by_Stage6.4A2",
            "images": EXPECTED_IMAGES,
            "source_counts": EXPECTED_SOURCES,
            "group_counts": group_counts,
            "annotation_count": len(gt["annotations"]),
            "sequence_independence_scope": "Stage 6.4A2 sequence_id grouping heuristic",
        },
        "model": {
            "name": "YOLO11s-DySample-P2Bridge-MLCA-TD",
            "weights": str(paths["weights"]),
            "weights_sha256": fingerprints["weights"],
            "same_weights_A_and_B": True,
        },
        "protocol": metadata["protocol"],
        "metrics": results,
        "delta_B_minus_A": delta,
        "runtime_component_only": runtime,
        "component_timing_note": (
            "Component-sum cost is not a standalone end-to-end wall-clock benchmark. "
            "It excludes checkpoints, Python orchestration, model loading and warm-up."
        ),
        "max_det_300_saturation": {
            "A_original": sum(x["a_boxes"] >= 300 for x in runtimes),
            "B_C2_area2048": sum(x["b_boxes"] >= 300 for x in runtimes),
            "A_original_visdrone": sum(x["a_boxes"] >= 300 for x in runtimes if x["source"] == "visdrone"),
            "B_C2_area2048_visdrone": sum(x["b_boxes"] >= 300 for x in runtimes if x["source"] == "visdrone"),
        },
        "inputs_sha256": fingerprints,
        "predictions_sha256": {
            "A_original": sha256_file(p_A), "B_C2_area2048": sha256_file(p_B),
        },
        "outputs": {
            "predictions_A_original": str(p_A.relative_to(ROOT)),
            "predictions_B_C2_area2048": str(p_B.relative_to(ROOT)),
            "runtime_per_image": str((out_dir / "runtime_per_image.csv").relative_to(ROOT)),
            "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        },
        "interpretation_limit": (
            "This evaluates an already selected/frozen slice strategy on the Stage 6.4A2 "
            "sequence-filtered independent split, which has been used previously "
            "to assess a different DySample detector. Do not use these scores "
            "to tune slice/fusion parameters or select a new M3 threshold. "
            "84 VisDrone images represent 84 sequence-id groups; 1,288 UAVDT "
            "images represent five sequence-id groups."
        ),
    }
    write_json(out_dir / "metrics_summary.json", full_summary)
    print("\n" + "=" * 78)
    print("Stage 7.2 COMPLETED")
    print("Full delta:", json.dumps(delta["full"], ensure_ascii=False, indent=2))
    print("VisDrone delta:", json.dumps(delta["visdrone"], ensure_ascii=False, indent=2))
    print("Output:", out_dir)
    print("=" * 78)


if __name__ == "__main__":
    main()
