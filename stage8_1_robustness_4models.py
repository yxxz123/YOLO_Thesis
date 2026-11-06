#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 8.1–8.4: controlled environmental robustness and source-stratified evaluation.

Models:
  1) YOLO11s
  2) YOLO11s-DySample
  3) YOLO11s-DySample-P2Bridge
  4) YOLO11s-DySample-P2Bridge-MLCA-TD

Conditions:
  clean + Gaussian blur (3 levels) + low light (3) +
  backlight (3) + fog/haze (3).

Corruptions are applied online to the frozen Stage 6.4A2 independent-test images.
The original GT boxes, image IDs, image dimensions, COCO evaluator, and inference
settings are unchanged. This script does not train models and does not use slicing.
Run from the YOLO_Thesis repository root.

Important interpretation:
  UAVDT/VisDrone metrics are source-stratified results of models trained on the
  mixed HighwayVehicle dataset. They are not strict zero-shot cross-domain transfer.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import shutil
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import stage6_4b_independent_test as stage6

ROOT = Path(__file__).resolve().parent
DATASET_DEFAULT = ROOT / "datasets/HighwayVehicle-v1.0"
PREFLIGHT = ROOT / "runs/detect/stage6_scene_selector/stage6_4_preflight"
MANIFEST_DEFAULT = PREFLIGHT / "independent_test_manifest.csv"
GT_DEFAULT = ROOT / "runs/detect/stage6_scene_selector/stage6_4b_independent_test/test_gt_coco.json"
STAGE7_REFERENCE = ROOT / "runs/detect/stage7_p2bridge_slice/stage7_2_independent_test/metrics_summary.json"
OUT_DEFAULT = ROOT / "runs/detect/stage8_complex_robustness_4models"

MODEL_CANDIDATES = {
    "YOLO11s": [
        ROOT / "runs/detect/stage2_screening/YOLO11s/weights/best.pt",
    ],
    "YOLO11s_DySample": [
        ROOT / "runs/detect/stage2_screening/YOLO11s-DySample/weights/best.pt",
    ],
    "YOLO11s_DySample_P2Bridge": [
        ROOT / "runs/detect/stage2_screening/YOLO11s-DySample-P2Bridge/weights/best.pt",
    ],
    "YOLO11s_DySample_P2Bridge_MLCA_TD": [
        ROOT / "runs/detect/stage3_fast_screen/YOLO11s-DySample-P2Bridge-MLCA-TD/weights/best.pt",
    ],
}

EXPECTED_IMAGES = 1372
EXPECTED_SOURCE_COUNTS = {"uavdt": 1288, "visdrone": 84}
EXPECTED_FINAL_SHA256 = "06f713f6c5513661280d7530a095f196151041c232bbe27ddd3d7b8c0b914a84"
EXPECTED_DYSAMPLE_SHA256 = "1a63af9a02c3970a71a2971890b84773851875971620c6eb9336becb090030bb"

CONDITIONS = [
    {"name": "clean", "type": "clean", "severity": 0, "parameter": "none"},
    {"name": "blur_s1", "type": "blur", "severity": 1, "kernel": 3},
    {"name": "blur_s2", "type": "blur", "severity": 2, "kernel": 5},
    {"name": "blur_s3", "type": "blur", "severity": 3, "kernel": 9},
    {"name": "lowlight_s1", "type": "lowlight", "severity": 1, "gamma": 1.25},
    {"name": "lowlight_s2", "type": "lowlight", "severity": 2, "gamma": 1.60},
    {"name": "lowlight_s3", "type": "lowlight", "severity": 3, "gamma": 2.00},
    {"name": "backlight_s1", "type": "backlight", "severity": 1, "gain": 0.92, "flare": 0.12},
    {"name": "backlight_s2", "type": "backlight", "severity": 2, "gain": 0.84, "flare": 0.24},
    {"name": "backlight_s3", "type": "backlight", "severity": 3, "gain": 0.76, "flare": 0.36},
    {"name": "fog_s1", "type": "fog", "severity": 1, "alpha": 0.12},
    {"name": "fog_s2", "type": "fog", "severity": 2, "alpha": 0.24},
    {"name": "fog_s3", "type": "fog", "severity": 3, "alpha": 0.36},
]

METRICS = ("AP50-95", "AP50", "APs", "APm", "APl")
SCOPES = ("full", "uavdt", "visdrone")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def resolve_weight(cli_value: str | None, candidates: list[Path], label: str) -> Path:
    if cli_value:
        path = Path(cli_value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{label}权重不存在：{path}")
        return path
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(
        f"未找到{label}权重。已检查：\n" + "\n".join(str(x) for x in candidates)
        + f"\n请通过 --{label.lower().replace('-', '_')}-weights 指定实际best.pt路径。"
    )


def make_rows(dataset_root: Path, manifest_path: Path, gt_path: Path,
              stage7_reference_path: Path):
    import csv as csv_module

    for p in (manifest_path, gt_path, stage7_reference_path):
        if not p.is_file():
            raise FileNotFoundError(f"必要输入不存在：{p}")
    reference = load_json(stage7_reference_path)
    if reference.get("status") != "completed":
        raise RuntimeError("Stage 7.2独立测试结果未完成")
    manifest_sha = sha256_file(manifest_path)
    gt_sha = sha256_file(gt_path)
    expected = reference.get("inputs_sha256", {})
    if expected.get("manifest") != manifest_sha:
        raise RuntimeError("当前独立测试manifest与Stage 7.2冻结manifest不一致")
    if expected.get("test_gt") != gt_sha:
        raise RuntimeError("当前COCO GT与Stage 7.2冻结GT不一致")

    gt = load_json(gt_path)
    gt_images = {int(x["id"]): x for x in gt.get("images", [])}
    if len(gt_images) != EXPECTED_IMAGES:
        raise RuntimeError(f"COCO GT应为{EXPECTED_IMAGES}张，实际{len(gt_images)}张")
    categories = gt.get("categories", [])
    if len(categories) != 1 or int(categories[0]["id"]) != 1:
        raise RuntimeError("GT必须是category_id=1的单类别车辆检测数据")
    stage6.CATEGORY_ID = 1

    image_dir = dataset_root / "images" / "test"
    label_dir = dataset_root / "labels" / "test"
    rows, counts, seen = [], Counter(), set()
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as f:
        raw_rows = list(csv_module.DictReader(f))
    if len(raw_rows) != EXPECTED_IMAGES:
        raise RuntimeError(f"manifest应为{EXPECTED_IMAGES}行，实际{len(raw_rows)}行")

    for raw in raw_rows:
        iid = int(raw["image_id"])
        if iid in seen:
            raise RuntimeError(f"manifest image_id重复：{iid}")
        seen.add(iid)
        if raw["split"] != "test":
            raise RuntimeError(f"manifest出现非test记录：{iid}")
        source = raw["source"].strip().lower()
        if source not in EXPECTED_SOURCE_COUNTS:
            raise RuntimeError(f"未知数据来源：{source}")
        rel = Path(raw["relative_path"])
        if rel.is_absolute() or ".." in rel.parts:
            raise RuntimeError(f"manifest相对路径非法：{rel}")
        image_path = (image_dir / rel).resolve()
        label_path = (label_dir / rel.with_suffix(".txt")).resolve()
        if not image_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(f"缺少图片或标签：{image_path} / {label_path}")
        info = gt_images.get(iid)
        if info is None:
            raise RuntimeError(f"GT缺少image_id={iid}")
        if Path(info["file_name"]).name != Path(raw["file_name"]).name:
            raise RuntimeError(f"GT与manifest文件名不一致：{iid}")
        width, height = int(raw["width"]), int(raw["height"])
        if width != int(info["width"]) or height != int(info["height"]):
            raise RuntimeError(f"GT与manifest尺寸不一致：{iid}")
        rows.append({
            "image_id": iid,
            "file_name": raw["file_name"],
            "relative_path": str(rel),
            "source": source,
            "group_id": raw["group_id"],
            "width": width,
            "height": height,
            "image_path": image_path,
        })
        counts[source] += 1

    if counts != Counter(EXPECTED_SOURCE_COUNTS):
        raise RuntimeError(f"数据来源数量异常：{dict(counts)}")
    if seen != set(gt_images):
        raise RuntimeError("manifest与COCO GT的image_id集合不一致")
    return rows, gt, {
        "manifest_sha256": manifest_sha,
        "gt_sha256": gt_sha,
        "stage7_reference_sha256": sha256_file(stage7_reference_path),
        "source_counts": dict(counts),
    }


def apply_corruption(image: np.ndarray, condition: dict[str, Any], cv2) -> np.ndarray:
    kind = condition["type"]
    if kind == "clean":
        return image
    if kind == "blur":
        k = int(condition["kernel"])
        return cv2.GaussianBlur(image, (k, k), sigmaX=0, sigmaY=0)
    if kind == "lowlight":
        gamma = float(condition["gamma"])
        lut = np.clip(
            np.power(np.arange(256, dtype=np.float32) / 255.0, gamma) * 255.0,
            0, 255
        ).astype(np.uint8)
        return cv2.LUT(image, lut)
    if kind == "backlight":
        h, w = image.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = 0.72 * w, 0.18 * h
        sx, sy = max(1.0, 0.24 * w), max(1.0, 0.22 * h)
        mask = np.exp(
            -0.5 * (((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2)
        )
        gain = float(condition["gain"])
        flare = float(condition["flare"])
        out = image.astype(np.float32) * gain
        out += (mask[:, :, None] * flare * 255.0)
        return np.clip(out, 0, 255).astype(np.uint8)
    if kind == "fog":
        alpha = float(condition["alpha"])
        # BGR atmospheric-light color; image geometry and labels remain unchanged.
        atmosphere = np.empty_like(image, dtype=np.float32)
        atmosphere[:] = (238.0, 240.0, 242.0)
        out = image.astype(np.float32) * (1.0 - alpha) + atmosphere * alpha
        return np.clip(out, 0, 255).astype(np.uint8)
    raise ValueError(f"未知退化类型：{kind}")


def condition_metadata(condition: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": condition["name"],
        "type": condition["type"],
        "severity": condition["severity"],
        "parameters": {k: v for k, v in condition.items()
                       if k not in {"name", "type", "severity"}},
        "method": {
            "clean": "original image without corruption",
            "blur": "Gaussian blur",
            "lowlight": "gamma darkening: output=(input/255)^gamma*255",
            "backlight": "global exposure reduction plus deterministic upper-right Gaussian light flare",
            "fog": "global atmospheric-light blending",
        }[condition["type"]],
    }


def read_checkpoint(path: Path, valid_ids: set[int], source_by_id: dict[int, str]):
    records = []
    if not path.is_file():
        return records
    last_good = 0
    with path.open("rb") as f:
        for line in f:
            if not line.endswith(b"\n"):
                break
            try:
                rec = json.loads(line.decode("utf-8"))
            except Exception:
                break
            iid = int(rec["image_id"])
            if iid not in valid_ids or rec["source"] != source_by_id[iid]:
                raise RuntimeError(f"checkpoint出现未知image_id或来源不匹配：{iid}")
            if not isinstance(rec.get("predictions"), list):
                raise RuntimeError(f"checkpoint预测字段异常：{iid}")
            records.append(rec)
            last_good = f.tell()
    if path.stat().st_size != last_good:
        with path.open("r+b") as f:
            f.truncate(last_good)
        print("[RESUME] 已截断checkpoint末尾不完整记录", flush=True)
    ids = [int(x["image_id"]) for x in records]
    if len(ids) != len(set(ids)):
        raise RuntimeError("checkpoint包含重复image_id")
    return records


def run_one_condition(model, model_key: str, weight_path: Path, weight_sha: str,
                      condition: dict[str, Any], rows, gt_path: Path, out_dir: Path,
                      input_hashes: dict[str, Any], device: str, checkpoint_every: int,
                      resume: bool, keep_predictions: bool, example_count: int,
                      cv2, torch, batched_nms):
    cond_name = condition["name"]
    work_dir = out_dir / "checkpoints" / model_key / cond_name
    work_dir.mkdir(parents=True, exist_ok=True)
    meta_path = work_dir / "metadata.json"
    checkpoint_path = work_dir / "records.jsonl"
    result_path = out_dir / "condition_metrics" / model_key / f"{cond_name}.json"

    metadata = {
        "experiment": "Stage8_Environmental_Robustness",
        "model": model_key,
        "weights_sha256": weight_sha,
        "condition": condition_metadata(condition),
        "input_hashes": input_hashes,
        "inference": {
            "imgsz": stage6.IMGSZ, "conf": stage6.CONF,
            "nms_iou": stage6.NMS_IOU, "max_det": stage6.MAX_DET,
            "half": stage6.HALF, "augment": stage6.AUGMENT,
            "batch": 1, "slice_inference": False,
        },
        "device": str(device),
        "images": len(rows),
    }

    if result_path.is_file():
        old_result = load_json(result_path)
        if old_result.get("metadata") != metadata:
            raise RuntimeError(f"已完成结果与当前实验设置不一致：{result_path}")
        print(f"[SKIP] {model_key}/{cond_name} 已完成", flush=True)
        return old_result["metrics"]

    if meta_path.is_file():
        if not resume:
            raise FileExistsError(f"发现未完成checkpoint，请使用--resume：{work_dir}")
        old_meta = load_json(meta_path)
        if old_meta != metadata:
            raise RuntimeError(f"续跑元数据不一致：{work_dir}")
    else:
        if checkpoint_path.exists() and checkpoint_path.stat().st_size > 0:
            raise RuntimeError(f"checkpoint存在但metadata缺失：{work_dir}")
        write_json(meta_path, metadata)

    valid_ids = {int(r["image_id"]) for r in rows}
    source_by_id = {int(r["image_id"]): r["source"] for r in rows}
    records = read_checkpoint(checkpoint_path, valid_ids, source_by_id) if resume else []
    completed = {int(x["image_id"]) for x in records}
    print(f"[RUN] {model_key}/{cond_name}: {len(completed)}/{len(rows)}", flush=True)

    example_dir = out_dir / "corruption_examples" / cond_name
    examples_saved = 0
    for row in rows:
        if row["image_id"] in completed:
            continue
        image = cv2.imread(str(row["image_path"]))
        if image is None:
            raise RuntimeError(f"图片读取失败：{row['image_path']}")
        if image.shape[1] != row["width"] or image.shape[0] != row["height"]:
            raise RuntimeError(f"图片尺寸与GT不一致：image_id={row['image_id']}")
        corrupted = apply_corruption(image, condition, cv2)

        # Save a small, deterministic set of examples once per condition.
        if condition["type"] != "clean" and examples_saved < example_count:
            source_quota = max(1, example_count // 2)
            already_source = len(list(example_dir.glob(f"{row['source']}_*.jpg"))) if example_dir.exists() else 0
            if already_source < source_quota:
                example_dir.mkdir(parents=True, exist_ok=True)
                out_image = example_dir / f"{row['source']}_{row['image_id']}_{Path(row['file_name']).stem}.jpg"
                cv2.imwrite(str(out_image), corrupted)
                examples_saved += 1

        pred_args = stage6.prediction_args(device)
        prediction = stage6.predict_original(model, corrupted, pred_args, torch, batched_nms)
        pred_coco = stage6.prediction_to_coco(int(row["image_id"]), prediction)
        rec = {
            "image_id": int(row["image_id"]),
            "source": row["source"],
            "predictions": pred_coco,
        }
        with checkpoint_path.open("a", encoding="utf-8", buffering=1) as f:
            f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":"),
                               allow_nan=False) + "\n")
            if (len(completed) + 1) % checkpoint_every == 0 or len(completed) + 1 == len(rows):
                f.flush()
                os.fsync(f.fileno())
        records.append(rec)
        completed.add(int(row["image_id"]))
        if len(completed) % 100 == 0 or len(completed) == len(rows):
            print(f"[PROGRESS] {model_key}/{cond_name}: {len(completed)}/{len(rows)}",
                  flush=True)

    if len(records) != len(rows) or {int(x["image_id"]) for x in records} != valid_ids:
        raise RuntimeError(f"{model_key}/{cond_name}预测未覆盖全部测试图片")

    predictions = []
    for rec in records:
        predictions.extend(rec["predictions"])
    if not predictions:
        raise RuntimeError(f"{model_key}/{cond_name}预测为空")

    temp_pred = out_dir / "temporary_predictions" / model_key / f"{cond_name}.json"
    stage6.write_json_atomic(temp_pred, predictions)
    metrics = stage6.evaluate_predictions(
        gt_path, {f"{model_key}_{cond_name}": temp_pred}, rows
    )[f"{model_key}_{cond_name}"]

    result = {
        "status": "completed",
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "metadata": metadata,
        "metrics": metrics,
        "prediction_count": len(predictions),
        "prediction_sha256": stage6.prediction_digest(predictions),
    }
    write_json(result_path, result)

    if not keep_predictions and temp_pred.exists():
        temp_pred.unlink()
    elif keep_predictions:
        keep_path = out_dir / "predictions" / model_key / f"{cond_name}.json"
        keep_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(temp_pred), str(keep_path))

    # Archive the completed append-only checkpoint compactly.
    if checkpoint_path.is_file():
        archive_path = checkpoint_path.with_suffix(".jsonl.gz")
        with checkpoint_path.open("rb") as src, gzip.open(archive_path, "wb", compresslevel=6) as dst:
            shutil.copyfileobj(src, dst)
        checkpoint_path.unlink()

    print(f"[DONE] {model_key}/{cond_name}: "
          f"Full AP50-95={metrics['full']['AP50-95']:.5f}, "
          f"APs={metrics['full']['APs']:.5f}", flush=True)
    return metrics


def build_tables(all_results: dict[str, dict[str, Any]], out_dir: Path):
    csv_rows = []
    clean_lookup = {}
    for model_key, cond_results in all_results.items():
        if "clean" not in cond_results:
            continue
        for scope in SCOPES:
            clean_lookup[(model_key, scope)] = cond_results["clean"][scope]

    for model_key, cond_results in all_results.items():
        for cond in CONDITIONS:
            name = cond["name"]
            if name not in cond_results:
                continue
            for scope in SCOPES:
                values = cond_results[name][scope]
                clean = clean_lookup.get((model_key, scope), {})
                row = {
                    "model": model_key,
                    "condition": name,
                    "corruption": cond["type"],
                    "severity": cond["severity"],
                    "scope": scope,
                    "images": values["images"],
                }
                for metric in METRICS:
                    value = float(values[metric])
                    clean_value = float(clean.get(metric, float("nan")))
                    row[metric] = value
                    row[f"clean_{metric}"] = clean_value
                    row[f"delta_{metric}"] = value - clean_value if math.isfinite(clean_value) else ""
                    row[f"retention_{metric}_percent"] = (
                        value / clean_value * 100.0
                        if math.isfinite(clean_value) and clean_value > 0 else ""
                    )
                csv_rows.append(row)

    write_csv(out_dir / "robustness_metrics.csv", csv_rows)

    average = {}
    for model_key, cond_results in all_results.items():
        nonclean = [v for k, v in cond_results.items() if k != "clean"]
        average[model_key] = {}
        for scope in SCOPES:
            clean_ap = clean_lookup.get((model_key, scope), {}).get("AP50-95")
            ap_values = [x[scope]["AP50-95"] for x in nonclean if scope in x]
            aps_values = [x[scope]["APs"] for x in nonclean if scope in x]
            average[model_key][scope] = {
                "mean_corrupted_AP50-95": float(np.mean(ap_values)) if ap_values else None,
                "mean_corrupted_APs": float(np.mean(aps_values)) if aps_values else None,
                "clean_AP50-95": clean_ap,
                "mean_AP50-95_retention_percent": (
                    float(np.mean(ap_values) / clean_ap * 100.0)
                    if ap_values and clean_ap and clean_ap > 0 else None
                ),
                "worst_corruption_AP50-95": float(np.min(ap_values)) if ap_values else None,
            }
    return average


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DATASET_DEFAULT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_DEFAULT)
    parser.add_argument("--gt", type=Path, default=GT_DEFAULT)
    parser.add_argument("--stage7-reference", type=Path, default=STAGE7_REFERENCE)
    parser.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--baseline-weights", default=None)
    parser.add_argument("--dysample-weights", default=None)
    parser.add_argument("--p2bridge-weights", default=None,
                        help="纯DySample+P2Bridge模型的best.pt，必须显式指定")
    parser.add_argument("--final-weights", default=None)
    parser.add_argument("--device", default="0")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--example-count", type=int, default=4)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-predictions", action="store_true")
    parser.add_argument("--limit-images", type=int, default=0,
                        help="仅用于冒烟测试；正式实验必须为0")
    args = parser.parse_args()

    if args.checkpoint_every < 1 or args.example_count < 0 or args.limit_images < 0:
        parser.error("checkpoint-every必须>=1，example-count和limit-images必须>=0")
    if not args.dataset_root.is_dir():
        raise FileNotFoundError(f"数据集目录不存在：{args.dataset_root}")

    rows, gt, input_hashes = make_rows(
        args.dataset_root.resolve(), args.manifest.resolve(),
        args.gt.resolve(), args.stage7_reference.resolve()
    )
    weights = {
        "YOLO11s": resolve_weight(args.baseline_weights, MODEL_CANDIDATES["YOLO11s"], "baseline"),
        "YOLO11s_DySample": resolve_weight(args.dysample_weights, MODEL_CANDIDATES["YOLO11s_DySample"], "dysample"),
        "YOLO11s_DySample_P2Bridge": resolve_weight(
            args.p2bridge_weights, MODEL_CANDIDATES["YOLO11s_DySample_P2Bridge"], "p2bridge"
        ),
        "YOLO11s_DySample_P2Bridge_MLCA_TD": resolve_weight(
            args.final_weights, MODEL_CANDIDATES["YOLO11s_DySample_P2Bridge_MLCA_TD"], "final"
        ),
    }
    weight_hashes = {k: sha256_file(v) for k, v in weights.items()}
    if weight_hashes["YOLO11s_DySample"] != EXPECTED_DYSAMPLE_SHA256:
        raise RuntimeError("DySample权重SHA256与Stage 6.4B使用的best.pt不一致")
    if weight_hashes["YOLO11s_DySample_P2Bridge_MLCA_TD"] != EXPECTED_FINAL_SHA256:
        raise RuntimeError("P2Bridge-MLCA-TD权重SHA256与Stage 7.2使用的best.pt不一致")

    if args.limit_images:
        # Select from both sources for a useful smoke test; never use this subset for thesis metrics.
        per_source = max(1, args.limit_images // 2)
        chosen = []
        for source in ("uavdt", "visdrone"):
            chosen.extend([r for r in rows if r["source"] == source][:per_source])
        rows = sorted(chosen[:args.limit_images], key=lambda x: x["image_id"])

    print("=" * 78)
    print("Stage 8.1–8.4: environmental robustness + source-stratified metrics")
    print("Images:", len(rows), "Source counts:",
          dict(Counter(x["source"] for x in rows)))
    print("Models:")
    for k, v in weights.items():
        print(f"  {k}: {v} | sha256={weight_hashes[k]}")
    print("Conditions:", len(CONDITIONS), "(clean + 4 corruption types × 3 levels)")
    print("Inference: imgsz=640, conf=0.001, IoU=0.7, max_det=300, original-only")
    print("Slice inference: DISABLED to isolate architecture robustness")
    print("=" * 78, flush=True)

    if args.preflight:
        import cv2
        import torch
        from ultralytics import YOLO
        for name, path in weights.items():
            model = YOLO(str(path))
            if len(model.names) != 1:
                raise RuntimeError(f"{name}不是单类别模型：{model.names}")
            del model
        sample = cv2.imread(str(rows[0]["image_path"]))
        if sample is None:
            raise RuntimeError("测试图片读取失败")
        for c in CONDITIONS:
            transformed = apply_corruption(sample, c, cv2)
            if transformed.shape != sample.shape or transformed.dtype != np.uint8:
                raise RuntimeError(f"退化变换输出异常：{c['name']}")
        print("PREFLIGHT PASS：模型、GT、manifest、图片、退化函数全部通过。")
        return

    if args.limit_images:
        print("WARNING: --limit-images is active; this run is only a smoke test, not thesis evidence.",
              flush=True)

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "metrics_summary.json").exists() and not args.resume:
        raise FileExistsError("已有metrics_summary.json；续跑请使用--resume，正式重做请更换输出目录。")

    # Freeze all four model paths and hashes for the entire experiment.
    # This prevents different checkpoints from being mixed across resumed runs.
    weight_manifest = {
        name: {"path": str(path), "sha256": weight_hashes[name]}
        for name, path in weights.items()
    }
    weight_manifest_path = out_dir / "model_weight_manifest.json"
    if weight_manifest_path.is_file():
        old_manifest = load_json(weight_manifest_path)
        if old_manifest != weight_manifest:
            raise RuntimeError(
                "当前模型权重路径或SHA256与本次实验首次运行时不一致。"
                "禁止在同一输出目录混用不同权重；请恢复原权重或使用全新输出目录。"
            )
    else:
        write_json(weight_manifest_path, weight_manifest)

    import cv2
    import torch
    from torchvision.ops import batched_nms
    from ultralytics import YOLO

    all_results: dict[str, dict[str, Any]] = {}
    for model_key, weight_path in weights.items():
        print(f"\n{'#' * 24} MODEL: {model_key} {'#' * 24}", flush=True)
        model = YOLO(str(weight_path))
        if len(model.names) != 1:
            raise RuntimeError(f"{model_key}类别数不是1：{model.names}")
        model_results = {}
        for condition in CONDITIONS:
            metrics = run_one_condition(
                model, model_key, weight_path, weight_hashes[model_key],
                condition, rows, args.gt.resolve(), out_dir, input_hashes,
                args.device, args.checkpoint_every, args.resume,
                args.keep_predictions, args.example_count, cv2, torch, batched_nms
            )
            model_results[condition["name"]] = metrics
        all_results[model_key] = model_results
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    average = build_tables(all_results, out_dir)
    final = {
        "experiment": "Stage8_Environmental_Robustness_and_Source_Stratifed_Evaluation_4Models",
        "status": "completed",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "images": len(rows),
        "source_counts": dict(Counter(x["source"] for x in rows)),
        "models": {
            k: {"weights": str(v), "sha256": weight_hashes[k]}
            for k, v in weights.items()
        },
        "input_hashes": input_hashes,
        "protocol": {
            "imgsz": stage6.IMGSZ, "conf": stage6.CONF,
            "nms_iou": stage6.NMS_IOU, "max_det": stage6.MAX_DET,
            "half": stage6.HALF, "augment": stage6.AUGMENT,
            "batch": 1, "slice_inference": False,
            "evaluator": "faster_coco_eval",
            "evaluator_version": "1.8.0",
            "coco_max_dets": stage6.EVAL_MAX_DETS,
            "coco_area_ranges": stage6.EVAL_AREA_RANGES,
        },
        "conditions": [condition_metadata(x) for x in CONDITIONS],
        "metrics": all_results,
        "robustness_average": average,
        "interpretation": {
            "size_robustness": "APs/APm/APl are reported for every condition using unchanged GT area definitions.",
            "source_generalization": "These are source-stratified results for models trained on the mixed HighwayVehicle dataset; they are not strict zero-shot cross-domain transfer.",
            "synthetic_conditions": "Blur, low-light, backlight and fog are deterministic synthetic corruptions, not real meteorological field measurements.",
        },
        "outputs": {
            "metrics_csv": "robustness_metrics.csv",
            "summary_json": "metrics_summary.json",
            "examples_dir": "corruption_examples/",
        },
    }
    write_json(out_dir / "metrics_summary.json", final)
    print("\nStage 8.1–8.4 completed.")
    print("Summary:", out_dir / "metrics_summary.json")
    print("CSV:", out_dir / "robustness_metrics.csv")


if __name__ == "__main__":
    main()
