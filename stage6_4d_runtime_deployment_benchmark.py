#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 6.4D: standalone end-to-end runtime benchmark for frozen A/B/C policies.

A: original inference only.
B: UAVDT original-only; all VisDrone images use 2x2 tiling + C2 fusion.
C: UAVDT original-only; VisDrone uses frozen M3 and tiles only if selected.

Unlike Stage 6.4B, A/B/C are run as separate pipelines. No training,
selector refitting, threshold tuning, or COCO evaluation is performed.
"""
from __future__ import annotations
import argparse, csv, gc, hashlib, json, shutil, time
from datetime import datetime
from pathlib import Path
from typing import Any
import numpy as np
import stage6_4b_independent_test as s64b

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs/detect/stage6_scene_selector/stage6_4d_runtime_deployment"
DECISIONS = ROOT / "runs/detect/stage6_scene_selector/stage6_4b_independent_test/selector_decisions.csv"

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def loadj(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def readcsv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def writej(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    tmp.replace(path)

def writecsv(path: Path, rows: list[dict[str, Any]]):
    if not rows:
        raise RuntimeError(f"拒绝写入空CSV：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), extrasaction="raise")
        w.writeheader(); w.writerows(rows)
    tmp.replace(path)

def run_policy(policy, repeat, rows, model, selector, cv2, torch,
               batched_nms, device, expected_decisions):
    pred_args = s64b.prediction_args(device)
    per_image = []
    tiled_ids = set()
    start_wall = time.perf_counter()

    for index, row in enumerate(rows, 1):
        iid = int(row["image_id"])
        s64b.synchronize(torch, device)
        image_start = time.perf_counter()

        t0 = time.perf_counter()
        image = cv2.imread(str(row["image_path"]))
        read_ms = (time.perf_counter() - t0) * 1000.0
        if image is None:
            raise RuntimeError(f"图片读取失败：{row['image_path']}")
        h, w = image.shape[:2]
        if (w, h) != (row["width"], row["height"]):
            raise RuntimeError(f"图片尺寸与manifest不一致：{row['file_name']}")

        s64b.synchronize(torch, device)
        t0 = time.perf_counter()
        original = s64b.predict_original(model, image, pred_args, torch, batched_nms)
        s64b.synchronize(torch, device)
        original_ms = (time.perf_counter() - t0) * 1000.0

        feature_ms = selector_ms = tile_ms = fusion_ms = 0.0
        selector_score = None
        tile_applied = False
        tile_count = 0

        if row["source"] == "visdrone" and policy == "B_C2_area2048":
            tile_applied = True
        elif row["source"] == "visdrone" and policy == "C_M3_selective":
            t0 = time.perf_counter()
            features = s64b.compute_scene_features(original, w, h)
            feature_ms = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            selector_score = s64b.selector_probability(selector, features)
            selector_ms = (time.perf_counter() - t0) * 1000.0
            tile_applied = selector_score >= s64b.SELECTOR_THRESHOLD

            expected = expected_decisions.get(iid)
            if expected is None:
                raise RuntimeError(f"Stage 6.4B缺少image_id={iid}选择器记录")
            if bool(int(expected["selected_for_tiling"])) != tile_applied:
                raise RuntimeError(f"Stage 6.4D与Stage 6.4B决策不一致：{iid}")
            if abs(float(expected["selector_score"]) - selector_score) > 1e-8:
                raise RuntimeError(f"Stage 6.4D与Stage 6.4B选择器分数不一致：{iid}")

        if tile_applied:
            tiled_ids.add(iid)
            s64b.synchronize(torch, device)
            t0 = time.perf_counter()
            tiled, tile_count = s64b.predict_tiled(
                model, image, pred_args, torch, batched_nms
            )
            s64b.synchronize(torch, device)
            tile_ms = (time.perf_counter() - t0) * 1000.0
            if tile_count != 4:
                raise RuntimeError(f"image_id={iid}切片数不是4")
            t0 = time.perf_counter()
            _fused = s64b.fuse_c2_area2048(
                original, tiled, w, h, torch, batched_nms
            )
            fusion_ms = (time.perf_counter() - t0) * 1000.0

        s64b.synchronize(torch, device)
        total_ms = (time.perf_counter() - image_start) * 1000.0
        per_image.append({
            "repeat": repeat, "policy": policy, "image_id": iid,
            "source": row["source"], "sequence_id": row["sequence_id"],
            "tile_applied": int(tile_applied),
            "selector_score": "" if selector_score is None else selector_score,
            "read_ms": read_ms, "original_ms": original_ms,
            "feature_ms": feature_ms, "selector_ms": selector_ms,
            "tile_ms": tile_ms, "fusion_ms": fusion_ms,
            "end_to_end_image_ms": total_ms, "tile_count": tile_count,
        })

        if index % 100 == 0 or index == len(rows):
            print(f"[repeat {repeat}] {policy}: {index}/{len(rows)} | "
                  f"tiled={len(tiled_ids)}", flush=True)

    wall = time.perf_counter() - start_wall
    lat = np.asarray([x["end_to_end_image_ms"] for x in per_image], dtype=np.float64)
    vis_count = sum(r["source"] == "visdrone" for r in rows)
    result = {
        "repeat": repeat, "policy": policy, "status": "completed",
        "images": len(rows), "visdrone_images": vis_count,
        "tiled_visdrone_images": len(tiled_ids),
        "trigger_rate_visdrone": len(tiled_ids) / vis_count,
        "tile_crops": sum(int(x["tile_count"]) for x in per_image),
        "wall_seconds": float(wall),
        "images_per_second": float(len(rows) / wall),
        "mean_image_ms": float(np.mean(lat)),
        "p50_image_ms": float(np.percentile(lat, 50)),
        "p95_image_ms": float(np.percentile(lat, 95)),
        "p99_image_ms": float(np.percentile(lat, 99)),
        "component_seconds": {
            key: float(sum(float(x[key]) for x in per_image) / 1000.0)
            for key in ("read_ms", "original_ms", "feature_ms",
                        "selector_ms", "tile_ms", "fusion_ms")
        },
        "timing_scope": (
            "Standalone full-test policy execution. Includes image read, inference "
            "and policy-specific feature/selector/tile/fusion. Excludes model loading "
            "and warm-up. Wall time includes loop and synchronization overhead."
        ),
    }
    return result, per_image

def main():
    ap = argparse.ArgumentParser(description="Stage 6.4D standalone A/B/C runtime benchmark")
    ap.add_argument("--dataset-root", default="datasets/HighwayVehicle-v1.0")
    ap.add_argument("--manifest", default="runs/detect/stage6_scene_selector/stage6_4_preflight/independent_test_manifest.csv")
    ap.add_argument("--weights", default="runs/detect/stage2_screening/YOLO11s-DySample/weights/best.pt")
    ap.add_argument("--device", default="0")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.repeats < 1 or args.warmup < 0:
        ap.error("repeats必须>=1，warmup必须>=0")
    if args.resume and args.overwrite:
        ap.error("--resume与--overwrite不能同时使用")
    if OUT.exists() and any(OUT.iterdir()):
        if args.overwrite:
            shutil.rmtree(OUT)
        elif not args.resume:
            raise FileExistsError(f"{OUT}已有内容；续跑用--resume，重做用--overwrite")
    OUT.mkdir(parents=True, exist_ok=True)

    dataset_root, manifest, weights = map(
        lambda x: Path(x).expanduser(), (args.dataset_root, args.manifest, args.weights)
    )
    if not dataset_root.is_absolute(): dataset_root = ROOT / dataset_root
    if not manifest.is_absolute(): manifest = ROOT / manifest
    if not weights.is_absolute(): weights = ROOT / weights

    rows, _fs, _meta, selector, input_hashes = s64b.validate_inputs(
        dataset_root, manifest, weights
    )
    if not DECISIONS.is_file():
        raise FileNotFoundError(DECISIONS)
    decision_rows = readcsv(DECISIONS)
    decisions = {int(r["image_id"]): r for r in decision_rows}
    if len(decisions) != 84:
        raise RuntimeError("Stage 6.4B selector_decisions.csv应有84条")

    run_meta = {
        "experiment": "Stage6.4D_Standalone_EndToEnd_Runtime_Benchmark",
        "repeats": args.repeats, "warmup": args.warmup, "device": args.device,
        "input_sha256": input_hashes, "selector_decisions_sha256": sha(DECISIONS),
    }
    meta_path = OUT / "checkpoint_metadata.json"
    if args.resume:
        if not meta_path.is_file() or loadj(meta_path) != run_meta:
            raise RuntimeError("续跑metadata不存在或参数/输入SHA256发生变化")
    else:
        writej(meta_path, run_meta)

    try:
        import cv2
        import torch
        from torchvision.ops import batched_nms
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError("需要opencv-python、torch、torchvision、ultralytics") from exc

    model = YOLO(str(weights))
    if len(model.names) != 1:
        raise RuntimeError(f"模型类别数应为1，实际{len(model.names)}")
    pred_args = s64b.prediction_args(args.device)
    first = cv2.imread(str(rows[0]["image_path"]))
    first_vis = next(r for r in rows if r["source"] == "visdrone")
    vis_image = cv2.imread(str(first_vis["image_path"]))
    if first is None or vis_image is None:
        raise RuntimeError("warmup图片读取失败")
    tiles, _ = s64b.make_tiles(vis_image)
    for _ in range(args.warmup):
        s64b.synchronize(torch, args.device)
        model.predict(source=first, batch=1, **pred_args)
        s64b.synchronize(torch, args.device)
        model.predict(source=tiles, batch=len(tiles), **pred_args)
        s64b.synchronize(torch, args.device)
    del first, vis_image, tiles
    gc.collect()

    policies = ["A_Original", "B_C2_area2048", "C_M3_selective"]
    results, all_per_image = [], []
    for repeat in range(1, args.repeats + 1):
        shift = (repeat - 1) % 3
        order = policies[shift:] + policies[:shift]
        for policy in order:
            result_path = OUT / f"repeat_{repeat}_{policy}.json"
            csv_path = OUT / f"repeat_{repeat}_{policy}_per_image.csv"
            if args.resume and result_path.is_file() and csv_path.is_file():
                cached = loadj(result_path)
                if cached.get("input_sha256") != input_hashes:
                    raise RuntimeError(f"{result_path.name}输入SHA不一致")
                result, per_rows = cached["result"], readcsv(csv_path)
                print(f"[RESUME] repeat={repeat} policy={policy}", flush=True)
            else:
                print(f"\n[RUN] repeat={repeat} policy={policy}", flush=True)
                result, per_rows = run_policy(
                    policy, repeat, rows, model, selector, cv2, torch,
                    batched_nms, args.device, decisions,
                )
                writecsv(csv_path, per_rows)
                writej(result_path, {"status": "completed",
                                     "input_sha256": input_hashes, "result": result})
            results.append(result); all_per_image.extend(per_rows)

    writecsv(OUT / "runtime_end_to_end_repeat_metrics.csv", results)
    writecsv(OUT / "runtime_end_to_end_per_image.csv", all_per_image)
    policy_summary = {}
    for policy in policies:
        sub = [r for r in results if r["policy"] == policy]
        if len(sub) != args.repeats:
            raise RuntimeError(f"{policy}重复次数不完整")
        policy_summary[policy] = {}
        for key in ("wall_seconds", "images_per_second", "mean_image_ms",
                    "p50_image_ms", "p95_image_ms", "p99_image_ms"):
            vals = np.asarray([float(x[key]) for x in sub], dtype=np.float64)
            policy_summary[policy][key] = {
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "min": float(vals.min()), "max": float(vals.max()),
            }
        policy_summary[policy]["tiled_visdrone_images"] = [
            int(x["tiled_visdrone_images"]) for x in sub
        ]

    a = policy_summary["A_Original"]["wall_seconds"]["mean"]
    b = policy_summary["B_C2_area2048"]["wall_seconds"]["mean"]
    c = policy_summary["C_M3_selective"]["wall_seconds"]["mean"]
    summary = {
        "experiment": "Stage6.4D_Standalone_EndToEnd_Runtime_Benchmark",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "completed", "protocol": run_meta,
        "timing_scope": (
            "A/B/C executed as separate full-test pipelines; includes image read, "
            "inference and policy-specific operations; excludes model initialization "
            "and warm-up. This is direct end-to-end timing, not Stage 6.4B's "
            "component-reconstructed estimate."
        ),
        "policy_summary": policy_summary,
        "relative_runtime": {
            "C_vs_A_total_time_change_percent": (c / a - 1) * 100 if a else None,
            "C_vs_B_total_time_change_percent": (c / b - 1) * 100 if b else None,
            "C_vs_A_fps_change_percent": (
                policy_summary["C_M3_selective"]["images_per_second"]["mean"]
                / policy_summary["A_Original"]["images_per_second"]["mean"] - 1
            ) * 100,
        },
        "input_sha256": input_hashes,
        "outputs": {"repeat_metrics": "runtime_end_to_end_repeat_metrics.csv",
                    "per_image": "runtime_end_to_end_per_image.csv"},
    }
    writej(OUT / "runtime_end_to_end_summary.json", summary)
    print("\nStage 6.4D completed.")
    print(json.dumps(policy_summary, ensure_ascii=False, indent=2))
    print("Relative runtime:", summary["relative_runtime"])
    print("Output:", OUT)

if __name__ == "__main__":
    main()
