#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 8.5: four-model complexity and latency benchmark on VAL ONLY.

Reuses stage7_1_p2bridge_slice_validation.py's frozen Stage 7.1
original/2x2 tile/C2-2048 implementation, NOT the Stage 6 test manifest.

Four models are benchmarked original-only on all 1,567 val images.
The final model is also benchmarked with the Stage 7.1 source policy:
UAVDT -> original; VisDrone -> original + 2x2 tiled + C2-2048 fusion.

Parameters/GFLOPs use model_info() like this repository's val.py.
Latency includes model.predict preprocessing/inference/postprocessing and
Stage7's additional NMS/fusion. Disk reads excluded. Batch 1 original;
4 image tiles in one batch. GPU synchronization surrounds timed segments.

Completed models are saved atomically and can be skipped with --resume.
An interrupted model is measured again from the start, deliberately,
so benchmarking remains a complete uninterrupted repeated measurement.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import stage7_1_p2bridge_slice_validation as stage7

ROOT = Path(__file__).resolve().parent
STAGE7_VAL = ROOT / 'runs/detect/stage7_p2bridge_slice/stage7_1_validation/metrics_summary.json'
STAGE8_WEIGHT_MANIFEST = ROOT / 'runs/detect/stage8_complex_robustness_4models/model_weight_manifest.json'
OUT_DEFAULT = ROOT / 'runs/detect/stage8_complex_robustness_4models/stage8_5_val_runtime'

MODEL_PATHS = {
    'YOLO11s': ROOT / 'runs/detect/stage2_screening/YOLO11s/weights/best.pt',
    'YOLO11s_DySample': ROOT / 'runs/detect/stage2_screening/YOLO11s-DySample/weights/best.pt',
    'YOLO11s_DySample_P2Bridge': ROOT / 'runs/detect/stage2_screening/YOLO11s-DySample-P2Bridge/weights/best.pt',
    'YOLO11s_DySample_P2Bridge_MLCA_TD': ROOT / 'runs/detect/stage3_fast_screen/YOLO11s-DySample-P2Bridge-MLCA-TD/weights/best.pt',
}
MODEL_HASHES = {
    'YOLO11s': '80626ed8713c03445a575c87c460d9980e4093c2440f10c81bb4d39ca6a56056',
    'YOLO11s_DySample': '1a63af9a02c3970a71a2971890b84773851875971620c6eb9336becb090030bb',
    'YOLO11s_DySample_P2Bridge': '8027bf248e086bde5a043ce33a541e924bf17b93bb81a31790052831e50180cd',
    'YOLO11s_DySample_P2Bridge_MLCA_TD': '06f713f6c5513661280d7530a095f196151041c232bbe27ddd3d7b8c0b914a84',
}
FINAL = 'YOLO11s_DySample_P2Bridge_MLCA_TD'
EXPECTED = {'uavdt': 1320, 'visdrone': 247}


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    with path.open(encoding='utf-8') as stream:
        return json.load(stream)


def atomic_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    tmp.replace(path)


def write_csv(path: Path, rows):
    if not rows:
        raise ValueError('Refusing empty CSV')
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def describe(values):
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0 or not np.isfinite(arr).all() or np.min(arr) <= 0:
        raise RuntimeError('Latency samples missing or invalid')
    return {
        'n': len(arr),
        'mean_ms': float(np.mean(arr)),
        'median_ms': float(np.median(arr)),
        'p95_ms': float(np.percentile(arr, 95)),
        'p99_ms': float(np.percentile(arr, 99)),
        'min_ms': float(np.min(arr)),
        'max_ms': float(np.max(arr)),
        'FPS_from_mean_latency': float(1000.0 / np.mean(arr)),
    }


def full_path(value, default):
    path = Path(value).expanduser() if value else default
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def check_inputs(args, weights):
    # The val COCO GT and source mapping are those used in Stage7.1.
    gt_path = Path(args.gt).expanduser().resolve()
    img_dir = Path(args.image_dir).expanduser().resolve()
    source_path = Path(args.source_map).expanduser().resolve()
    for path in (gt_path, img_dir, source_path, STAGE7_VAL, STAGE8_WEIGHT_MANIFEST):
        if not path.exists():
            raise FileNotFoundError(path)

    ref = read_json(STAGE7_VAL)
    if ref.get('status') != 'completed' or ref['dataset']['split'] != 'validation/development':
        raise RuntimeError('Stage 7.1 reference is not completed VAL')
    if sha256(gt_path) != ref['dataset']['gt_sha256']:
        raise RuntimeError('VAL COCO GT hash differs from frozen Stage 7.1')
    if sha256(source_path) != ref['dataset']['source_map_sha256']:
        raise RuntimeError('VAL source mapping hash differs from frozen Stage 7.1')

    _, records, _, _, counts = stage7.load_records(gt_path, img_dir, source_path)
    records.sort(key=lambda x: int(x['image_id']))
    if len(records) != 1567 or counts != EXPECTED:
        raise RuntimeError(f'Incorrect VAL split: {len(records)}, {counts}')

    saved_weights = read_json(STAGE8_WEIGHT_MANIFEST)
    for name, path in weights.items():
        actual = sha256(path)
        if actual != MODEL_HASHES[name] or actual != saved_weights[name]['sha256']:
            raise RuntimeError(f'Weight SHA256 mismatch: {name}, {actual}')
        if name == FINAL and actual != ref['weights_sha256']:
            raise RuntimeError('Final weight does not match Stage7.1 VAL')
    return records, {
        'gt_sha256': sha256(gt_path),
        'source_map_sha256': sha256(source_path),
        'source_counts': counts,
        'images': len(records),
        'weights_sha256': MODEL_HASHES,
    }


def gpu_sync(torch, device):
    if str(device).lower() not in {'cpu', '-1'}:
        torch.cuda.synchronize(torch.device('cuda:' + str(device)) if str(device).isdigit() else torch.device(device))


def load_image(cv2, record):
    img = cv2.imread(str(record['path']))
    if img is None:
        raise RuntimeError(f"Cannot read image: {record['path']}")
    h, w = img.shape[:2]
    if (w, h) != (int(record['width']), int(record['height'])):
        raise RuntimeError(f"Image dimensions do not match val GT: {record['path']}")
    return img


def verify_inference(model, rows, device, do_c2, torch, cv2, bnms):
    args = stage7.pred_args(device)
    img = load_image(cv2, rows[0])
    o = stage7.predict_original(model, img, args, torch, bnms)
    if len(o) != 3:
        raise RuntimeError('Invalid original prediction')
    if do_c2:
        vis = next((r for r in rows if r['source'] == 'visdrone'), None)
        if vis is None:
            raise RuntimeError('Missing VisDrone VAL image for C2')
        img = load_image(cv2, vis)
        orig = stage7.predict_original(model, img, args, torch, bnms)
        tiles = stage7.predict_tiles(model, img, args, torch, bnms)
        fused = stage7.fuse_c2(orig, tiles, img.shape[1], img.shape[0], torch, bnms)
        if len(fused) != 3:
            raise RuntimeError('Invalid C2 fusion')
    gpu_sync(torch, device)


def run_benchmark(model, name, rows, device, warmup, repeats, do_c2, torch, cv2, bnms):
    args = stage7.pred_args(device)
    warm_orig = load_image(cv2, rows[0])
    warm_vis = load_image(cv2, next(x for x in rows if x['source'] == 'visdrone'))
    # Warmup outside timing. C2 warmup specifically exercises batch=4.
    for _ in range(warmup):
        stage7.predict_original(model, warm_orig, args, torch, bnms)
        if do_c2:
            base = stage7.predict_original(model, warm_vis, args, torch, bnms)
            tiled = stage7.predict_tiles(model, warm_vis, args, torch, bnms)
            stage7.fuse_c2(base, tiled, warm_vis.shape[1], warm_vis.shape[0], torch, bnms)
    gpu_sync(torch, device)

    records = []
    times = {'original_only': {'full': [], 'uavdt': [], 'visdrone': []}}
    if do_c2:
        times['C2_2048_policy'] = {'full': [], 'uavdt': [], 'visdrone': []}
        times['C2_tile_plus_fusion_extra'] = {'visdrone': []}

    for rep in range(1, repeats + 1):
        print(f'[RUN] {name} repeat {rep}/{repeats}, val={len(rows)}', flush=True)
        for n, row in enumerate(rows, 1):
            img = load_image(cv2, row)  # disk I/O excluded from timing
            src = row['source']
            gpu_sync(torch, device)
            t0 = time.perf_counter()
            original = stage7.predict_original(model, img, args, torch, bnms)
            gpu_sync(torch, device)
            orig_ms = (time.perf_counter() - t0) * 1000
            times['original_only']['full'].append(orig_ms)
            times['original_only'][src].append(orig_ms)
            records.append({
                'model': name, 'repeat': rep, 'image_id': row['image_id'],
                'source': src, 'mode': 'original_only', 'latency_ms': orig_ms,
            })
            if do_c2:
                additional_ms = 0.0
                if src == 'visdrone':
                    gpu_sync(torch, device)
                    t1 = time.perf_counter()
                    tiled = stage7.predict_tiles(model, img, args, torch, bnms)
                    stage7.fuse_c2(original, tiled, img.shape[1], img.shape[0], torch, bnms)
                    gpu_sync(torch, device)
                    additional_ms = (time.perf_counter() - t1) * 1000
                    times['C2_tile_plus_fusion_extra']['visdrone'].append(additional_ms)
                    records.append({
                        'model': name, 'repeat': rep, 'image_id': row['image_id'],
                        'source': src, 'mode': 'C2_tile_plus_fusion_extra', 'latency_ms': additional_ms,
                    })
                full_ms = orig_ms + additional_ms
                times['C2_2048_policy']['full'].append(full_ms)
                times['C2_2048_policy'][src].append(full_ms)
                records.append({
                    'model': name, 'repeat': rep, 'image_id': row['image_id'],
                    'source': src, 'mode': 'C2_2048_policy', 'latency_ms': full_ms,
                })
            if n % 200 == 0 or n == len(rows):
                print(f'  {name}: repeat {rep}, {n}/{len(rows)}', flush=True)
    stats = {mode: {scope: describe(vals) for scope, vals in groups.items()}
             for mode, groups in times.items()}
    return stats, records


def main():
    p = argparse.ArgumentParser(description='Stage8.5: four models, VAL only')
    p.add_argument('--image-dir', default=str(stage7.DEFAULTS['image_dir']))
    p.add_argument('--gt', default=str(stage7.DEFAULTS['gt']))
    p.add_argument('--source-map', default=str(stage7.DEFAULTS['source_map']))
    p.add_argument('--out-dir', default=str(OUT_DEFAULT))
    p.add_argument('--baseline-weights')
    p.add_argument('--dysample-weights')
    p.add_argument('--p2bridge-weights')
    p.add_argument('--final-weights')
    p.add_argument('--device', default='0')
    p.add_argument('--warmup', type=int, default=30)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--include-c2', action='store_true')
    p.add_argument('--preflight', action='store_true')
    p.add_argument('--limit-per-source', type=int, default=0,
                   help='Smoke only; 0 means all 1567 VAL images')
    p.add_argument('--resume', action='store_true', help='Skip saved complete models')
    args = p.parse_args()
    if args.warmup < 1 or args.repeats < 1 or args.limit_per_source < 0:
        p.error('warmup/repeats must be >=1; limit-per-source >=0')
    if args.device.lower() not in {'cpu', '-1'} and not (args.device.isdigit() or args.device.startswith('cuda:')):
        p.error('--device should be 0, cuda:0, or cpu')

    weights = {
        name: full_path(getattr(args, flag), MODEL_PATHS[name])
        for name, flag in [
            ('YOLO11s', 'baseline_weights'),
            ('YOLO11s_DySample', 'dysample_weights'),
            ('YOLO11s_DySample_P2Bridge', 'p2bridge_weights'),
            (FINAL, 'final_weights'),
        ]
    }
    rows, integrity = check_inputs(args, weights)
    if args.limit_per_source:
        k = args.limit_per_source
        if k > min(EXPECTED.values()):
            p.error('limit-per-source exceeds number of images in a source')
        rows = (
            [r for r in rows if r['source'] == 'uavdt'][:k] +
            [r for r in rows if r['source'] == 'visdrone'][:k]
        )
        rows.sort(key=lambda r: int(r['image_id']))
    print(f"[DATA] {'SMOKE' if args.limit_per_source else 'FORMAL'} VAL images={len(rows)}, sources={dict(Counter(x['source'] for x in rows))}", flush=True)

    import cv2
    import torch
    from torchvision.ops import batched_nms
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import model_info

    device = str(args.device)
    if device.lower() not in {'cpu', '-1'} and not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available')
    device_obj = torch.device('cpu' if device.lower() in {'cpu', '-1'} else
                              (f'cuda:{device}' if device.isdigit() else device))
    gpu_name = torch.cuda.get_device_name(device_obj) if device_obj.type == 'cuda' else 'CPU'
    hardware = {'GPU': gpu_name, 'device': str(device_obj), 'torch': str(torch.__version__),
                'CUDA': str(torch.version.cuda), 'OpenCV': str(cv2.__version__),
                'Python': sys.version.split()[0], 'platform': platform.platform()}
    protocol = {
        'dataset_split': 'val', 'dataset_reference': 'Stage7.1 validation COCO GT and source map',
        'images': len(rows), 'source_counts': dict(Counter(x['source'] for x in rows)),
        'imgsz': stage7.IMGSZ, 'conf': stage7.CONF, 'nms_iou': stage7.NMS_IOU,
        'max_det': stage7.MAX_DET, 'half': False, 'augment': False,
        'original_batch': 1, 'tile_batch': 4, 'warmup': args.warmup,
        'repeats': args.repeats, 'include_c2': args.include_c2,
        'model_GFLOPs_method': 'ultralytics.utils.torch_utils.model_info (same as val.py)',
        'latency_scope': 'model.predict preprocess+forward+postprocess, Stage7 global NMS; excludes disk reads',
        'C2_policy': 'UAVDT original-only; VisDrone original+4 tiles (2x2 overlap=0.15)+C2 area<2048; fusion included',
        'note': 'Source-aware split uses dataset source labels (not an automatic deployed source classifier).'
    }
    signature = {'hardware': hardware, 'protocol': protocol, 'input_sha256': integrity,
                 'paths': {k: str(v) for k, v in weights.items()}}
    if args.preflight:
        for name, path in weights.items():
            model = YOLO(str(path))
            if len(model.names) != 1:
                raise RuntimeError(f'{name}: expected nc=1')
            verify_inference(model, rows, device, args.include_c2 and name == FINAL,
                             torch, cv2, batched_nms)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f'[PREFLIGHT PASS] {name}', flush=True)
        print('PREFLIGHT PASS: VAL source mapping, GT, model hashes and C2 policy verified.')
        return

    out = Path(args.out_dir).expanduser().resolve()
    meta_path = out / 'benchmark_manifest.json'
    if meta_path.exists():
        if not args.resume:
            raise FileExistsError(f'{meta_path} already exists; use --resume or another --out-dir')
        if read_json(meta_path) != signature:
            raise RuntimeError('Existing run manifest differs: cannot resume mixed configs')
    else:
        if out.exists() and any(out.iterdir()):
            raise FileExistsError(f'Nonempty output directory without manifest: {out}')
        atomic_json(meta_path, signature)

    models = {}
    all_samples = []
    for name, path in weights.items():
        result_file = out / 'per_model' / f'{name}.json'
        sample_file = out / 'per_model' / f'{name}_latency.csv'
        if args.resume and result_file.exists() and sample_file.exists():
            model_result = read_json(result_file)
            if model_result.get('status') == 'completed' and model_result.get('sha256') == MODEL_HASHES[name]:
                print(f'[SKIP completed] {name}', flush=True)
                models[name] = model_result
                with sample_file.open('r', encoding='utf-8-sig', newline='') as f:
                    all_samples.extend(csv.DictReader(f))
                continue
        print(f'[MODEL] {name}', flush=True)
        model = YOLO(str(path))
        if len(model.names) != 1:
            raise RuntimeError(f'{name}: nc != 1')
        model.model.to(device_obj)
        model.model.eval()
        # Initialize prediction backend (including possible fusion) before model_info;
        # val.py also calls model_info after validation/predictor initialization.
        verify_inference(model, rows, device, False, torch, cv2, batched_nms)
        layers, params, gradients, gflops = model_info(model.model, imgsz=640)
        if not (params > 0 and gflops > 0):
            raise RuntimeError(f'{name}: params/GFLOPs calculation invalid')
        do_c2 = bool(args.include_c2 and name == FINAL)
        stats, samples = run_benchmark(model, name, rows, device, args.warmup,
                                      args.repeats, do_c2, torch, cv2, batched_nms)
        result = {
            'status': 'completed', 'model': name, 'weights': str(path),
            'sha256': MODEL_HASHES[name], 'layers': int(layers),
            'parameters': int(params), 'parameters_M': float(params / 1e6),
            'gradients': int(gradients), 'GFLOPs_640_batch1': float(gflops),
            'latency': stats,
        }
        write_csv(sample_file, samples)
        atomic_json(result_file, result)
        models[name] = result
        all_samples.extend(samples)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[DONE] {name}: {params/1e6:.4f} M params, {gflops:.4f} GFLOPs, "
              f"{stats['original_only']['full']['FPS_from_mean_latency']:.2f} FPS", flush=True)

    # All results and aggregate CSVs (one benchmark per image per repeat per policy).
    if len(models) != 4:
        raise RuntimeError(f'Incomplete models: {len(models)}/4')
    expected_original = len(rows) * args.repeats * 4
    got_original = sum(1 for x in all_samples if x['mode'] == 'original_only')
    if got_original != expected_original:
        raise RuntimeError(f'Original samples: {got_original} vs {expected_original}')
    if args.include_c2:
        expected_policy = len(rows) * args.repeats
        expected_tile = sum(x['source'] == 'visdrone' for x in rows) * args.repeats
        for mode, expect in [('C2_2048_policy', expected_policy),
                             ('C2_tile_plus_fusion_extra', expected_tile)]:
            got = sum(1 for x in all_samples if x['mode'] == mode)
            if got != expect:
                raise RuntimeError(f'{mode}: {got} vs {expect}')

    final = {
        'status': 'completed', 'experiment': 'Stage8.5_Complexity_Runtime_VAL_4models',
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'hardware': hardware, 'protocol': protocol, 'input_sha256': integrity,
        'models': models,
        'Stage7_1_final_model_val_AP_reference': (
            {'original': read_json(STAGE7_VAL)['metrics']['A_original'],
             'C2_2048': read_json(STAGE7_VAL)['metrics']['B_C2_area2048']}
            if args.include_c2 and not args.limit_per_source else None
        ),
    }
    summary_rows = []
    for name, m in models.items():
        for mode, scopes in m['latency'].items():
            for scope, stats in scopes.items():
                summary_rows.append({
                    'model': name, 'mode': mode, 'scope': scope,
                    'val_images': protocol['images'], 'samples': stats['n'],
                    'Parameters_M': m['parameters_M'],
                    'GFLOPs_640_batch1': m['GFLOPs_640_batch1'],
                    **{k: v for k, v in stats.items() if k != 'n'}
                })
    write_csv(out / 'complexity_runtime_summary.csv', summary_rows)
    write_csv(out / 'runtime_samples.csv', all_samples)
    atomic_json(out / 'complexity_runtime_summary.json', final)
    print('[COMPLETED] VAL ONLY!')
    print('Summary:', out / 'complexity_runtime_summary.json')
    print('CSV:', out / 'complexity_runtime_summary.csv')
    print('Raw samples:', out / 'runtime_samples.csv')

if __name__ == '__main__':
    main()
