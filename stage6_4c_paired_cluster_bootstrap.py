#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 6.4C: paired sequence-cluster bootstrap on 84 independent VisDrone test sequences.

No YOLO training/inference, selector refitting, or threshold tuning.
The 84 independent sequence groups each contribute exactly one test image.
"""
from __future__ import annotations
import argparse, csv, gc, hashlib, json, math, shutil, time
from collections import defaultdict
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any
import numpy as np
from faster_coco_eval import COCO, COCOeval_faster

ROOT = Path(__file__).resolve().parent
SCENE = ROOT / "runs/detect/stage6_scene_selector"
FILTER = SCENE / "stage6_4_preflight"
TEST = SCENE / "stage6_4b_independent_test"
REPEAT = TEST / "repeats"
MANIFEST = FILTER / "independent_test_manifest.csv"
FILTER_SUMMARY = FILTER / "sequence_filter_summary.json"
GT_PATH = TEST / "test_gt_coco.json"
METRICS_PATH = TEST / "metrics_summary.json"
DECISIONS_PATH = TEST / "selector_decisions.csv"
PRED_PATHS = {
    "A_Original": TEST / "test_predictions_A_original.json",
    "B_C2_area2048": TEST / "test_predictions_B_C2_area2048.json",
    "C_M3_selective": TEST / "test_predictions_C_M3_selective.json",
}
OUT = SCENE / "stage6_4c_paired_cluster_bootstrap"
SAMPLES = OUT / "paired_cluster_bootstrap_samples.csv"
SUMMARY = OUT / "paired_cluster_bootstrap_summary.json"
SUBGROUP = OUT / "selected_unselected_subgroup_metrics.json"
META = OUT / "checkpoint_metadata.json"

NUMPY_VERSION = "1.26.4"
EVALUATOR_VERSION = "1.8.0"
N_VIS = 84
N_SELECTED = 60
THRESHOLD = 0.5071783812207581
METRICS = ["AP50-95", "AP50", "APs", "APm", "APl"]
METRIC_KEYS = {
    "AP50-95": "AP_all", "AP50": "AP_50", "APs": "AP_small",
    "APm": "AP_medium", "APl": "AP_large",
}
IOU = np.linspace(0.50, 0.95, 10, dtype=np.float64)
RECALL = np.linspace(0.0, 1.0, 101, dtype=np.float64)
MAX_DETS = [1, 10, 100]
AREA_RANGES = [[0, 1e10], [0, 1024], [1024, 9216], [9216, 1e10]]
AREA_LABELS = ["all", "small", "medium", "large"]

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
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)

def signature(items):
    return sorted((int(x["category_id"]), round(float(x["score"]), 5),
                   tuple(round(float(v), 4) for v in x["bbox"])) for x in items)

def load_pred(path, valid_ids, category_id):
    raw = loadj(path)
    if not isinstance(raw, list):
        raise RuntimeError(f"预测文件不是COCO detection list：{path}")
    out = defaultdict(list)
    for i, p in enumerate(raw):
        iid = int(p["image_id"])
        if iid not in valid_ids or int(p["category_id"]) != category_id:
            raise RuntimeError(f"{path.name}第{i}条image/category id异常")
        box = [float(v) for v in p["bbox"]]
        score = float(p["score"])
        if len(box) != 4 or not all(math.isfinite(v) for v in box) or not math.isfinite(score):
            raise RuntimeError(f"{path.name}第{i}条预测框或分数非法")
        out[iid].append(p)
    return dict(out)

def evaluate(coco_gt, preds, image_ids, category_ids):
    if not image_ids or not preds:
        raise RuntimeError("评价范围为空或没有预测框")
    dt = coco_gt.loadRes(preds)
    ev = COCOeval_faster(coco_gt, dt, iouType="bbox", lvis_style=False,
                         print_function=lambda *a, **k: None)
    ev.params.imgIds = sorted(map(int, image_ids))
    ev.params.catIds = list(category_ids)
    ev.params.iouThrs = IOU.copy()
    ev.params.recThrs = RECALL.copy()
    ev.params.maxDets = list(MAX_DETS)
    ev.params.areaRng = [x.copy() for x in AREA_RANGES]
    ev.params.areaRngLbl = list(AREA_LABELS)
    ev.params.useCats = 1
    ev.evaluate(); ev.accumulate(); ev.summarize()
    result = {m: float(ev.stats_as_dict[k]) for m, k in METRIC_KEYS.items()}
    del ev, dt
    return result

def make_boot_gt(base_gt, group_to_id, sampled_groups):
    image_by_id = {int(x["id"]): x for x in base_gt["images"]}
    anns_by_id = defaultdict(list)
    for ann in base_gt["annotations"]:
        anns_by_id[int(ann["image_id"])].append(ann)
    images, anns, pairs = [], [], []
    new_iid, new_aid = 1, 1
    for group in sampled_groups:
        old_iid = group_to_id[group]
        new_image = dict(image_by_id[old_iid]); new_image["id"] = new_iid
        images.append(new_image); pairs.append((old_iid, new_iid))
        for ann in anns_by_id.get(old_iid, []):
            copied = dict(ann); copied["id"] = new_aid; copied["image_id"] = new_iid
            anns.append(copied); new_aid += 1
        new_iid += 1
    ds = {"info": dict(base_gt.get("info", {})), "licenses": list(base_gt.get("licenses", [])),
          "images": images, "annotations": anns,
          "categories": [dict(x) for x in base_gt["categories"]]}
    coco = COCO(); coco.dataset = ds; coco.createIndex()
    return coco, list(range(1, len(images) + 1)), pairs

def remap_pred(index, pairs):
    out = []
    for old_iid, new_iid in pairs:
        for p in index.get(old_iid, []):
            q = dict(p); q["image_id"] = new_iid; out.append(q)
    return out

def dist(values):
    a = np.asarray(values, dtype=np.float64)
    if len(a) < 100 or not np.all(np.isfinite(a)):
        raise RuntimeError("Bootstrap样本不足或包含NaN/Inf")
    lo, hi = np.percentile(a, [2.5, 97.5])
    return {"mean": float(a.mean()), "median": float(np.median(a)),
            "std": float(a.std(ddof=1)), "ci95_low": float(lo), "ci95_high": float(hi),
            "ci95_excludes_zero": bool(lo > 0 or hi < 0),
            "probability_delta_gt_zero": float(np.mean(a > 0))}

def validate():
    required = [MANIFEST, FILTER_SUMMARY, GT_PATH, METRICS_PATH, DECISIONS_PATH, *PRED_PATHS.values()]
    missing = [p for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("缺少Stage 6.4输入：\n" + "\n".join(map(str, missing)))
    if np.__version__ != NUMPY_VERSION:
        raise RuntimeError(f"NumPy必须为{NUMPY_VERSION}，当前{np.__version__}")
    if version("faster-coco-eval") != EVALUATOR_VERSION:
        raise RuntimeError(f"faster-coco-eval必须为{EVALUATOR_VERSION}")

    fs, sm = loadj(FILTER_SUMMARY), loadj(METRICS_PATH)
    if fs.get("status") != "READY_FOR_STAGE6_4B" or sm.get("status") != "completed":
        raise RuntimeError("Stage 6.4A2/B状态未完成")
    if sm["protocol"]["evaluator_version"] != EVALUATOR_VERSION or sm["protocol"]["repeats"] != 3:
        raise RuntimeError("Stage 6.4B评价器版本或重复次数不符合预期")
    if sm["test_set"]["images"] != 1372 or sm["test_set"]["visdrone_selector_test_images"] != N_VIS:
        raise RuntimeError("Stage 6.4B测试集数量与预期不一致")
    if not math.isclose(float(sm["protocol"]["selector_threshold"]), THRESHOLD, abs_tol=1e-12):
        raise RuntimeError("M3阈值不是冻结阈值")

    manifest = readcsv(MANIFEST)
    if len(manifest) != 1372:
        raise RuntimeError("独立测试manifest不是1372行")
    vis = [r for r in manifest if r["source"].lower() == "visdrone"]
    if len(vis) != N_VIS:
        raise RuntimeError("独立VisDrone测试图片不是84张")
    group_to_id, vis_ids = {}, set()
    for r in vis:
        iid, group = int(r["image_id"]), r["group_id"].strip()
        if iid in vis_ids or group in group_to_id or not group.startswith("visdrone:"):
            raise RuntimeError(f"VisDrone序列/image重复或group格式错误：{group}")
        if r["sequence_id"].strip() != group.split(":", 1)[1]:
            raise RuntimeError(f"sequence_id与group_id不一致：{group}")
        group_to_id[group] = iid; vis_ids.add(iid)

    dec_rows = readcsv(DECISIONS_PATH)
    if len(dec_rows) != N_VIS:
        raise RuntimeError("selector_decisions.csv不是84行")
    selected = {}
    for r in dec_rows:
        iid, score, threshold = int(r["image_id"]), float(r["selector_score"]), float(r["threshold"])
        flag = bool(int(r["selected_for_tiling"]))
        if iid in selected or iid not in vis_ids or not math.isfinite(score):
            raise RuntimeError("选择器决策记录重复、越界或分数非法")
        if not math.isclose(threshold, THRESHOLD, abs_tol=1e-12) or flag != (score >= THRESHOLD):
            raise RuntimeError(f"选择器阈值/决策不一致：{iid}")
        selected[iid] = flag
    if set(selected) != vis_ids or sum(selected.values()) != N_SELECTED:
        raise RuntimeError("选择器记录集合或60张触发数量不一致")

    gt = loadj(GT_PATH)
    all_ids = {int(x["id"]) for x in gt["images"]}
    manifest_ids = {int(r["image_id"]) for r in manifest}
    if all_ids != manifest_ids or len(gt["categories"]) != 1:
        raise RuntimeError("COCO GT与manifest的image_id或类别结构不一致")
    cat_ids = [int(gt["categories"][0]["id"])]

    preds = {}
    for name, path in PRED_PATHS.items():
        if sha(path) != sm["prediction_sha256"][name]:
            raise RuntimeError(f"{name}预测SHA256与metrics_summary不一致")
        preds[name] = load_pred(path, all_ids, cat_ids[0])
    if sha(GT_PATH) != sm["ground_truth_sha256"]:
        raise RuntimeError("COCO GT SHA256与metrics_summary不一致")

    input_hashes = {
        "manifest": sha(MANIFEST), "filter_summary": sha(FILTER_SUMMARY),
        "metrics_summary": sha(METRICS_PATH), "test_gt": sha(GT_PATH),
        "selector_decisions": sha(DECISIONS_PATH),
        **{f"{k}_predictions": sha(v) for k, v in PRED_PATHS.items()},
    }
    for key, expected_key in (("manifest", "independent_test_manifest"),
                              ("filter_summary", "sequence_filter_summary")):
        expected = sm.get("input_sha256", {}).get(expected_key)
        if expected and input_hashes[key] != expected:
            raise RuntimeError(f"Stage 6.4B输入SHA256不一致：{key}")

    # Verify three repeated predictions are identical to final predictions.
    suffix = {"A_Original": "A", "B_C2_area2048": "B", "C_M3_selective": "C"}
    for rep in (1, 2, 3):
        for name, final_path in PRED_PATHS.items():
            rp = REPEAT / f"repeat_{rep}_predictions_{suffix[name]}.json"
            if not rp.is_file() or sha(rp) != sha(final_path):
                raise RuntimeError(f"repeat {rep} {name}预测文件缺失或SHA不一致")
    base_coco = COCO(str(GT_PATH))
    return vis, sm, gt, preds, selected, group_to_id, vis_ids, cat_ids, base_coco, input_hashes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--checkpoint-every", type=int, default=10)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.n_bootstrap < 100 or args.checkpoint_every < 1:
        ap.error("n-bootstrap至少100，checkpoint-every至少1")
    if args.resume and args.overwrite:
        ap.error("--resume与--overwrite不能同时使用")
    if OUT.exists() and any(OUT.iterdir()):
        if args.overwrite:
            shutil.rmtree(OUT)
        elif not args.resume:
            raise FileExistsError(f"{OUT}已有内容；续跑用--resume，重做用--overwrite")
    OUT.mkdir(parents=True, exist_ok=True)

    (vis, sm, base_gt, indexes, selected, group_to_id, vis_ids, cat_ids,
     base_coco, input_hashes) = validate()
    ids = sorted(vis_ids)
    selected_ids = sorted(i for i in ids if selected[i])
    unselected_ids = sorted(i for i in ids if not selected[i])
    print("=" * 84)
    print("Stage 6.4C | paired VisDrone sequence-cluster bootstrap")
    print("VisDrone independent groups:", len(ids), "| selected:", len(selected_ids),
          "| unselected:", len(unselected_ids))
    print("Bootstrap replicates:", args.n_bootstrap, "| seed:", args.seed)
    print("No YOLO inference/training or M3 tuning.")
    print("=" * 84, flush=True)

    # Reproduce point estimates from Stage 6.4B.
    point_metrics = {}
    for policy, index in indexes.items():
        pred = [p for iid in ids for p in index.get(iid, [])]
        m = evaluate(base_coco, pred, ids, cat_ids)
        expected = sm["metrics"][policy]["visdrone"]
        for metric in METRICS:
            if abs(m[metric] - float(expected[metric])) > 1e-6:
                raise RuntimeError(f"{policy}/{metric}点估计复现失败")
        point_metrics[policy] = m
        print(f"[POINT PASS] {policy}: AP50-95={m['AP50-95']:.6f}, APs={m['APs']:.6f}")

    # Verify hybrid C is exactly B on selected images and A on unselected images.
    for iid in ids:
        ref = "B_C2_area2048" if selected[iid] else "A_Original"
        if signature(indexes["C_M3_selective"].get(iid, [])) != signature(indexes[ref].get(iid, [])):
            raise RuntimeError(f"C预测混合身份检查失败：image_id={iid}, expected={ref}")

    # Descriptive subgroup analysis: are selected images where tiling helps?
    subgroup = {}
    for name, subset in (("selected_60", selected_ids), ("unselected_24", unselected_ids)):
        subgroup[name] = {"images": len(subset)}
        for policy in ("A_Original", "B_C2_area2048"):
            pred = [p for iid in subset for p in indexes[policy].get(iid, [])]
            subgroup[name][policy] = evaluate(base_coco, pred, subset, cat_ids)
        subgroup[name]["delta_B_vs_A"] = {
            m: subgroup[name]["B_C2_area2048"][m] - subgroup[name]["A_Original"][m]
            for m in METRICS
        }
    writej(SUBGROUP, {
        "experiment": "Stage6.4C_Selected_Unselected_Subgroup_Point_Metrics",
        "status": "completed", "visdrone_images": 84,
        "selected_images": 60, "unselected_images": 24,
        "metrics": subgroup,
        "interpretation": "Subgroup APs are descriptive. Primary inference is paired bootstrap over all 84 independent sequences.",
        "input_sha256": input_hashes,
    })

    meta = {"experiment": "Stage6.4C_Paired_Independent_VisDrone_Cluster_Bootstrap",
            "n_bootstrap": args.n_bootstrap, "seed": args.seed,
            "sequence_groups": len(ids), "input_sha256": input_hashes}
    if args.resume:
        if not META.is_file() or not SAMPLES.is_file():
            raise RuntimeError("--resume要求metadata和samples checkpoint均存在")
        old = loadj(META)
        if old != meta:
            raise RuntimeError("续跑参数或输入SHA256与原任务不一致")
        rows = readcsv(SAMPLES)
        completed = {int(r["replicate"]) for r in rows}
        if len(completed) != len(rows) or any(i < 0 or i >= args.n_bootstrap for i in completed):
            raise RuntimeError("已有Bootstrap checkpoint包含重复或越界replicate")
        print(f"[RESUME] {len(completed)}/{args.n_bootstrap}")
    else:
        writej(META, meta); rows = []; completed = set()

    start = time.perf_counter()
    for rep in range(args.n_bootstrap):
        if rep in completed:
            continue
        rng = np.random.default_rng(args.seed + rep)
        sampled = rng.choice(sorted(group_to_id), size=len(ids), replace=True).tolist()
        coco_boot, boot_ids, pairs = make_boot_gt(base_gt, group_to_id, sampled)
        row = {"replicate": rep, "sampled_groups": len(sampled),
               "unique_groups": len(set(sampled)),
               "sampled_group_sha256": hashlib.sha256("\n".join(sampled).encode()).hexdigest()}
        for policy, index in indexes.items():
            remapped = remap_pred(index, pairs)
            m = evaluate(coco_boot, remapped, boot_ids, cat_ids)
            for metric in METRICS:
                row[f"{policy}_{metric}"] = m[metric]
            del remapped, m
        for policy, ref in (("B_C2_area2048", "A_Original"),
                            ("C_M3_selective", "A_Original"),
                            ("C_M3_selective", "B_C2_area2048")):
            for metric in METRICS:
                row[f"delta_{policy}_vs_{ref}_{metric}"] = row[f"{policy}_{metric}"] - row[f"{ref}_{metric}"]
        rows.append(row); completed.add(rep); del coco_boot
        if (rep + 1) % 25 == 0:
            gc.collect()
        if len(completed) % args.checkpoint_every == 0 or rep == args.n_bootstrap - 1:
            rows.sort(key=lambda x: int(x["replicate"]))
            writecsv(SAMPLES, rows)
            print(f"[BOOTSTRAP] {len(completed)}/{args.n_bootstrap} | "
                  f"elapsed={(time.perf_counter()-start)/60:.1f} min", flush=True)

    rows.sort(key=lambda x: int(x["replicate"]))
    writecsv(SAMPLES, rows)
    if len(rows) != args.n_bootstrap or {int(r["replicate"]) for r in rows} != set(range(args.n_bootstrap)):
        raise RuntimeError("Bootstrap replicate不完整")

    metric_dist, delta_dist = {}, {}
    for policy in ("A_Original", "B_C2_area2048", "C_M3_selective"):
        for metric in METRICS:
            key = f"{policy}_{metric}"
            metric_dist[key] = dist([float(r[key]) for r in rows])
    for policy, ref in (("B_C2_area2048", "A_Original"),
                        ("C_M3_selective", "A_Original"),
                        ("C_M3_selective", "B_C2_area2048")):
        for metric in METRICS:
            key = f"delta_{policy}_vs_{ref}_{metric}"
            delta_dist[key] = dist([float(r[key]) for r in rows])

    final = {
        "experiment": "Stage6.4C_Paired_Independent_VisDrone_Cluster_Bootstrap",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "completed",
        "bootstrap": {
            "replicates": args.n_bootstrap, "seed": args.seed,
            "resampling_unit": "independent VisDrone sequence group",
            "sequence_groups": 84, "images_per_sequence": 1,
            "sampling": "84 sequence groups sampled with replacement; paired A/B/C evaluation",
            "confidence_interval": "paired percentile 95% CI, 2.5th and 97.5th percentiles",
        },
        "test_scope": {"visdrone_images": 84, "selected_images": 60,
                       "unselected_images": 24, "selector_threshold": THRESHOLD},
        "point_metrics_reproduced": point_metrics,
        "metric_distributions": metric_dist,
        "paired_delta_distributions": delta_dist,
        "selected_unselected_subgroup_metrics": str(SUBGROUP.relative_to(ROOT)),
        "interpretation_limit": (
            "Inference applies only to 84 independent VisDrone sequences. Each contributes "
            "one image, so sequence resampling is equivalent to resampling the 84 images. "
            "Do not extend these CIs to all 1,372 images: 1,288 UAVDT images come from only "
            "five sequences. Full-test metrics remain descriptive point estimates."
        ),
        "input_sha256": input_hashes,
        "outputs": {"samples_csv": str(SAMPLES.relative_to(ROOT)),
                    "summary_json": str(SUMMARY.relative_to(ROOT)),
                    "subgroup_json": str(SUBGROUP.relative_to(ROOT))},
    }
    writej(SUMMARY, final)
    print("\nStage 6.4C completed.")
    print("Bootstrap samples:", SAMPLES)
    print("Summary:", SUMMARY)
    for key in ("delta_B_C2_area2048_vs_A_Original_AP50-95",
                "delta_B_C2_area2048_vs_A_Original_APs",
                "delta_C_M3_selective_vs_A_Original_AP50-95",
                "delta_C_M3_selective_vs_A_Original_APs",
                "delta_C_M3_selective_vs_B_C2_area2048_AP50-95",
                "delta_C_M3_selective_vs_B_C2_area2048_APs"):
        print(key, delta_dist[key])

if __name__ == "__main__":
    main()
