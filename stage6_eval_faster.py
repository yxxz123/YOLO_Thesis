
import json
import csv
from pathlib import Path

from faster_coco_eval import COCO, COCOeval_faster


ROOT = Path("runs/detect/stage6_selective_oracle")

GT_JSON = Path(
    "runs/detect/source_validation/"
    "Stage5-Baseline-Unified/"
    "annotations_auto_detect_val.json"
)

OUT_JSON = ROOT / "faster_eval_metrics.json"

PRED_FILES = {
    "A_Original": ROOT / "all_original_predictions.json",
    "B_Selective_Tile": ROOT / "selective_tile_predictions.json",
    "C_Selective_Fusion": ROOT / "selective_fusion_predictions.json",
}

STAGE5_REFERENCE = {
    "full": {"AP50-95": 0.4474, "APs": 0.3575},
    "uavdt": {"AP50-95": 0.4747, "APs": 0.4051},
    "visdrone": {"AP50-95": 0.3707, "APs": 0.2297},
}


def load_mapping():

    mapping = {}

    with open(
        ROOT / "image_source_mapping.csv",
        "r",
        encoding="utf-8-sig"
    ) as f:

        for row in csv.DictReader(f):

            mapping[int(row["image_id"])] = (
                row["source"].lower()
            )

    return mapping


def evaluate_one(coco_gt, pred_path, image_ids):

    coco_dt = coco_gt.loadRes(str(pred_path))

    evaluator = COCOeval_faster(
        coco_gt,
        coco_dt,
        iouType="bbox",
        lvis_style=False,
        print_function=print,
    )

    # 与Stage5保持相同的评价器默认参数
    # 不手动修改maxDets
    evaluator.params.imgIds = sorted(image_ids)

    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    stats = evaluator.stats_as_dict

    return {
        "AP50-95": float(stats["AP_all"]),
        "AP50": float(stats["AP_50"]),
        "APs": float(stats["AP_small"]),
        "APm": float(stats["AP_medium"]),
        "APl": float(stats["AP_large"]),
    }


def main():

    for p in [GT_JSON, *PRED_FILES.values()]:

        if not p.exists():
            raise FileNotFoundError(p)

    coco_gt = COCO(str(GT_JSON))

    mapping = load_mapping()

    all_ids = sorted(mapping)

    assert len(all_ids) == 1567

    assert sum(
        s == "uavdt" for s in mapping.values()
    ) == 1320

    assert sum(
        s == "visdrone" for s in mapping.values()
    ) == 247

    scopes = {
        "full": all_ids,
        "uavdt": [
            i for i, s in mapping.items()
            if s == "uavdt"
        ],
        "visdrone": [
            i for i, s in mapping.items()
            if s == "visdrone"
        ],
    }

    results = {}

    for group, pred_path in PRED_FILES.items():

        print("\n", "=" * 60)
        print("实验组：", group)
        print("=" * 60)

        results[group] = {}

        for scope, ids in scopes.items():

            print("\n评价范围：", scope)

            results[group][scope] = evaluate_one(
                coco_gt,
                pred_path,
                ids
            )

    # 计算相对于A组的变化
    delta = {}

    for group in results:

        delta[group] = {}

        for scope in scopes:

            delta[group][scope] = {}

            for metric in results[group][scope]:

                delta[group][scope][metric] = round(
                    results[group][scope][metric]
                    - results["A_Original"][scope][metric],
                    6
                )

    output = {
        "evaluator": "faster_coco_eval",
        "stage5_reference": STAGE5_REFERENCE,
        "results": results,
        "delta_vs_A": delta,
    }

    with open(
        OUT_JSON,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=4,
            ensure_ascii=False,
            allow_nan=False
        )

    print("\n评价完成：", OUT_JSON)


if __name__ == "__main__":
    main()
