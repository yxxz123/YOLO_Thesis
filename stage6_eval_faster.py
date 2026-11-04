
# ============================================================
# Unified COCO Evaluation Protocol V1.0
#
# 所有最终模型和推理策略统一使用此评价程序。
#
# Evaluator:
#   faster-coco-eval
#
# Fixed parameters:
#   IoU thresholds = 0.50:0.05:0.95
#   Recall thresholds = 0:0.01:1
#   maxDets = [1,10,100]
#   Area = standard COCO
#
# ============================================================

import json
import csv
import hashlib
import platform

from pathlib import Path
from importlib.metadata import version

import numpy as np

from faster_coco_eval import COCO, COCOeval_faster


# ============================================================
# 1. 固定路径
# ============================================================

ROOT = Path(
    "runs/detect/stage6_selective_oracle"
)

GT_JSON = Path(
    "runs/detect/source_validation/"
    "Stage5-Baseline-Unified/"
    "annotations_auto_detect_val.json"
)

FUSION_DIR = Path(
    "runs/detect/stage6_size_aware_fusion_iou07"
)


OUT_JSON = (
    FUSION_DIR / "faster_eval_metrics.json"
)


# ============================================================
# 2. 固定推理参数
# ============================================================

IMGSZ = 640

CONF = 0.001

NMS_IOU = 0.7

NMS_MAX_DET = 300

OVERLAP = 0.15

HALF = False

AUGMENT = False


# ============================================================
# 3. 固定COCO评价参数
# ============================================================

EVAL_IOU_THRESHOLDS = np.linspace(
    0.50,
    0.95,
    10
)

EVAL_RECALL_THRESHOLDS = np.linspace(
    0.0,
    1.0,
    101
)

EVAL_MAX_DETS = [
    1,
    10,
    100
]

EVAL_AREA_RANGES = [

    [0 ** 2, 1e5 ** 2],

    [0 ** 2, 32 ** 2],

    [32 ** 2, 96 ** 2],

    [96 ** 2, 1e5 ** 2]

]

EVAL_AREA_LABELS = [

    "all",

    "small",

    "medium",

    "large"

]


# ============================================================
# 4. 参与评价的预测文件
# ============================================================

PRED_FILES = {

    "A_Original":
        ROOT / "all_original_predictions.json",

    "B_Selective_Tile":
        ROOT / "selective_tile_predictions.json",

    "C_Selective_Fusion":
        ROOT / "selective_fusion_predictions.json",

    "C2_area512":
        FUSION_DIR / "C2_area512_predictions.json",

    "C2_area1024":
        FUSION_DIR / "C2_area1024_predictions.json",

    "C2_area2048":
        FUSION_DIR / "C2_area2048_predictions.json",

}


# ============================================================
# 5. 文件检查
# ============================================================

FUSION_DIR.mkdir(
    parents=True,
    exist_ok=True
)


for path in [
    GT_JSON,
    *PRED_FILES.values()
]:

    if not path.exists():

        raise FileNotFoundError(
            f"缺少评价文件：{path}"
        )


# ============================================================
# 6. SHA256
# ============================================================

def sha256_file(path):

    digest = hashlib.sha256()

    with open(path, "rb") as f:

        for block in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):

            digest.update(block)

    return digest.hexdigest()


# ============================================================
# 7. 图片来源映射
# ============================================================

mapping_path = (
    ROOT / "image_source_mapping.csv"
)

source_by_id = {}


with open(
    mapping_path,
    "r",
    encoding="utf-8-sig"
) as f:

    for row in csv.DictReader(f):

        image_id = int(row["image_id"])

        source_by_id[image_id] = (
            row["source"].lower()
        )


if len(source_by_id) != 1567:

    raise RuntimeError(
        "图片来源映射数量不正确。"
    )


all_ids = sorted(source_by_id)


uavdt_ids = sorted(

    i for i, source in source_by_id.items()

    if source == "uavdt"

)

visdrone_ids = sorted(

    i for i, source in source_by_id.items()

    if source == "visdrone"

)


assert len(uavdt_ids) == 1320

assert len(visdrone_ids) == 247


SCOPES = {

    "full": all_ids,

    "uavdt": uavdt_ids,

    "visdrone": visdrone_ids

}


# ============================================================
# 8. 初始化GT
# ============================================================

print("\n加载统一COCO GT...")

coco_gt = COCO(
    str(GT_JSON)
)


gt_image_ids = sorted(
    int(x["id"])
    for x in coco_gt.dataset["images"]
)

if gt_image_ids != all_ids:

    raise RuntimeError(
        "GT图片ID与来源映射不一致。"
    )


category_ids = sorted(
    int(x)
    for x in coco_gt.getCatIds()
)


# ============================================================
# 9. 固定评价函数
# ============================================================

def evaluate_one(
    pred_path,
    image_ids
):

    coco_dt = coco_gt.loadRes(
        str(pred_path)
    )


    evaluator = COCOeval_faster(
        coco_gt,
        coco_dt,
        iouType="bbox",
        lvis_style=False,
        print_function=print
    )


    # --------------------------------------------------------
    # 图片ID
    # --------------------------------------------------------

    evaluator.params.imgIds = sorted(
        image_ids
    )


    # --------------------------------------------------------
    # 类别ID
    # --------------------------------------------------------

    evaluator.params.catIds = (
        category_ids
    )


    # --------------------------------------------------------
    # IoU阈值
    # --------------------------------------------------------

    evaluator.params.iouThrs = (
        EVAL_IOU_THRESHOLDS.copy()
    )


    # --------------------------------------------------------
    # Recall阈值
    # --------------------------------------------------------

    evaluator.params.recThrs = (
        EVAL_RECALL_THRESHOLDS.copy()
    )


    # --------------------------------------------------------
    # COCO maxDets
    # --------------------------------------------------------

    evaluator.params.maxDets = (
        EVAL_MAX_DETS.copy()
    )


    # --------------------------------------------------------
    # COCO面积范围
    # --------------------------------------------------------

    evaluator.params.areaRng = [
        x.copy()
        for x in EVAL_AREA_RANGES
    ]


    evaluator.params.areaRngLbl = (
        EVAL_AREA_LABELS.copy()
    )


    evaluator.params.useCats = 1


    # --------------------------------------------------------
    # 正式评价
    # --------------------------------------------------------

    evaluator.evaluate()

    evaluator.accumulate()

    evaluator.summarize()


    stats = evaluator.stats_as_dict


    required_keys = [

        "AP_all",

        "AP_50",

        "AP_small",

        "AP_medium",

        "AP_large"

    ]


    for key in required_keys:

        if key not in stats:

            raise RuntimeError(
                f"评价器缺少指标：{key}"
            )


    return {

        "AP50-95": float(
            stats["AP_all"]
        ),

        "AP50": float(
            stats["AP_50"]
        ),

        "APs": float(
            stats["AP_small"]
        ),

        "APm": float(
            stats["AP_medium"]
        ),

        "APl": float(
            stats["AP_large"]
        )

    }


# ============================================================
# 10. 执行全部实验评价
# ============================================================

results = {}

prediction_hashes = {}


for group_name, pred_path in PRED_FILES.items():

    print("\n" + "=" * 65)

    print("评价实验组：", group_name)

    print("=" * 65)


    results[group_name] = {}

    prediction_hashes[group_name] = (
        sha256_file(pred_path)
    )


    for scope_name, image_ids in SCOPES.items():

        print("\n评价范围：", scope_name)

        metrics = evaluate_one(
            pred_path,
            image_ids
        )

        results[group_name][scope_name] = (
            metrics
        )

        print(metrics)


# ============================================================
# 11. 计算相对于A组的变化
# ============================================================

delta_vs_A = {}


for group_name in results:

    delta_vs_A[group_name] = {}


    for scope_name in SCOPES:

        delta_vs_A[group_name][scope_name] = {}


        for metric in results[group_name][scope_name]:

            delta = (

                results[group_name][scope_name][metric]

                -

                results["A_Original"][scope_name][metric]

            )


            delta_vs_A[group_name][scope_name][metric] = (
                round(float(delta), 6)
            )


# ============================================================
# 12. 评价协议记录
# ============================================================

try:

    evaluator_version = version(
        "faster-coco-eval"
    )

except Exception:

    evaluator_version = "unknown"


protocol = {

    "protocol_name": "YOLO_Thesis_COCO_Evaluation_V1.0",

    "evaluator": "faster_coco_eval",

    "evaluator_version": evaluator_version,

    "task": "bbox",

    "imgsz": IMGSZ,

    "conf": CONF,

    "nms_iou": NMS_IOU,

    "nms_max_det": NMS_MAX_DET,

    "overlap": OVERLAP,

    "half": HALF,

    "augment": AUGMENT,

    "coco_iou_thresholds": (
        EVAL_IOU_THRESHOLDS.tolist()
    ),

    "coco_recall_thresholds": (
        EVAL_RECALL_THRESHOLDS.tolist()
    ),

    "coco_max_dets": EVAL_MAX_DETS,

    "coco_area_ranges": EVAL_AREA_RANGES,

    "coco_area_labels": EVAL_AREA_LABELS,

    "use_categories": 1,

    "image_counts": {

        "full": len(all_ids),

        "uavdt": len(uavdt_ids),

        "visdrone": len(visdrone_ids)

    }

}


# ============================================================
# 13. 保存评价结果
# ============================================================

output = {

    "protocol": protocol,

    "ground_truth": {

        "path": str(GT_JSON),

        "sha256": sha256_file(GT_JSON)

    },

    "prediction_sha256": prediction_hashes,

    "results": results,

    "delta_vs_A": delta_vs_A,

    "environment": {

        "python": platform.python_version(),

        "numpy": np.__version__

    }

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


print("\n" + "=" * 65)

print("统一COCO评价完成")

print("=" * 65)

print("评价协议：", protocol["protocol_name"])

print("评价器版本：", evaluator_version)

print("结果文件：", OUT_JSON)
