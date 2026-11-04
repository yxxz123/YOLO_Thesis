
# ============================================================
# Stage 6.1
# YOLO11s-DySample 尺度感知选择性切片融合
#
# 输入：
#   stage6_selective_oracle/
#       all_original_predictions.json
#       selective_tile_predictions.json
#       image_source_mapping.csv
#       metrics_summary.json
#
# 实验：
#   C2-512
#   C2-1024
#   C2-2048
#
# 规则：
#   UAVDT：只使用原图预测
#   VisDrone：原图预测 + 面积筛选后的切片预测
#   最后执行统一全局NMS
#
# 不重新训练、不重新推理
# ============================================================

import json
import csv
import math
import hashlib
import platform
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

import numpy as np
import torch
import torchvision

from torchvision.ops import batched_nms


# ============================================================
# 1. 固定实验配置
# ============================================================

ROOT = Path(
    "runs/detect/stage6_selective_oracle"
)

GT_JSON = Path(
    "runs/detect/source_validation/"
    "Stage5-Baseline-Unified/"
    "annotations_auto_detect_val.json"
)

A_JSON = ROOT / "all_original_predictions.json"

B_JSON = ROOT / "selective_tile_predictions.json"

MAPPING_CSV = ROOT / "image_source_mapping.csv"

CONFIG_JSON = ROOT / "metrics_summary.json"


OUT_DIR = Path(
    "runs/detect/stage6_size_aware_fusion_iou07"
)


# -------------------- 固定NMS参数 --------------------

NMS_IOU = 0.7

MAX_DET = 300


# -------------------- 固定实验阈值 --------------------

AREA_THRESHOLDS = [
    512,
    1024,
    2048
]


# -------------------- 数据数量 --------------------

EXPECTED_TOTAL = 1567

EXPECTED_SOURCE_COUNTS = {
    "uavdt": 1320,
    "visdrone": 247
}


# ============================================================
# 2. 基础文件检查
# ============================================================

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


REQUIRED_FILES = [
    GT_JSON,
    A_JSON,
    B_JSON,
    MAPPING_CSV,
    CONFIG_JSON
]


for path in REQUIRED_FILES:

    if not path.exists():

        raise FileNotFoundError(
            f"缺少必要文件：{path}"
        )


# ============================================================
# 3. 检查上一阶段实验参数
# ============================================================

def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


stage6_config = load_json(CONFIG_JSON)

config = stage6_config["config"]


EXPECTED_CONFIG = {
    "imgsz": 640,
    "conf": 0.001,
    "nms_iou": 0.7,
    "max_det": 300,
    "overlap": 0.15,
    "half": False,
    "augment": False
}


for key, expected in EXPECTED_CONFIG.items():

    actual = config.get(key)

    if isinstance(expected, float):

        matched = (
            actual is not None
            and math.isclose(
                float(actual),
                expected,
                abs_tol=1e-8
            )
        )

    else:

        matched = actual == expected

    if not matched:

        raise RuntimeError(
            f"实验参数不一致：{key}\n"
            f"预期：{expected}\n"
            f"实际：{actual}"
        )


if not math.isclose(
    NMS_IOU,
    float(config["nms_iou"]),
    abs_tol=1e-8
):

    raise RuntimeError(
        "C2的NMS IoU与Stage6输入预测不一致。"
    )


if MAX_DET != int(config["max_det"]):

    raise RuntimeError(
        "C2的MAX_DET与Stage6输入预测不一致。"
    )


print("=" * 65)
print("Stage 6.1：尺度感知融合实验")
print("=" * 65)

print("NMS IoU：", NMS_IOU)
print("MAX_DET：", MAX_DET)
print("面积阈值：", AREA_THRESHOLDS)
print("输出目录：", OUT_DIR)


# ============================================================
# 4. 读取GT和图片来源映射
# ============================================================

gt = load_json(GT_JSON)

image_info = {

    int(x["id"]): x

    for x in gt["images"]

}

category_ids = {

    int(x["id"])

    for x in gt["categories"]

}


all_image_ids = sorted(image_info)


if len(all_image_ids) != EXPECTED_TOTAL:

    raise RuntimeError(
        f"GT图片数量错误：{len(all_image_ids)}"
    )


source_by_id = {}


with open(
    MAPPING_CSV,
    "r",
    encoding="utf-8-sig"
) as f:

    reader = csv.DictReader(f)

    required_columns = {
        "image_id",
        "source"
    }

    if not required_columns.issubset(
        set(reader.fieldnames or [])
    ):

        raise RuntimeError(
            "image_source_mapping.csv字段不完整。"
        )

    for row in reader:

        image_id = int(row["image_id"])

        source = row["source"].strip().lower()

        if image_id in source_by_id:

            raise RuntimeError(
                f"重复image_id：{image_id}"
            )

        if source not in {
            "uavdt",
            "visdrone"
        }:

            raise RuntimeError(
                f"未知数据来源：{source}"
            )

        source_by_id[image_id] = source


if set(source_by_id) != set(all_image_ids):

    raise RuntimeError(
        "来源映射与GT图片ID不完全一致。"
    )


actual_source_counts = Counter(
    source_by_id.values()
)


for source, expected in EXPECTED_SOURCE_COUNTS.items():

    if actual_source_counts[source] != expected:

        raise RuntimeError(
            f"{source}图片数量不正确："
            f"{actual_source_counts[source]}"
        )


print("\n图片数量检查通过")

print("UAVDT：", actual_source_counts["uavdt"])

print("VisDrone：", actual_source_counts["visdrone"])


# ============================================================
# 5. 读取预测文件
# ============================================================

pred_a = load_json(A_JSON)

pred_b = load_json(B_JSON)


if not isinstance(pred_a, list):

    raise TypeError(
        "A预测文件不是COCO预测列表。"
    )


if not isinstance(pred_b, list):

    raise TypeError(
        "B预测文件不是COCO预测列表。"
    )


# ============================================================
# 6. 预测数据完整性检查
# ============================================================

def group_predictions(predictions, group_name):

    result = defaultdict(list)

    for index, p in enumerate(predictions):

        required = {
            "image_id",
            "category_id",
            "bbox",
            "score"
        }

        if not required.issubset(p):

            raise RuntimeError(
                f"{group_name}第{index}条预测字段缺失。"
            )

        image_id = int(p["image_id"])

        category_id = int(p["category_id"])

        if image_id not in image_info:

            raise RuntimeError(
                f"{group_name}存在未知image_id：{image_id}"
            )

        if category_id not in category_ids:

            raise RuntimeError(
                f"{group_name}存在未知类别：{category_id}"
            )

        bbox = p["bbox"]

        if len(bbox) != 4:

            raise RuntimeError(
                f"{group_name}bbox格式错误。"
            )

        x, y, w, h = map(float, bbox)

        score = float(p["score"])

        if not all(
            math.isfinite(v)
            for v in [x, y, w, h, score]
        ):

            raise RuntimeError(
                f"{group_name}出现NaN或Inf。"
            )

        if w <= 0 or h <= 0:

            raise RuntimeError(
                f"{group_name}出现无效预测框。"
            )

        if not 0 <= score <= 1:

            raise RuntimeError(
                f"{group_name}出现非法置信度。"
            )

        result[image_id].append(p)

    return result


a_by_id = group_predictions(
    pred_a,
    "A_Original"
)

b_by_id = group_predictions(
    pred_b,
    "B_Selective_Tile"
)


# ============================================================
# 7. 检查UAVDT的A、B预测是否完全一致
# ============================================================

def prediction_signature(records):

    result = []

    for p in records:

        x, y, w, h = map(
            float,
            p["bbox"]
        )

        result.append((
            int(p["category_id"]),
            round(x, 5),
            round(y, 5),
            round(w, 5),
            round(h, 5),
            round(float(p["score"]), 7)
        ))

    return sorted(result)


for image_id in all_image_ids:

    if source_by_id[image_id] != "uavdt":

        continue

    sig_a = prediction_signature(
        a_by_id.get(image_id, [])
    )

    sig_b = prediction_signature(
        b_by_id.get(image_id, [])
    )

    if sig_a != sig_b:

        raise RuntimeError(
            f"UAVDT图片{image_id}的A、B预测不一致。\n"
            "请检查Stage6输入预测文件。"
        )


print("\nUAVDT预测一致性检查通过。")


# ============================================================
# 8. COCO bbox格式转换与全局NMS
# ============================================================

def global_nms(
    candidates,
    image_id
):

    """
    输入COCO xywh格式。

    输出COCO xywh格式。

    使用：
        NMS IoU = 0.7
        max_det = 300

    类别之间独立执行NMS。
    """

    if len(candidates) == 0:

        return [], 0


    info = image_info[image_id]

    width = int(info["width"])

    height = int(info["height"])


    boxes = []

    scores = []

    classes = []

    valid_records = []


    for p in candidates:

        x, y, w, h = map(
            float,
            p["bbox"]
        )

        x1 = max(
            0.0,
            min(float(width), x)
        )

        y1 = max(
            0.0,
            min(float(height), y)
        )

        x2 = max(
            0.0,
            min(float(width), x + w)
        )

        y2 = max(
            0.0,
            min(float(height), y + h)
        )

        if x2 <= x1 or y2 <= y1:

            continue

        boxes.append([
            x1,
            y1,
            x2,
            y2
        ])

        scores.append(
            float(p["score"])
        )

        classes.append(
            int(p["category_id"])
        )

        valid_records.append(p)


    valid_count = len(boxes)


    if valid_count == 0:

        return [], 0


    boxes_tensor = torch.tensor(
        boxes,
        dtype=torch.float32
    )

    scores_tensor = torch.tensor(
        scores,
        dtype=torch.float32
    )

    classes_tensor = torch.tensor(
        classes,
        dtype=torch.int64
    )


    keep = batched_nms(
        boxes_tensor,
        scores_tensor,
        classes_tensor,
        NMS_IOU
    )


    keep = keep[:MAX_DET].cpu().tolist()


    output = []


    for index in keep:

        x1, y1, x2, y2 = (
            boxes_tensor[index].tolist()
        )

        p = valid_records[index]

        output.append({

            "image_id": int(image_id),

            "category_id": int(
                p["category_id"]
            ),

            "bbox": [
                float(x1),
                float(y1),
                float(x2 - x1),
                float(y2 - y1)
            ],

            "score": float(
                p["score"]
            )

        })


    return output, valid_count


# ============================================================
# 9. 尺度感知融合
# ============================================================

def fuse_one_threshold(
    area_threshold
):

    predictions = []

    audit_rows = []

    source_summary = {

        "uavdt": {
            "images": 0,
            "original": 0,
            "tile_candidates": 0,
            "tile_selected": 0,
            "final": 0
        },

        "visdrone": {
            "images": 0,
            "original": 0,
            "tile_candidates": 0,
            "tile_selected": 0,
            "final": 0
        }

    }


    for image_id in all_image_ids:

        info = image_info[image_id]

        source = source_by_id[image_id]


        original = list(
            a_by_id.get(image_id, [])
        )


        tile_all = []

        tile_selected = []


        # ----------------------------------------------------
        # UAVDT：只使用原图结果
        # ----------------------------------------------------

        if source == "uavdt":

            final = original

            valid_count = len(final)


        # ----------------------------------------------------
        # VisDrone：原图+筛选后的切片结果
        # ----------------------------------------------------

        elif source == "visdrone":

            tile_all = list(
                b_by_id.get(image_id, [])
            )


            for p in tile_all:

                x, y, w, h = map(
                    float,
                    p["bbox"]
                )

                area = w * h


                # 严格小于面积阈值
                if area < area_threshold:

                    tile_selected.append(p)


            # 没有新增切片框时，直接保留原图预测
            if len(tile_selected) == 0:

                final = original

                valid_count = len(final)


            else:

                combined = (
                    original
                    +
                    tile_selected
                )

                final, valid_count = global_nms(
                    combined,
                    image_id
                )


        else:

            raise RuntimeError(
                f"未知来源：{source}"
            )


        predictions.extend(final)


        # ----------------------------------------------------
        # 逐图审计
        # ----------------------------------------------------

        audit_rows.append({

            "area_threshold": area_threshold,

            "image_id": image_id,

            "file_name": info["file_name"],

            "source": source,

            "original_count": len(original),

            "tile_candidate_count": len(tile_all),

            "tile_selected_count": len(tile_selected),

            "combined_count": (
                len(original)
                +
                len(tile_selected)
            ),

            "valid_candidate_count": valid_count,

            "final_count": len(final),

            "nms_removed_count": max(
                0,
                valid_count - len(final)
            )

        })


        # ----------------------------------------------------
        # 来源统计
        # ----------------------------------------------------

        item = source_summary[source]

        item["images"] += 1

        item["original"] += len(original)

        item["tile_candidates"] += len(tile_all)

        item["tile_selected"] += len(tile_selected)

        item["final"] += len(final)


    return (
        predictions,
        audit_rows,
        source_summary
    )


# ============================================================
# 10. 文件保存
# ============================================================

def save_json(data, path):

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
            allow_nan=False
        )


# ============================================================
# 11. 生成三个阈值实验
# ============================================================

all_audit = []

summary_results = {}


for threshold in AREA_THRESHOLDS:

    print("\n" + "=" * 65)

    print(
        f"开始实验：C2-area{threshold}"
    )

    print("=" * 65)


    predictions, audit, source_summary = (
        fuse_one_threshold(threshold)
    )


    # 按image_id整理输出
    predictions.sort(
        key=lambda x: (
            int(x["image_id"]),
            -float(x["score"])
        )
    )


    output_file = (
        OUT_DIR
        /
        f"C2_area{threshold}_predictions.json"
    )


    save_json(
        predictions,
        output_file
    )


    all_audit.extend(audit)


    summary_results[str(threshold)] = {

        "prediction_count": len(predictions),

        "source_summary": source_summary,

        "prediction_file": str(output_file)

    }


    print("阈值：", threshold)

    print("预测框数量：", len(predictions))

    print("UAVDT：", source_summary["uavdt"])

    print("VisDrone：", source_summary["visdrone"])

    print("保存：", output_file)


# ============================================================
# 12. 保存逐图审计CSV
# ============================================================

audit_path = (
    OUT_DIR / "fusion_audit.csv"
)


audit_fields = [

    "area_threshold",

    "image_id",

    "file_name",

    "source",

    "original_count",

    "tile_candidate_count",

    "tile_selected_count",

    "combined_count",

    "valid_candidate_count",

    "final_count",

    "nms_removed_count"

]


with open(
    audit_path,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=audit_fields
    )

    writer.writeheader()

    writer.writerows(all_audit)


# ============================================================
# 13. 记录输入文件SHA256
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
# 14. 保存实验配置与结果摘要
# ============================================================

summary = {

    "experiment": "Stage6.1_Size_Aware_Fusion",

    "timestamp": datetime.now().isoformat(
        timespec="seconds"
    ),

    "input": {

        "original_predictions": str(A_JSON),

        "tile_predictions": str(B_JSON),

        "ground_truth": str(GT_JSON),

        "source_mapping": str(MAPPING_CSV),

        "input_sha256": {

            "original": sha256_file(A_JSON),

            "tile": sha256_file(B_JSON),

            "ground_truth": sha256_file(GT_JSON)

        }

    },

    "inference_protocol": {

        "imgsz": 640,

        "conf": 0.001,

        "nms_iou": NMS_IOU,

        "max_det": MAX_DET,

        "overlap": 0.15,

        "rect": False,

        "half": False,

        "augment": False

    },

    "fusion_protocol": {

        "uavdt": "original_only",

        "visdrone": "original_plus_filtered_tiles",

        "area_unit": "original_image_pixel_squared",

        "area_comparison": "strict_less_than",

        "global_nms_iou": NMS_IOU,

        "global_max_det": MAX_DET

    },

    "image_count": {

        "full": EXPECTED_TOTAL,

        "uavdt": EXPECTED_SOURCE_COUNTS["uavdt"],

        "visdrone": EXPECTED_SOURCE_COUNTS["visdrone"]

    },

    "thresholds": summary_results,

    "environment": {

        "python": platform.python_version(),

        "torch": torch.__version__,

        "torchvision": torchvision.__version__,

        "numpy": np.__version__

    }

}


summary_path = (
    OUT_DIR / "fusion_summary.json"
)


save_json(
    summary,
    summary_path
)


# ============================================================
# 15. 最终检查
# ============================================================

for threshold in AREA_THRESHOLDS:

    path = (
        OUT_DIR
        /
        f"C2_area{threshold}_predictions.json"
    )

    data = load_json(path)

    image_ids_in_output = {
        int(x["image_id"])
        for x in data
    }

    if not image_ids_in_output.issubset(
        set(all_image_ids)
    ):

        raise RuntimeError(
            f"C2-{threshold}存在未知图片ID。"
        )


print("\n" + "=" * 65)

print("Stage 6.1 尺度感知融合全部完成")

print("=" * 65)

print("\n输出目录：", OUT_DIR)

print("\n生成文件：")

for threshold in AREA_THRESHOLDS:

    print(
        f"C2_area{threshold}_predictions.json"
    )

print("fusion_audit.csv")

print("fusion_summary.json")

print("\n本次没有重新训练模型，也没有重新推理。")
