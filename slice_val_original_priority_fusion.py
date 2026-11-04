
# ============================================================
# Stage 6.2 / C3
# Original-Priority Residual Tile Fusion
#
# A: 原图预测
# B: 2x2切片预测
# C2: 面积筛选 + 全局NMS
# C3: 原图优先 + 切片残差补充
#
# 不重新训练、不重新推理
# ============================================================

import json
import csv
import hashlib
import math

from pathlib import Path
from collections import defaultdict, Counter

import numpy as np


# ============================================================
# 1. 实验配置
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
    "runs/detect/stage6_original_priority_fusion_iou07"
)

AREA_THRESHOLD = 2048
OVERLAP_IOU = 0.7
MAX_DET = 300

EXPECTED_TOTAL = 1567

EXPECTED_COUNTS = {
    "uavdt": 1320,
    "visdrone": 247
}


# ============================================================
# 2. 基础函数
# ============================================================

def load_json(path):

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):

    with open(path, "w", encoding="utf-8") as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
            allow_nan=False
        )


def sha256_file(path):

    digest = hashlib.sha256()

    with open(path, "rb") as f:

        for block in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            digest.update(block)

    return digest.hexdigest()


def normalize_name(name):

    return str(name).replace("\\", "/")


def prediction_signature(records):

    result = []

    for p in records:

        x, y, w, h = map(float, p["bbox"])

        result.append((
            int(p["category_id"]),
            round(x, 5),
            round(y, 5),
            round(w, 5),
            round(h, 5),
            round(float(p["score"]), 7)
        ))

    return sorted(result)


def top100_retention(original, final):

    original_top = sorted(
        original,
        key=lambda p: float(p["score"]),
        reverse=True
    )[:100]

    final_top = sorted(
        final,
        key=lambda p: float(p["score"]),
        reverse=True
    )[:100]

    remaining = Counter(
        prediction_signature(final_top)
    )

    retained = 0

    for sig in prediction_signature(original_top):

        if remaining[sig] > 0:

            retained += 1
            remaining[sig] -= 1

    return len(original_top), retained


# ============================================================
# 3. 检查输入文件
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
            f"缺少文件：{path}"
        )


config = load_json(CONFIG_JSON)["config"]

expected_config = {
    "imgsz": 640,
    "conf": 0.001,
    "nms_iou": 0.7,
    "max_det": 300,
    "overlap": 0.15,
    "half": False,
    "augment": False
}

for key, expected in expected_config.items():

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
            f"输入实验参数不一致：{key}，"
            f"预期={expected}，实际={actual}"
        )


# ============================================================
# 4. 读取GT及来源映射
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

all_ids = sorted(image_info)

if len(all_ids) != EXPECTED_TOTAL:

    raise RuntimeError(
        f"GT图片数量错误：{len(all_ids)}"
    )


source_by_id = {}

filename_by_id = {}

with open(
    MAPPING_CSV,
    "r",
    encoding="utf-8-sig"
) as f:

    reader = csv.DictReader(f)

    for row in reader:

        image_id = int(row["image_id"])

        if image_id in source_by_id:

            raise RuntimeError(
                f"重复image_id：{image_id}"
            )

        source = row["source"].strip().lower()

        if source not in EXPECTED_COUNTS:

            raise RuntimeError(
                f"未知来源：{source}"
            )

        source_by_id[image_id] = source

        filename_by_id[image_id] = (
            normalize_name(row["file_name"])
        )


if set(source_by_id) != set(all_ids):

    raise RuntimeError(
        "来源映射与GT图片ID不一致。"
    )


for image_id in all_ids:

    if filename_by_id[image_id] != normalize_name(
        image_info[image_id]["file_name"]
    ):

        raise RuntimeError(
            f"图片文件名映射不一致：{image_id}"
        )


source_counts = Counter(source_by_id.values())

for source, expected in EXPECTED_COUNTS.items():

    if source_counts[source] != expected:

        raise RuntimeError(
            f"{source}图片数量异常。"
        )


# ============================================================
# 5. 读取并验证预测结果
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
                f"{group_name}第{index}条记录字段不完整。"
            )

        image_id = int(p["image_id"])
        category_id = int(p["category_id"])

        if image_id not in image_info:

            raise RuntimeError(
                f"未知图片ID：{image_id}"
            )

        if category_id not in category_ids:

            raise RuntimeError(
                f"未知类别ID：{category_id}"
            )

        x, y, w, h = map(float, p["bbox"])
        score = float(p["score"])

        if not all(
            math.isfinite(v)
            for v in [x, y, w, h, score]
        ):

            raise RuntimeError(
                f"{group_name}出现非法数值。"
            )

        if w <= 0 or h <= 0:

            raise RuntimeError(
                f"{group_name}存在无效预测框。"
            )

        if not 0 <= score <= 1:

            raise RuntimeError(
                f"{group_name}存在非法置信度。"
            )

        result[image_id].append(p)

    for image_id, records in result.items():

        if len(records) > MAX_DET:

            raise RuntimeError(
                f"{group_name}图片{image_id}超过300个预测框。"
            )

    return result


pred_a = load_json(A_JSON)
pred_b = load_json(B_JSON)

a_by_id = group_predictions(
    pred_a,
    "A_Original"
)

b_by_id = group_predictions(
    pred_b,
    "B_Selective_Tile"
)


# UAVDT必须与A组原图预测完全一致

for image_id in all_ids:

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
            f"UAVDT图片{image_id}的A/B预测不一致。"
        )


# ============================================================
# 6. IoU计算
# ============================================================

def boxes_xyxy(records):

    if len(records) == 0:

        return np.empty(
            (0, 4),
            dtype=np.float32
        )

    result = []

    for p in records:

        x, y, w, h = map(
            float,
            p["bbox"]
        )

        result.append([
            x,
            y,
            x + w,
            y + h
        ])

    return np.asarray(
        result,
        dtype=np.float32
    )


def pairwise_iou(tile_boxes, original_boxes):

    if len(tile_boxes) == 0 or len(original_boxes) == 0:

        return np.zeros(
            (len(tile_boxes), len(original_boxes)),
            dtype=np.float32
        )

    lt = np.maximum(
        tile_boxes[:, None, :2],
        original_boxes[None, :, :2]
    )

    rb = np.minimum(
        tile_boxes[:, None, 2:],
        original_boxes[None, :, 2:]
    )

    wh = np.maximum(
        rb - lt,
        0
    )

    intersection = (
        wh[:, :, 0] * wh[:, :, 1]
    )

    area_a = (
        (tile_boxes[:, 2] - tile_boxes[:, 0])
        *
        (tile_boxes[:, 3] - tile_boxes[:, 1])
    )

    area_b = (
        (original_boxes[:, 2] - original_boxes[:, 0])
        *
        (original_boxes[:, 3] - original_boxes[:, 1])
    )

    union = (
        area_a[:, None]
        +
        area_b[None, :]
        -
        intersection
    )

    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0
    )


# ============================================================
# 7. C3原图优先残差融合
# ============================================================

predictions = []

audit_rows = []

source_summary = {

    source: {
        "images": 0,
        "original": 0,
        "tile_candidates": 0,
        "tile_area_selected": 0,
        "tile_overlap_removed": 0,
        "tile_remaining": 0,
        "tile_added": 0,
        "capacity_removed": 0,
        "final": 0,
        "original_top100": 0,
        "original_top100_retained": 0
    }

    for source in EXPECTED_COUNTS
}


for image_id in all_ids:

    source = source_by_id[image_id]

    original = sorted(
        a_by_id.get(image_id, []),
        key=lambda p: float(p["score"]),
        reverse=True
    )

    if len(original) > MAX_DET:

        raise RuntimeError(
            f"原图预测超过MAX_DET：{image_id}"
        )


    tile_all = []
    tile_area_selected = []
    tile_remaining = []
    tile_added = []

    free_slots = max(
        0,
        MAX_DET - len(original)
    )


    # --------------------------------------------------------
    # UAVDT：完全保留原图预测
    # --------------------------------------------------------

    if source == "uavdt":

        final = list(original)


    # --------------------------------------------------------
    # VisDrone：面积筛选 + 原图优先
    # --------------------------------------------------------

    elif source == "visdrone":

        tile_all = list(
            b_by_id.get(image_id, [])
        )


        # 1. 面积筛选，严格小于2048

        for p in tile_all:

            _, _, w, h = map(
                float,
                p["bbox"]
            )

            if w * h < AREA_THRESHOLD:

                tile_area_selected.append(p)


        # 2. 剔除与原图预测高度重叠的切片框

        if len(original) > 0 and len(tile_area_selected) > 0:

            original_boxes = boxes_xyxy(original)

            tile_boxes = boxes_xyxy(
                tile_area_selected
            )

            ious = pairwise_iou(
                tile_boxes,
                original_boxes
            )

            tile_classes = np.asarray(
                [
                    int(p["category_id"])
                    for p in tile_area_selected
                ]
            )

            original_classes = np.asarray(
                [
                    int(p["category_id"])
                    for p in original
                ]
            )

            same_class = (
                tile_classes[:, None]
                ==
                original_classes[None, :]
            )

            ious[~same_class] = 0.0

            duplicate_mask = (
                ious >= OVERLAP_IOU
            ).any(axis=1)

            tile_remaining = [
                p
                for p, duplicate in zip(
                    tile_area_selected,
                    duplicate_mask
                )
                if not duplicate
            ]

        else:

            tile_remaining = list(
                tile_area_selected
            )


        # 3. 根据剩余容量补充切片框

        tile_remaining.sort(
            key=lambda p: float(p["score"]),
            reverse=True
        )

        tile_added = tile_remaining[:free_slots]


        # 4. 保留全部原图框，不再执行全局NMS

        final = (
            list(original)
            +
            list(tile_added)
        )

        final.sort(
            key=lambda p: float(p["score"]),
            reverse=True
        )


    else:

        raise RuntimeError(
            f"未知来源：{source}"
        )


    # --------------------------------------------------------
    # 5. 最终结果检查
    # --------------------------------------------------------

    if len(final) > MAX_DET:

        raise RuntimeError(
            f"C3图片{image_id}超过300个预测框。"
        )


    # 验证原图预测框没有从输出文件中消失

    if not (
        Counter(prediction_signature(original))
        <=
        Counter(prediction_signature(final))
    ):

        raise RuntimeError(
            f"C3图片{image_id}丢失原图预测框。"
        )


    # --------------------------------------------------------
    # 6. 原图Top100保留率审计
    # --------------------------------------------------------

    top100_count, top100_retained = (
        top100_retention(original, final)
    )


    predictions.extend(final)


    # --------------------------------------------------------
    # 7. 逐图审计
    # --------------------------------------------------------

    audit = {

        "image_id": image_id,

        "file_name": image_info[image_id]["file_name"],

        "source": source,

        "original_count": len(original),

        "tile_candidate_count": len(tile_all),

        "tile_area_selected_count": len(tile_area_selected),

        "tile_overlap_removed_count": (
            len(tile_area_selected)
            -
            len(tile_remaining)
        ),

        "tile_remaining_count": len(tile_remaining),

        "free_slots": free_slots,

        "tile_added_count": len(tile_added),

        "capacity_removed_count": (
            len(tile_remaining)
            -
            len(tile_added)
        ),

        "final_count": len(final),

        "original_top100_count": top100_count,

        "original_top100_retained": top100_retained,

        "original_top100_retention_rate": (
            round(
                top100_retained / top100_count,
                6
            )
            if top100_count > 0
            else ""
        )

    }

    audit_rows.append(audit)


    # --------------------------------------------------------
    # 8. 来源统计
    # --------------------------------------------------------

    item = source_summary[source]

    item["images"] += 1

    item["original"] += len(original)

    item["tile_candidates"] += len(tile_all)

    item["tile_area_selected"] += len(tile_area_selected)

    item["tile_overlap_removed"] += (
        len(tile_area_selected)
        -
        len(tile_remaining)
    )

    item["tile_remaining"] += len(tile_remaining)

    item["tile_added"] += len(tile_added)

    item["capacity_removed"] += (
        len(tile_remaining)
        -
        len(tile_added)
    )

    item["final"] += len(final)

    item["original_top100"] += top100_count

    item["original_top100_retained"] += top100_retained


# ============================================================
# 8. 保存预测文件
# ============================================================

predictions.sort(
    key=lambda p: (
        int(p["image_id"]),
        -float(p["score"])
    )
)

PRED_OUT = (
    OUT_DIR / "C3_original_priority_predictions.json"
)

save_json(
    predictions,
    PRED_OUT
)


# ============================================================
# 9. 保存逐图审计文件
# ============================================================

AUDIT_OUT = (
    OUT_DIR / "C3_fusion_audit.csv"
)

audit_fields = list(
    audit_rows[0].keys()
)

with open(
    AUDIT_OUT,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=audit_fields
    )

    writer.writeheader()

    writer.writerows(audit_rows)


# ============================================================
# 10. 保存实验摘要
# ============================================================

SUMMARY_OUT = (
    OUT_DIR / "C3_fusion_summary.json"
)

summary = {

    "experiment": "Stage6.2_C3_Original_Priority_Fusion",

    "input": {

        "original_predictions": str(A_JSON),

        "tile_predictions": str(B_JSON),

        "ground_truth": str(GT_JSON),

        "source_mapping": str(MAPPING_CSV),

        "sha256": {

            "original": sha256_file(A_JSON),

            "tile": sha256_file(B_JSON),

            "ground_truth": sha256_file(GT_JSON),

            "source_mapping": sha256_file(MAPPING_CSV)

        }

    },

    "protocol": {

        "area_threshold": AREA_THRESHOLD,

        "area_comparison": "strict_less_than",

        "overlap_iou": OVERLAP_IOU,

        "overlap_comparison": "greater_than_or_equal",

        "max_det": MAX_DET,

        "original_priority": True,

        "global_nms_after_fusion": False,

        "uavdt": "original_only",

        "visdrone": "original_plus_residual_tiles"

    },

    "image_count": {

        "full": EXPECTED_TOTAL,

        **EXPECTED_COUNTS

    },

    "prediction_count": len(predictions),

    "source_summary": source_summary,

    "output": {

        "prediction_file": str(PRED_OUT),

        "prediction_sha256": sha256_file(PRED_OUT),

        "audit_file": str(AUDIT_OUT)

    }

}

save_json(
    summary,
    SUMMARY_OUT
)


# ============================================================
# 11. 最终输出
# ============================================================

print("\n" + "=" * 70)

print("Stage 6.2 / C3 原图优先残差融合完成")

print("=" * 70)

print("预测框总数：", len(predictions))

print("UAVDT：", source_summary["uavdt"])

print("VisDrone：", source_summary["visdrone"])

print("\n预测文件：", PRED_OUT)

print("审计文件：", AUDIT_OUT)

print("摘要文件：", SUMMARY_OUT)

print("\n没有重新训练，也没有重新推理。")
