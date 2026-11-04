# ============================================================
# Stage 6.3B
# Per-image Slice Gain Analysis
#
# 目标：
#   利用 A_Original 与 C2_area2048 的已有预测，
#   计算每张图片的小目标检测收益。
#
# 特别注意：
#   1. 不重新推理
#   2. 不重新训练
#   3. 每张图只使用Top100预测，与COCO maxDets=100一致
#   4. selector标签只对VisDrone有效
#      UAVDT当前没有IoU=0.7切片预测，不作为收益标签
# ============================================================

import csv
import json
import math
import hashlib

from pathlib import Path
from collections import defaultdict, Counter

import numpy as np


# ============================================================
# 1. 路径
# ============================================================

SCENE_DIR = Path(
    "runs/detect/stage6_scene_selector"
)

ORACLE_DIR = Path(
    "runs/detect/stage6_selective_oracle"
)

C2_DIR = Path(
    "runs/detect/stage6_size_aware_fusion_iou07"
)

GT_JSON = Path(
    "runs/detect/source_validation/"
    "Stage5-Baseline-Unified/"
    "annotations_auto_detect_val.json"
)

FEATURE_CSV = (
    SCENE_DIR / "corrected_scene_features.csv"
)

A_JSON = (
    ORACLE_DIR / "all_original_predictions.json"
)

C2_JSON = (
    C2_DIR / "C2_area2048_predictions.json"
)

OUT_CSV = (
    SCENE_DIR / "selector_gain_analysis.csv"
)

OUT_JSON = (
    SCENE_DIR / "selector_gain_summary.json"
)


# ============================================================
# 2. 固定分析参数
# ============================================================

TOP_K = 100

SMALL_AREA_THRESHOLD = 32 ** 2

IOU_THRESHOLDS = [
    0.50,
    0.75
]

EXPECTED_TOTAL = 1567

EXPECTED_COUNTS = {
    "uavdt": 1320,
    "visdrone": 247
}


# ============================================================
# 3. 基础函数
# ============================================================

def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def sha256_file(path):

    digest = hashlib.sha256()

    with open(path, "rb") as f:

        for block in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):

            digest.update(block)

    return digest.hexdigest()


def safe_float(value):

    if value in {
        "",
        None
    }:
        return None

    value = float(value)

    if not math.isfinite(value):
        return None

    return value


def bbox_xywh_to_xyxy(bbox):

    x, y, w, h = map(
        float,
        bbox
    )

    return np.asarray([
        x,
        y,
        x + w,
        y + h
    ], dtype=np.float64)


def iou_one_to_many(
    box,
    boxes
):

    """
    box:
        shape (4,)

    boxes:
        shape (N, 4)
    """

    if len(boxes) == 0:

        return np.empty(
            (0,),
            dtype=np.float64
        )

    xx1 = np.maximum(
        box[0],
        boxes[:, 0]
    )

    yy1 = np.maximum(
        box[1],
        boxes[:, 1]
    )

    xx2 = np.minimum(
        box[2],
        boxes[:, 2]
    )

    yy2 = np.minimum(
        box[3],
        boxes[:, 3]
    )

    inter_w = np.maximum(
        0.0,
        xx2 - xx1
    )

    inter_h = np.maximum(
        0.0,
        yy2 - yy1
    )

    intersection = (
        inter_w * inter_h
    )

    area_a = max(
        0.0,
        (box[2] - box[0])
        *
        (box[3] - box[1])
    )

    area_b = np.maximum(
        0.0,
        (boxes[:, 2] - boxes[:, 0])
        *
        (boxes[:, 3] - boxes[:, 1])
    )

    union = (
        area_a
        +
        area_b
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
# 4. 小目标匹配
# ============================================================

def match_small_gt(
    predictions,
    small_gt,
    iou_threshold
):

    """
    使用置信度降序预测框进行贪心匹配。

    只评价small GT。

    prediction:
        COCO prediction dict

    small_gt:
        COCO annotation dict

    返回匹配到的小目标GT数量。
    """

    if len(small_gt) == 0:
        return 0

    if len(predictions) == 0:
        return 0

    predictions = sorted(
        predictions,
        key=lambda p: float(
            p["score"]
        ),
        reverse=True
    )[:TOP_K]

    gt_boxes = np.asarray([
        bbox_xywh_to_xyxy(
            ann["bbox"]
        )
        for ann in small_gt
    ])

    gt_classes = np.asarray([
        int(ann["category_id"])
        for ann in small_gt
    ])

    matched_gt = np.zeros(
        len(small_gt),
        dtype=bool
    )

    matched_count = 0

    for pred in predictions:

        pred_class = int(
            pred["category_id"]
        )

        pred_box = bbox_xywh_to_xyxy(
            pred["bbox"]
        )

        candidate_mask = (
            (~matched_gt)
            &
            (gt_classes == pred_class)
        )

        candidate_indices = np.where(
            candidate_mask
        )[0]

        if len(candidate_indices) == 0:
            continue

        candidate_boxes = gt_boxes[
            candidate_indices
        ]

        ious = iou_one_to_many(
            pred_box,
            candidate_boxes
        )

        if len(ious) == 0:
            continue

        best_local = int(
            np.argmax(ious)
        )

        best_iou = float(
            ious[best_local]
        )

        if best_iou >= iou_threshold:

            best_gt_index = int(
                candidate_indices[
                    best_local
                ]
            )

            matched_gt[
                best_gt_index
            ] = True

            matched_count += 1

    return matched_count


# ============================================================
# 5. 读取特征
# ============================================================

for path in [
    FEATURE_CSV,
    A_JSON,
    C2_JSON,
    GT_JSON
]:

    if not path.exists():

        raise FileNotFoundError(
            f"缺少输入文件：{path}"
        )


feature_by_id = {}

with open(
    FEATURE_CSV,
    "r",
    encoding="utf-8-sig"
) as f:

    reader = csv.DictReader(f)

    for row in reader:

        image_id = int(
            row["image_id"]
        )

        feature_by_id[
            image_id
        ] = {

            "image_id": image_id,

            "file_name":
                row["file_name"],

            "source":
                row["source"],

            "sequence_id":
                row["sequence_id"],

            "width":
                int(row["width"]),

            "height":
                int(row["height"]),

            "letterbox_scale":
                float(
                    row[
                        "letterbox_scale"
                    ]
                ),

            "n_det_conf025":
                int(
                    row[
                        "n_det_conf025"
                    ]
                ),

            "median_box_area_640":
                safe_float(
                    row[
                        "median_box_area_640"
                    ]
                ),

            "tiny_ratio":
                float(
                    row["tiny_ratio"]
                ),

            "tiny_count":
                int(
                    row["tiny_count"]
                )
        }


if len(feature_by_id) != EXPECTED_TOTAL:

    raise RuntimeError(
        f"特征图片数量错误："
        f"{len(feature_by_id)}"
    )


# ============================================================
# 6. 读取GT
# ============================================================

gt = load_json(
    GT_JSON
)

gt_image_ids = {
    int(x["id"])
    for x in gt["images"]
}

if gt_image_ids != set(
    feature_by_id
):

    raise RuntimeError(
        "GT与特征image_id不一致。"
    )


small_gt_by_id = defaultdict(
    list
)

all_gt_count_by_id = Counter()


for ann in gt["annotations"]:

    image_id = int(
        ann["image_id"]
    )

    # crowd不作为本次简单收益标签对象
    if int(
        ann.get(
            "iscrowd",
            0
        )
    ) == 1:

        continue

    all_gt_count_by_id[
        image_id
    ] += 1

    if "area" in ann:

        area = float(
            ann["area"]
        )

    else:

        _, _, w, h = map(
            float,
            ann["bbox"]
        )

        area = w * h

    if area < SMALL_AREA_THRESHOLD:

        small_gt_by_id[
            image_id
        ].append(ann)


# ============================================================
# 7. 读取预测并按图片分组
# ============================================================

def group_predictions(
    path
):

    predictions = load_json(
        path
    )

    result = defaultdict(
        list
    )

    for p in predictions:

        image_id = int(
            p["image_id"]
        )

        if image_id not in feature_by_id:

            raise RuntimeError(
                f"预测出现未知image_id："
                f"{image_id}"
            )

        score = float(
            p["score"]
        )

        if not math.isfinite(score):

            raise RuntimeError(
                "预测置信度出现非法值。"
            )

        result[
            image_id
        ].append(p)

    # 固定按照score排序
    for image_id in result:

        result[
            image_id
        ].sort(
            key=lambda p: float(
                p["score"]
            ),
            reverse=True
        )

    return result


print(
    "正在读取A组预测..."
)

a_by_id = group_predictions(
    A_JSON
)

print(
    "正在读取C2-2048预测..."
)

c2_by_id = group_predictions(
    C2_JSON
)


# ============================================================
# 8. UAVDT一致性检查
# ============================================================

def prediction_signature(
    records
):

    records = sorted(
        records,
        key=lambda p: float(
            p["score"]
        ),
        reverse=True
    )[:TOP_K]

    signature = []

    for p in records:

        x, y, w, h = map(
            float,
            p["bbox"]
        )

        signature.append((
            int(
                p["category_id"]
            ),
            round(
                float(p["score"]),
                7
            ),
            round(x, 4),
            round(y, 4),
            round(w, 4),
            round(h, 4)
        ))

    return signature


uavdt_mismatch = 0

for image_id, feature in (
    feature_by_id.items()
):

    if feature["source"] != "uavdt":
        continue

    sig_a = prediction_signature(
        a_by_id.get(
            image_id,
            []
        )
    )

    sig_c2 = prediction_signature(
        c2_by_id.get(
            image_id,
            []
        )
    )

    if sig_a != sig_c2:

        uavdt_mismatch += 1


if uavdt_mismatch != 0:

    raise RuntimeError(
        "C2-2048中的UAVDT预测"
        "不再与A组一致："
        f"{uavdt_mismatch}张"
    )


print(
    "UAVDT A/C2一致性检查通过。"
)


# ============================================================
# 9. 逐图计算收益
# ============================================================

rows_out = []


for image_id in sorted(
    feature_by_id
):

    feature = feature_by_id[
        image_id
    ]

    source = feature[
        "source"
    ]

    small_gt = small_gt_by_id.get(
        image_id,
        []
    )

    small_gt_count = len(
        small_gt
    )

    pred_a = a_by_id.get(
        image_id,
        []
    )

    pred_c2 = c2_by_id.get(
        image_id,
        []
    )


    # --------------------------------------------------------
    # IoU=0.50
    # --------------------------------------------------------

    a_match50 = match_small_gt(
        pred_a,
        small_gt,
        0.50
    )

    c2_match50 = match_small_gt(
        pred_c2,
        small_gt,
        0.50
    )


    # --------------------------------------------------------
    # IoU=0.75
    # --------------------------------------------------------

    a_match75 = match_small_gt(
        pred_a,
        small_gt,
        0.75
    )

    c2_match75 = match_small_gt(
        pred_c2,
        small_gt,
        0.75
    )


    delta_match50 = (
        c2_match50
        -
        a_match50
    )

    delta_match75 = (
        c2_match75
        -
        a_match75
    )

    gain_score = (
        delta_match50
        +
        delta_match75
    )


    if small_gt_count > 0:

        a_recall50 = (
            a_match50
            /
            small_gt_count
        )

        c2_recall50 = (
            c2_match50
            /
            small_gt_count
        )

        a_recall75 = (
            a_match75
            /
            small_gt_count
        )

        c2_recall75 = (
            c2_match75
            /
            small_gt_count
        )

        delta_recall50 = (
            c2_recall50
            -
            a_recall50
        )

        delta_recall75 = (
            c2_recall75
            -
            a_recall75
        )

    else:

        a_recall50 = ""
        c2_recall50 = ""

        a_recall75 = ""
        c2_recall75 = ""

        delta_recall50 = ""
        delta_recall75 = ""


    # --------------------------------------------------------
    # selector标签
    # --------------------------------------------------------

    if source != "visdrone":

        # UAVDT当前没有正式IoU=0.7切片收益
        selector_label = (
            "not_measured"
        )

        label_available = 0

    elif small_gt_count == 0:

        selector_label = (
            "no_small_gt"
        )

        label_available = 0

    else:

        label_available = 1

        if gain_score > 0:

            selector_label = (
                "beneficial"
            )

        elif gain_score < 0:

            selector_label = (
                "harmful"
            )

        else:

            selector_label = (
                "neutral"
            )


    row = {

        **feature,

        "all_gt_count":
            all_gt_count_by_id[
                image_id
            ],

        "small_gt_count":
            small_gt_count,

        "a_top100_count":
            min(
                len(pred_a),
                TOP_K
            ),

        "c2_top100_count":
            min(
                len(pred_c2),
                TOP_K
            ),

        "a_small_match50":
            a_match50,

        "c2_small_match50":
            c2_match50,

        "delta_small_match50":
            delta_match50,

        "a_small_recall50":
            (
                round(
                    a_recall50,
                    6
                )
                if a_recall50 != ""
                else ""
            ),

        "c2_small_recall50":
            (
                round(
                    c2_recall50,
                    6
                )
                if c2_recall50 != ""
                else ""
            ),

        "delta_small_recall50":
            (
                round(
                    delta_recall50,
                    6
                )
                if delta_recall50 != ""
                else ""
            ),

        "a_small_match75":
            a_match75,

        "c2_small_match75":
            c2_match75,

        "delta_small_match75":
            delta_match75,

        "a_small_recall75":
            (
                round(
                    a_recall75,
                    6
                )
                if a_recall75 != ""
                else ""
            ),

        "c2_small_recall75":
            (
                round(
                    c2_recall75,
                    6
                )
                if c2_recall75 != ""
                else ""
            ),

        "delta_small_recall75":
            (
                round(
                    delta_recall75,
                    6
                )
                if delta_recall75 != ""
                else ""
            ),

        "gain_score":
            gain_score,

        "label_available":
            label_available,

        "selector_label":
            selector_label
    }

    rows_out.append(row)


# ============================================================
# 10. 保存逐图结果
# ============================================================

with open(
    OUT_CSV,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            rows_out[0].keys()
        )
    )

    writer.writeheader()

    writer.writerows(
        rows_out
    )


# ============================================================
# 11. 汇总统计
# ============================================================

def correlation(
    rows,
    feature_name,
    target_name
):

    pairs = []

    for row in rows:

        x = row.get(
            feature_name
        )

        y = row.get(
            target_name
        )

        if x in {
            "",
            None
        }:

            continue

        if y in {
            "",
            None
        }:

            continue

        x = float(x)
        y = float(y)

        if (
            math.isfinite(x)
            and math.isfinite(y)
        ):

            pairs.append(
                (x, y)
            )

    if len(pairs) < 3:

        return None

    xs = np.asarray([
        x for x, _ in pairs
    ])

    ys = np.asarray([
        y for _, y in pairs
    ])

    if (
        np.std(xs) == 0
        or
        np.std(ys) == 0
    ):

        return None

    return round(
        float(
            np.corrcoef(
                xs,
                ys
            )[0, 1]
        ),
        6
    )


vis_rows = [
    row
    for row in rows_out
    if row["source"] == "visdrone"
]

vis_labeled = [
    row
    for row in vis_rows
    if row[
        "label_available"
    ] == 1
]


label_counts = Counter(
    row["selector_label"]
    for row in vis_rows
)


total_small_gt = sum(
    row["small_gt_count"]
    for row in vis_labeled
)

a_total_match50 = sum(
    row["a_small_match50"]
    for row in vis_labeled
)

c2_total_match50 = sum(
    row["c2_small_match50"]
    for row in vis_labeled
)

a_total_match75 = sum(
    row["a_small_match75"]
    for row in vis_labeled
)

c2_total_match75 = sum(
    row["c2_small_match75"]
    for row in vis_labeled
)


summary = {

    "experiment":
        "Stage6.3B_PerImage_Slice_Gain",

    "protocol": {

        "top_k":
            TOP_K,

        "small_area_threshold":
            SMALL_AREA_THRESHOLD,

        "iou_thresholds":
            IOU_THRESHOLDS,

        "match_method":
            (
                "score-descending greedy "
                "class-aware matching"
            ),

        "selector_target":
            (
                "C2_area2048 gain "
                "relative to A_Original"
            ),

        "uavdt_label":
            (
                "not_measured because "
                "C2 uses original prediction "
                "for UAVDT"
            )
    },

    "input": {

        "features": str(
            FEATURE_CSV
        ),

        "features_sha256":
            sha256_file(
                FEATURE_CSV
            ),

        "original_predictions":
            str(A_JSON),

        "original_sha256":
            sha256_file(
                A_JSON
            ),

        "c2_predictions":
            str(C2_JSON),

        "c2_sha256":
            sha256_file(
                C2_JSON
            ),

        "ground_truth":
            str(GT_JSON),

        "gt_sha256":
            sha256_file(
                GT_JSON
            )
    },

    "image_count": {

        "total":
            len(rows_out),

        "uavdt":
            sum(
                row["source"]
                == "uavdt"
                for row in rows_out
            ),

        "visdrone":
            len(vis_rows),

        "visdrone_labeled":
            len(vis_labeled)
    },

    "visdrone_label_counts":
        dict(label_counts),

    "visdrone_aggregate_small_recall": {

        "small_gt_count":
            total_small_gt,

        "A_recall50":
            (
                round(
                    a_total_match50
                    /
                    total_small_gt,
                    6
                )
                if total_small_gt
                else None
            ),

        "C2_recall50":
            (
                round(
                    c2_total_match50
                    /
                    total_small_gt,
                    6
                )
                if total_small_gt
                else None
            ),

        "delta_recall50":
            (
                round(
                    (
                        c2_total_match50
                        -
                        a_total_match50
                    )
                    /
                    total_small_gt,
                    6
                )
                if total_small_gt
                else None
            ),

        "A_recall75":
            (
                round(
                    a_total_match75
                    /
                    total_small_gt,
                    6
                )
                if total_small_gt
                else None
            ),

        "C2_recall75":
            (
                round(
                    c2_total_match75
                    /
                    total_small_gt,
                    6
                )
                if total_small_gt
                else None
            ),

        "delta_recall75":
            (
                round(
                    (
                        c2_total_match75
                        -
                        a_total_match75
                    )
                    /
                    total_small_gt,
                    6
                )
                if total_small_gt
                else None
            )
    },

    "feature_gain_correlation": {

        "n_det_conf025":
            correlation(
                vis_labeled,
                "n_det_conf025",
                "gain_score"
            ),

        "median_box_area_640":
            correlation(
                vis_labeled,
                "median_box_area_640",
                "gain_score"
            ),

        "tiny_ratio":
            correlation(
                vis_labeled,
                "tiny_ratio",
                "gain_score"
            )
    },

    "sequence_count": {

        "visdrone":
            len(set(
                row["sequence_id"]
                for row in vis_rows
            ))
    },

    "output": {

        "analysis_csv":
            str(OUT_CSV),

        "summary_json":
            str(OUT_JSON)
    }
}


with open(
    OUT_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# 12. 输出
# ============================================================

print("\n" + "=" * 70)

print(
    "Stage 6.3B 切片收益分析完成"
)

print("=" * 70)

print(
    "图片总数：",
    len(rows_out)
)

print(
    "VisDrone图片：",
    len(vis_rows)
)

print(
    "VisDrone可用于收益标签：",
    len(vis_labeled)
)

print(
    "\nVisDrone标签分布："
)

print(
    json.dumps(
        dict(label_counts),
        indent=2,
        ensure_ascii=False
    )
)

print(
    "\nVisDrone小目标累计Recall："
)

print(
    json.dumps(
        summary[
            "visdrone_aggregate_small_recall"
        ],
        indent=2,
        ensure_ascii=False
    )
)

print(
    "\n特征与gain_score相关性："
)

print(
    json.dumps(
        summary[
            "feature_gain_correlation"
        ],
        indent=2,
        ensure_ascii=False
    )
)

print("\n输出文件：")

print(OUT_CSV)

print(OUT_JSON)

print(
    "\n没有重新训练，没有重新推理。"
)
