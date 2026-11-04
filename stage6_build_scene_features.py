
# ============================================================
# Stage 6.3A
# Rebuild Corrected Scene Features
#
# 数据来源：
# 1. Stage6.0原图预测
# 2. 统一COCO GT
# 3. 图片来源映射
#
# 不重新推理，不重新训练
# ============================================================

import csv
import json
import hashlib
import math
import posixpath

from pathlib import Path
from collections import defaultdict, Counter

import numpy as np


# ============================================================
# 1. 路径及参数
# ============================================================

ROOT = Path(
    "runs/detect/stage6_selective_oracle"
)

PRED_JSON = (
    ROOT / "all_original_predictions.json"
)

GT_JSON = Path(
    "runs/detect/source_validation/"
    "Stage5-Baseline-Unified/"
    "annotations_auto_detect_val.json"
)

MAPPING_CSV = (
    ROOT / "image_source_mapping.csv"
)

OUT_DIR = Path(
    "runs/detect/stage6_scene_selector"
)

OUT_CSV = (
    OUT_DIR / "corrected_scene_features.csv"
)

OUT_SUMMARY = (
    OUT_DIR / "feature_summary.json"
)


# ============================================================
# 2. 固定特征参数
# ============================================================

FEATURE_CONF = 0.25

FEATURE_IMGSZ = 640

TINY_AREA_THRESHOLD = 32 ** 2

EXPECTED_COUNTS = {
    "uavdt": 1320,
    "visdrone": 247
}

EXPECTED_TOTAL = 1567


# ============================================================
# 3. 基础函数
# ============================================================

def load_json(path):

    with open(path, "r", encoding="utf-8") as f:
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


def normalize_name(name):

    name = str(name).replace("\\", "/")

    return posixpath.normpath(name)


def sequence_id_from_name(filename):

    stem = Path(filename).stem

    parts = stem.rsplit("_", 1)

    if len(parts) == 2:
        return parts[0]

    return stem


def describe(values):

    values = np.asarray(
        values,
        dtype=np.float64
    )

    if len(values) == 0:
        return {"count": 0}

    return {
        "count": int(len(values)),
        "mean": round(float(np.mean(values)), 6),
        "std": round(float(np.std(values)), 6),
        "min": round(float(np.min(values)), 6),
        "p25": round(float(np.percentile(values, 25)), 6),
        "median": round(float(np.median(values)), 6),
        "p75": round(float(np.percentile(values, 75)), 6),
        "max": round(float(np.max(values)), 6)
    }


# ============================================================
# 4. 检查输入文件
# ============================================================

for path in [
    PRED_JSON,
    GT_JSON,
    MAPPING_CSV
]:

    if not path.exists():
        raise FileNotFoundError(path)


OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 5. 读取GT图片信息
# ============================================================

gt = load_json(GT_JSON)

image_info = {
    int(x["id"]): x
    for x in gt["images"]
}

if len(image_info) != EXPECTED_TOTAL:

    raise RuntimeError(
        f"GT图片数量错误：{len(image_info)}"
    )


# ============================================================
# 6. 读取来源映射
# ============================================================

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

        source_by_id[image_id] = (
            row["source"].strip().lower()
        )

        filename_by_id[image_id] = (
            normalize_name(row["file_name"])
        )


if set(source_by_id) != set(image_info):

    raise RuntimeError(
        "GT与来源映射的image_id不一致。"
    )


for image_id, info in image_info.items():

    if filename_by_id[image_id] != normalize_name(
        info["file_name"]
    ):

        raise RuntimeError(
            f"图片名称映射错误：{image_id}"
        )


source_counts = Counter(
    source_by_id.values()
)

for source, expected in EXPECTED_COUNTS.items():

    if source_counts[source] != expected:

        raise RuntimeError(
            f"{source}数量错误："
            f"{source_counts[source]}"
        )


# ============================================================
# 7. 读取原图预测结果
# ============================================================

print("正在读取原图预测JSON...")

predictions = load_json(PRED_JSON)

areas_by_id = defaultdict(list)

prediction_count = 0

for p in predictions:

    image_id = int(p["image_id"])

    if image_id not in image_info:

        raise RuntimeError(
            f"预测中出现未知image_id：{image_id}"
        )

    score = float(p["score"])

    if not math.isfinite(score):
        raise RuntimeError("预测置信度非法。")

    if score < FEATURE_CONF:
        continue

    bbox = p["bbox"]

    if len(bbox) != 4:
        raise RuntimeError("bbox格式错误。")

    x, y, w, h = map(float, bbox)

    if not all(
        math.isfinite(v)
        for v in [x, y, w, h]
    ):
        raise RuntimeError("预测框包含非法数值。")

    if w <= 0 or h <= 0:
        raise RuntimeError("预测框宽高非法。")

    info = image_info[image_id]

    image_width = int(info["width"])

    image_height = int(info["height"])

    if image_width <= 0 or image_height <= 0:
        raise RuntimeError("GT图片尺寸非法。")

    # 与YOLO LetterBox一致的等比例缩放
    scale = min(
        FEATURE_IMGSZ / image_width,
        FEATURE_IMGSZ / image_height
    )

    scaled_width = w * scale

    scaled_height = h * scale

    area = scaled_width * scaled_height

    areas_by_id[image_id].append(area)

    prediction_count += 1


# ============================================================
# 8. 逐图生成特征
# ============================================================

feature_rows = []

for image_id in sorted(image_info):

    info = image_info[image_id]

    source = source_by_id[image_id]

    width = int(info["width"])

    height = int(info["height"])

    scale = min(
        FEATURE_IMGSZ / width,
        FEATURE_IMGSZ / height
    )

    areas = np.asarray(
        areas_by_id.get(image_id, []),
        dtype=np.float64
    )

    n_det = len(areas)

    if n_det > 0:

        tiny_count = int(
            np.sum(areas < TINY_AREA_THRESHOLD)
        )

        median_area = round(
            float(np.median(areas)),
            4
        )

        tiny_ratio = round(
            tiny_count / n_det,
            6
        )

    else:

        tiny_count = 0

        median_area = ""

        tiny_ratio = 0.0


    filename = info["file_name"]

    feature_rows.append({

        "image_id": image_id,

        "file_name": filename,

        "source": source,

        "sequence_id": sequence_id_from_name(
            filename
        ),

        "width": width,

        "height": height,

        "letterbox_scale": round(
            scale, 8
        ),

        "n_det_conf025": n_det,

        "median_box_area_640": median_area,

        "tiny_ratio": tiny_ratio,

        "tiny_count": tiny_count

    })


# ============================================================
# 9. 保存CSV
# ============================================================

fieldnames = list(
    feature_rows[0].keys()
)

with open(
    OUT_CSV,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(feature_rows)


# ============================================================
# 10. 统计特征分布
# ============================================================

source_summary = {}

for source in EXPECTED_COUNTS:

    rows = [
        x for x in feature_rows
        if x["source"] == source
    ]

    nonempty = [
        x for x in rows
        if x["n_det_conf025"] > 0
    ]

    source_summary[source] = {

        "images": len(rows),

        "sequence_count": len(set(
            x["sequence_id"] for x in rows
        )),

        "zero_detection_images": sum(
            x["n_det_conf025"] == 0
            for x in rows
        ),

        "n_det_conf025": describe([
            x["n_det_conf025"]
            for x in rows
        ]),

        "median_box_area_640": describe([
            x["median_box_area_640"]
            for x in nonempty
        ]),

        "tiny_ratio": describe([
            x["tiny_ratio"]
            for x in nonempty
        ])

    }


summary = {

    "experiment": "Stage6.3A_Corrected_Scene_Features",

    "feature_protocol": {

        "feature_conf": FEATURE_CONF,

        "feature_imgsz": FEATURE_IMGSZ,

        "tiny_area_threshold": TINY_AREA_THRESHOLD,

        "scale_formula": (
            "min(640/image_width,640/image_height)"
        ),

        "area_formula": (
            "(bbox_width*scale)*(bbox_height*scale)"
        ),

        "sequence_id": (
            "filename stem with final underscore segment removed"
        )

    },

    "input": {

        "prediction": str(PRED_JSON),

        "prediction_sha256": sha256_file(PRED_JSON),

        "gt": str(GT_JSON),

        "gt_sha256": sha256_file(GT_JSON),

        "mapping": str(MAPPING_CSV),

        "mapping_sha256": sha256_file(MAPPING_CSV)

    },

    "image_count": {

        "total": len(feature_rows),

        **EXPECTED_COUNTS

    },

    "prediction_count_conf025": prediction_count,

    "source_summary": source_summary,

    "output": {

        "features": str(OUT_CSV),

        "summary": str(OUT_SUMMARY)

    }

}


with open(
    OUT_SUMMARY,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# 11. 输出结果
# ============================================================

print("\n" + "=" * 65)

print("Stage 6.3A 场景特征重建完成")

print("=" * 65)

print("图片总数：", len(feature_rows))

print("conf >= 0.25预测框数量：", prediction_count)

print("\n来源特征统计：")

print(json.dumps(
    source_summary,
    ensure_ascii=False,
    indent=2
))

print("\n输出文件：")

print(OUT_CSV)

print(OUT_SUMMARY)

print("\n没有重新推理，没有重新训练。")
