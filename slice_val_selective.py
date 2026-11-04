# ============================================================
# Stage 6.0
# YOLO11s-DySample 选择性切片与原图融合实验
#
# A组：全部原图推理
# B组：UAVDT原图 + VisDrone 2×2切片
# C组：UAVDT原图 + VisDrone原图/切片融合
#
# 评价：
# 1. 完整验证集COCO指标
# 2. UAVDT来源COCO指标
# 3. VisDrone来源COCO指标
# 4. 单张图片预测特征统计
# 5. 推理时间统计
# ============================================================


import os
import json
import csv
import math
import time
import posixpath
import warnings

from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch

from torchvision.ops import batched_nms
from ultralytics import YOLO

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


warnings.filterwarnings("ignore")


# ============================================================
# 1. 实验参数
# ============================================================

# -------------------- 模型权重 --------------------

WEIGHT = Path(
    "runs/detect/stage2_screening/"
    "YOLO11s-DySample/weights/best.pt"
)


# -------------------- 完整数据集 --------------------

DATA_ROOT = Path(
    "datasets/HighwayVehicle-v1.0"
)

IMAGE_DIR = DATA_ROOT / "images" / "val"


# -------------------- 完整验证集COCO标注 --------------------
# 使用你指定的Stage5统一验证集GT

GT_JSON = Path(
    "runs/detect/source_validation/"
    "Stage5-Baseline-Unified/"
    "annotations_auto_detect_val.json"
)


# -------------------- 分来源COCO标注 --------------------
# 用于识别每张图片究竟来自UAVDT还是VisDrone

SOURCE_GT = {

    "uavdt": Path(
        "runs/detect/source_validation/"
        "DySample-uavdt-val/"
        "annotations_auto_detect_val.json"
    ),

    "visdrone": Path(
        "runs/detect/source_validation/"
        "DySample-visdrone-val/"
        "annotations_auto_detect_val.json"
    )

}


# -------------------- 实验结果保存目录 --------------------

OUT = Path(
    "runs/detect/stage6_selective_oracle"
)


# -------------------- 推理参数 --------------------

IMGSZ = 640

CONF = 0.001

NMS_IOU = 0.7

MAX_DET = 300

DEVICE = 0

HALF = False


# -------------------- 切片参数 --------------------

# 与之前2×2切片实验保持一致

OVERLAP = 0.15


# -------------------- 场景特征统计参数 --------------------

FEATURE_CONF = 0.25

FEATURE_IMGSZ = 640

TINY_AREA_THRESHOLD = 32 ** 2


# -------------------- 数据数量检查 --------------------

EXPECTED_SOURCE_COUNTS = {

    "uavdt": 1320,

    "visdrone": 247

}

EXPECTED_TOTAL_IMAGES = 1567


# -------------------- Ultralytics推理参数 --------------------

PRED_ARGS = {

    "imgsz": IMGSZ,

    "conf": CONF,

    "iou": NMS_IOU,

    "max_det": MAX_DET,

    "device": DEVICE,

    "half": HALF,

    "rect": False,

    "augment": False,

    "agnostic_nms": False,

    "verbose": False,

    "save": False,

}


# ============================================================
# 2. 文件检查
# ============================================================

CHECK_FILES = [

    WEIGHT,

    IMAGE_DIR,

    GT_JSON,

    SOURCE_GT["uavdt"],

    SOURCE_GT["visdrone"]

]


for path in CHECK_FILES:

    if not path.exists():

        raise FileNotFoundError(
            f"\n文件或目录不存在：{path}\n"
        )


OUT.mkdir(
    parents=True,
    exist_ok=True
)


print("\n" + "=" * 65)

print("Stage 6.0：选择性切片实验")

print("=" * 65)

print("模型权重：", WEIGHT)

print("验证图片目录：", IMAGE_DIR)

print("完整GT：", GT_JSON)

print("结果目录：", OUT)


# ============================================================
# 3. 读取COCO标注
# ============================================================

def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


gt_data = load_json(GT_JSON)

gt_images = gt_data["images"]

gt_annotations = gt_data["annotations"]

categories = sorted(
    gt_data["categories"],
    key=lambda x: x["id"]
)


print("\n完整验证集图片数量：", len(gt_images))

print("完整验证集标注数量：", len(gt_annotations))

print("类别信息：", categories)


if len(gt_images) != EXPECTED_TOTAL_IMAGES:

    raise RuntimeError(
        f"\n完整验证集数量异常！\n"
        f"预期：{EXPECTED_TOTAL_IMAGES}\n"
        f"实际：{len(gt_images)}\n"
        f"请检查统一COCO标注文件。"
    )


if len(categories) != 1:

    raise RuntimeError(
        "当前实验按照单类别车辆检测配置编写，"
        "但GT中类别数量不为1。"
    )


# ============================================================
# 4. 根据分来源GT建立图片来源索引
# ============================================================

def normalize_name(name):

    """
    统一文件路径格式。

    兼容：
    1. Linux路径
    2. Windows路径
    3. 相对路径
    4. 带有./的路径
    """

    name = str(name).replace("\\", "/")

    name = posixpath.normpath(name)

    if name.startswith("./"):

        name = name[2:]

    return name


source_exact = {}

source_basename = {}

source_gt_counts = {}


for source, json_path in SOURCE_GT.items():

    data = load_json(json_path)

    images = data["images"]

    source_gt_counts[source] = len(images)

    for info in images:

        filename = normalize_name(
            info["file_name"]
        )

        basename = posixpath.basename(
            filename
        )

        source_exact.setdefault(
            filename,
            set()
        ).add(source)

        source_basename.setdefault(
            basename,
            set()
        ).add(source)


print("\n分来源GT图片数量：")

for source, count in source_gt_counts.items():

    print(f"{source}: {count}")


def identify_source(filename):

    """
    优先使用完整文件名匹配。

    如果完整路径不同，再使用basename匹配。

    如果仍然不能唯一确定来源，则直接报错，
    避免错误地将UAVDT识别成VisDrone。
    """

    key = normalize_name(filename)

    matches = source_exact.get(
        key,
        set()
    )

    if len(matches) == 1:

        return next(iter(matches))

    if len(matches) > 1:

        raise RuntimeError(
            f"图片来源存在歧义：{filename}"
        )

    basename = posixpath.basename(key)

    matches = source_basename.get(
        basename,
        set()
    )

    if len(matches) == 1:

        return next(iter(matches))

    if len(matches) > 1:

        raise RuntimeError(
            f"图片文件名重复，无法确定来源：{filename}"
        )

    raise RuntimeError(
        f"无法识别图片来源：{filename}\n"
        f"请检查统一GT与分来源GT是否对应。"
    )


# ============================================================
# 5. 建立本地图片索引
# ============================================================

image_files = [

    p for p in IMAGE_DIR.rglob("*")

    if p.is_file()

    and p.suffix.lower() in {

        ".jpg",

        ".jpeg",

        ".png",

        ".bmp"

    }

]


if len(image_files) == 0:

    raise RuntimeError(
        "验证图片目录为空。"
    )


image_by_relative = {}

image_by_basename = {}


for image_path in image_files:

    relative_name = normalize_name(
        image_path.relative_to(IMAGE_DIR)
    )

    basename = image_path.name

    image_by_relative[relative_name] = image_path

    image_by_basename.setdefault(
        basename,
        []
    ).append(image_path)


def resolve_image_path(filename):

    """
    根据COCO GT中的file_name定位本地图片。

    优先匹配完整相对路径。

    如果匹配不到，则匹配basename。
    """

    raw = Path(str(filename))

    # 绝对路径
    if raw.is_absolute() and raw.is_file():

        return raw

    # 直接拼接
    candidate = IMAGE_DIR / raw

    if candidate.is_file():

        return candidate

    # 相对路径匹配
    key = normalize_name(filename)

    if key in image_by_relative:

        return image_by_relative[key]

    # 文件名匹配
    basename = posixpath.basename(key)

    candidates = image_by_basename.get(
        basename,
        []
    )

    if len(candidates) == 1:

        return candidates[0]

    if len(candidates) > 1:

        raise RuntimeError(
            f"存在重名图片，无法唯一定位：{filename}"
        )

    raise FileNotFoundError(
        f"找不到本地图片：{filename}"
    )


# ============================================================
# 6. 生成统一验证记录
# ============================================================

records = []

image_ids = []

image_ids_by_source = {

    "uavdt": [],

    "visdrone": []

}


for info in gt_images:

    image_id = int(info["id"])

    source = identify_source(
        info["file_name"]
    )

    image_path = resolve_image_path(
        info["file_name"]
    )

    records.append({

        "info": info,

        "image_id": image_id,

        "source": source,

        "image_path": image_path

    })

    image_ids.append(image_id)

    image_ids_by_source[source].append(
        image_id
    )


# 检查image_id是否重复

if len(image_ids) != len(set(image_ids)):

    raise RuntimeError(
        "统一GT中存在重复image_id。"
    )


# 检查分来源数量

actual_source_counts = {

    source: len(ids)

    for source, ids in image_ids_by_source.items()

}


print("\n实际识别的图片来源数量：")

for source, count in actual_source_counts.items():

    print(f"{source}: {count}")


for source in EXPECTED_SOURCE_COUNTS:

    expected = EXPECTED_SOURCE_COUNTS[source]

    actual = actual_source_counts[source]

    if actual != expected:

        raise RuntimeError(

            f"\n{source}图片数量异常！\n"

            f"预期：{expected}\n"

            f"实际：{actual}\n"

            f"请检查来源GT匹配是否正确。"

        )


if len(records) != EXPECTED_TOTAL_IMAGES:

    raise RuntimeError(
        "图片总数与预期不一致。"
    )


print("\n数据检查通过。")

print("UAVDT：", actual_source_counts["uavdt"])

print("VisDrone：", actual_source_counts["visdrone"])

print("总计：", len(records))


# ============================================================
# 7. 加载模型
# ============================================================

print("\n正在加载模型...")

model = YOLO(str(WEIGHT))

print("模型权重：", WEIGHT)

print("模型类别：", model.names)


model_nc = len(model.names)

if model_nc != len(categories):

    raise RuntimeError(

        f"模型类别数量与GT类别数量不一致："

        f"模型={model_nc}，GT={len(categories)}"

    )


# 当前数据集为单类别车辆检测

CATEGORY_ID = int(categories[0]["id"])


# ============================================================
# 8. 空预测结果
# ============================================================

def empty_result():

    return (

        np.zeros(
            (0, 4),
            dtype=np.float32
        ),

        np.zeros(
            (0,),
            dtype=np.float32
        ),

        np.zeros(
            (0,),
            dtype=np.int64
        )

    )


# ============================================================
# 9. 提取模型预测结果
# ============================================================

def extract_result(

    result,

    offset_x=0,

    offset_y=0

):

    """

    提取YOLO预测框。

    将切片局部坐标映射回原图坐标。
    """

    if result.boxes is None:

        return empty_result()

    if len(result.boxes) == 0:

        return empty_result()


    boxes = (

        result.boxes.xyxy

        .detach()

        .cpu()

        .numpy()

        .astype(np.float32)

    )

    scores = (

        result.boxes.conf

        .detach()

        .cpu()

        .numpy()

        .astype(np.float32)

    )

    classes = (

        result.boxes.cls

        .detach()

        .cpu()

        .numpy()

        .astype(np.int64)

    )


    # 坐标映射

    boxes[:, [0, 2]] += offset_x

    boxes[:, [1, 3]] += offset_y


    return boxes, scores, classes


# ============================================================
# 10. 全局NMS
# ============================================================

def merge_nms(

    boxes,

    scores,

    classes,

    width,

    height

):

    """

    将原图预测框、切片预测框进行全局NMS。

    保持与原有切片实验相同的NMS阈值。
    """

    if len(boxes) == 0:

        return empty_result()


    boxes = np.asarray(

        boxes,

        dtype=np.float32

    ).reshape(-1, 4)


    scores = np.asarray(

        scores,

        dtype=np.float32

    )


    classes = np.asarray(

        classes,

        dtype=np.int64

    )


    # 限制预测框坐标

    boxes[:, [0, 2]] = np.clip(

        boxes[:, [0, 2]],

        0,

        width

    )


    boxes[:, [1, 3]] = np.clip(

        boxes[:, [1, 3]],

        0,

        height

    )


    # 删除无效框

    valid = (

        (boxes[:, 2] > boxes[:, 0])

        &

        (boxes[:, 3] > boxes[:, 1])

        &

        np.isfinite(scores)

    )


    boxes = boxes[valid]

    scores = scores[valid]

    classes = classes[valid]


    if len(boxes) == 0:

        return empty_result()


    keep = batched_nms(

        torch.from_numpy(boxes),

        torch.from_numpy(scores),

        torch.from_numpy(classes),

        NMS_IOU

    )


    keep = keep[:MAX_DET].cpu().numpy()


    return (

        boxes[keep],

        scores[keep],

        classes[keep]

    )


# ============================================================
# 11. 生成2×2重叠切片
# ============================================================

def make_tiles(image):

    """

    按照原有slice_val_2x2.py的方式生成2×2切片。

    OVERLAP=0.15。

    四个切片映射回原图时使用各自的左上角偏移量。
    """

    height, width = image.shape[:2]


    tile_w = math.ceil(

        width / (2 - OVERLAP)

    )


    tile_h = math.ceil(

        height / (2 - OVERLAP)

    )


    tile_w = min(tile_w, width)

    tile_h = min(tile_h, height)


    xs = sorted(set([

        0,

        max(0, width - tile_w)

    ]))


    ys = sorted(set([

        0,

        max(0, height - tile_h)

    ]))


    tiles = []

    offsets = []


    for y in ys:

        for x in xs:

            tile = image[

                y:y + tile_h,

                x:x + tile_w

            ].copy()


            tiles.append(tile)

            offsets.append((x, y))


    return tiles, offsets


# ============================================================
# 12. 原图推理
# ============================================================

def predict_original(image):

    """

    原图640×640推理。

    返回原图坐标系下的预测结果。
    """

    results = model.predict(

        source=image,

        batch=1,

        **PRED_ARGS

    )


    result = results[0]


    boxes, scores, classes = extract_result(
        result
    )


    height, width = image.shape[:2]


    return merge_nms(

        boxes,

        scores,

        classes,

        width,

        height

    )


# ============================================================
# 13. 2×2切片推理
# ============================================================

def predict_tiled(image):

    """

    将原图划分为2×2重叠切片。

    每个切片分别推理。

    所有预测框映射回原图坐标。

    最后统一进行全局NMS。
    """

    tiles, offsets = make_tiles(image)


    results = model.predict(

        source=tiles,

        batch=len(tiles),

        **PRED_ARGS

    )


    if len(results) != len(tiles):

        raise RuntimeError(
            "切片数量与推理结果数量不一致。"
        )


    all_boxes = []

    all_scores = []

    all_classes = []


    for result, (offset_x, offset_y) in zip(

        results,

        offsets

    ):


        boxes, scores, classes = extract_result(

            result,

            offset_x=offset_x,

            offset_y=offset_y

        )


        if len(boxes) > 0:

            all_boxes.append(boxes)

            all_scores.append(scores)

            all_classes.append(classes)


    if len(all_boxes) == 0:

        return empty_result()


    boxes = np.concatenate(

        all_boxes,

        axis=0

    )


    scores = np.concatenate(

        all_scores,

        axis=0

    )


    classes = np.concatenate(

        all_classes,

        axis=0

    )


    height, width = image.shape[:2]


    # 切片预测框全局NMS

    return merge_nms(

        boxes,

        scores,

        classes,

        width,

        height

    )


# ============================================================
# 14. 原图 + 切片预测融合
# ============================================================

def predict_fused(

    original_result,

    tiled_result,

    image

):

    """

    C组：

    将原图预测框与切片预测框进行融合。

    两组预测均已经映射至原图坐标系。

    最后再次进行全局NMS。
    """

    b0, s0, c0 = original_result

    b1, s1, c1 = tiled_result


    if len(b0) == 0 and len(b1) == 0:

        return empty_result()


    if len(b0) == 0:

        boxes = b1

        scores = s1

        classes = c1


    elif len(b1) == 0:

        boxes = b0

        scores = s0

        classes = c0


    else:

        boxes = np.concatenate(

            [b0, b1],

            axis=0

        )


        scores = np.concatenate(

            [s0, s1],

            axis=0

        )


        classes = np.concatenate(

            [c0, c1],

            axis=0

        )


    height, width = image.shape[:2]


    return merge_nms(

        boxes,

        scores,

        classes,

        width,

        height

    )


# ============================================================
# 15. 转换为COCO预测格式
# ============================================================

def to_coco(

    image_id,

    boxes,

    scores,

    classes

):

    output = []


    for box, score, cls in zip(

        boxes,

        scores,

        classes

    ):


        cls = int(cls)


        if cls != 0:

            continue


        x1, y1, x2, y2 = map(

            float,

            box

        )


        box_w = x2 - x1

        box_h = y2 - y1


        if box_w <= 0 or box_h <= 0:

            continue


        output.append({

            "image_id": int(image_id),

            "category_id": CATEGORY_ID,

            "bbox": [

                x1,

                y1,

                box_w,

                box_h

            ],

            "score": float(score)

        })


    return output


# ============================================================
# 16. 保存COCO预测文件
# ============================================================

def save_predictions(

    predictions,

    filename

):

    path = OUT / filename


    with open(

        path,

        "w",

        encoding="utf-8"

    ) as f:


        json.dump(

            predictions,

            f,

            ensure_ascii=False,

            separators=(",", ":"),

            allow_nan=False

        )


    print(

        f"\n预测文件已保存：{path}"

    )


    print(

        "预测框数量：",

        len(predictions)

    )


# ============================================================
# 17. 原图预测特征统计
# ============================================================

def calculate_features(

    boxes,

    scores,

    image_width,

    image_height

):

    """

    使用原图推理结果提取场景特征。

    置信度阈值：0.25。

    将原图预测框面积换算到640×640尺度。

    统计：

    1. 预测目标数量
    2. 中位框面积
    3. 小框比例
    """

    keep = scores >= FEATURE_CONF


    selected_boxes = boxes[keep]


    n_det = len(selected_boxes)


    if n_det == 0:

        return {

            "n_det_conf025": 0,

            "median_box_area_640": "",

            "tiny_ratio": 0.0,

            "tiny_count": 0

        }


    box_width = (

        selected_boxes[:, 2]

        -

        selected_boxes[:, 0]

    )


    box_height = (

        selected_boxes[:, 3]

        -

        selected_boxes[:, 1]

    )


    scale_x = FEATURE_IMGSZ / image_width

    scale_y = FEATURE_IMGSZ / image_height


    scaled_width = box_width * scale_x

    scaled_height = box_height * scale_y


    areas = (

        scaled_width

        *

        scaled_height

    )


    tiny_count = int(

        np.sum(

            areas < TINY_AREA_THRESHOLD

        )

    )


    median_area = float(

        np.median(areas)

    )


    tiny_ratio = tiny_count / n_det


    return {

        "n_det_conf025": int(n_det),

        "median_box_area_640": round(

            median_area,

            4

        ),

        "tiny_ratio": round(

            tiny_ratio,

            6

        ),

        "tiny_count": tiny_count

    }


# ============================================================
# 18. 初始化三组预测
# ============================================================

pred_original = []

pred_selective_tile = []

pred_selective_fusion = []


# 按image_id保存原图预测

original_by_id = {}


# 推理时间统计

time_original_by_source = {

    "uavdt": 0.0,

    "visdrone": 0.0

}


time_tiled_visdrone = 0.0

time_fusion_postprocess = 0.0


feature_rows = []


# ============================================================
# 19. 正式执行三组实验
# ============================================================

print("\n" + "=" * 65)

print("开始三组选择性推理实验")

print("=" * 65)


total_start = perf_counter()


for index, record in enumerate(

    records,

    start=1

):


    info = record["info"]

    image_id = record["image_id"]

    source = record["source"]

    image_path = record["image_path"]


    # --------------------------------------------------------
    # 读取原图
    # --------------------------------------------------------

    image = cv2.imread(

        str(image_path)

    )


    if image is None:

        raise RuntimeError(

            f"图片读取失败：{image_path}"

        )


    height, width = image.shape[:2]


    if (

        width != int(info["width"])

        or

        height != int(info["height"])

    ):

        raise RuntimeError(

            f"\n图片尺寸与GT不一致：{image_path.name}\n"

            f"实际：{width}×{height}\n"

            f"GT：{info['width']}×{info['height']}"

        )


    # ========================================================
    # A组：所有图片原图推理
    # ========================================================

    start = perf_counter()


    original_result = predict_original(

        image

    )


    original_time = (

        perf_counter() - start

    )


    time_original_by_source[source] += original_time


    original_by_id[image_id] = original_result


    b0, s0, c0 = original_result


    pred_original.extend(

        to_coco(

            image_id,

            b0,

            s0,

            c0

        )

    )


    # ========================================================
    # B组、C组：按照来源选择推理方式
    # ========================================================

    if source == "uavdt":


        # ----------------------------------------------------
        # UAVDT：B组和C组均使用原图结果
        # ----------------------------------------------------

        pred_selective_tile.extend(

            to_coco(

                image_id,

                b0,

                s0,

                c0

            )

        )


        pred_selective_fusion.extend(

            to_coco(

                image_id,

                b0,

                s0,

                c0

            )

        )


    elif source == "visdrone":


        # ----------------------------------------------------
        # VisDrone：执行2×2切片推理
        # ----------------------------------------------------

        start = perf_counter()


        tiled_result = predict_tiled(

            image

        )


        time_tiled_visdrone += (

            perf_counter() - start

        )


        b1, s1, c1 = tiled_result


        # ----------------------------------------------------
        # B组：仅使用切片预测
        # ----------------------------------------------------

        pred_selective_tile.extend(

            to_coco(

                image_id,

                b1,

                s1,

                c1

            )

        )


        # ----------------------------------------------------
        # C组：原图 + 切片融合
        # ----------------------------------------------------

        start = perf_counter()


        fused_result = predict_fused(

            original_result,

            tiled_result,

            image

        )


        time_fusion_postprocess += (

            perf_counter() - start

        )


        b2, s2, c2 = fused_result


        pred_selective_fusion.extend(

            to_coco(

                image_id,

                b2,

                s2,

                c2

            )

        )


    else:

        raise RuntimeError(

            f"未知数据来源：{source}"

        )


    # ========================================================
    # 统计原图预测特征
    # ========================================================

    features = calculate_features(

        b0,

        s0,

        width,

        height

    )


    feature_rows.append({

        "image_id": image_id,

        "file_name": info["file_name"],

        "source": source,

        "width": width,

        "height": height,

        **features

    })


    # ========================================================
    # 输出进度
    # ========================================================

    if (

        index % 20 == 0

        or

        index == len(records)

    ):


        print(

            f"进度：{index}/{len(records)} | "

            f"来源：{source} | "

            f"A组预测：{len(pred_original)} | "

            f"B组预测：{len(pred_selective_tile)} | "

            f"C组预测：{len(pred_selective_fusion)}"

        )


total_inference_time = (

    perf_counter() - total_start

)


# ============================================================
# 20. 保存三组预测结果
# ============================================================

print("\n开始保存预测JSON...")


save_predictions(

    pred_original,

    "all_original_predictions.json"

)


save_predictions(

    pred_selective_tile,

    "selective_tile_predictions.json"

)


save_predictions(

    pred_selective_fusion,

    "selective_fusion_predictions.json"

)


# ============================================================
# 21. 保存每张图片的场景特征
# ============================================================

feature_path = OUT / "per_image_features.csv"


with open(

    feature_path,

    "w",

    newline="",

    encoding="utf-8-sig"

) as f:


    fieldnames = [

        "image_id",

        "file_name",

        "source",

        "width",

        "height",

        "n_det_conf025",

        "median_box_area_640",

        "tiny_ratio",

        "tiny_count"

    ]


    writer = csv.DictWriter(

        f,

        fieldnames=fieldnames

    )


    writer.writeheader()


    writer.writerows(

        feature_rows

    )


print(

    "\n场景特征文件已保存：",

    feature_path

)


# ============================================================
# 22. 保存来源映射
# ============================================================

mapping_path = OUT / "image_source_mapping.csv"


with open(

    mapping_path,

    "w",

    newline="",

    encoding="utf-8-sig"

) as f:


    writer = csv.DictWriter(

        f,

        fieldnames=[

            "image_id",

            "file_name",

            "source"

        ]

    )


    writer.writeheader()


    for record in records:


        writer.writerow({

            "image_id": record["image_id"],

            "file_name": record["info"]["file_name"],

            "source": record["source"]

        })


# ============================================================
# 23. 统一COCO评价函数
# ============================================================

print("\n正在初始化COCO评价器...")


coco_gt = COCO(

    str(GT_JSON)

)


def evaluate_coco(

    predictions,

    selected_image_ids,

    group_name

):

    """

    对指定图片集合进行COCO bbox评价。

    支持：

    1. 完整验证集
    2. UAVDT子集
    3. VisDrone子集
    """

    selected_image_ids = sorted(

        set(

            int(x)

            for x in selected_image_ids

        )

    )


    if len(selected_image_ids) == 0:

        raise RuntimeError(

            f"{group_name}没有有效图片。"

        )


    selected_id_set = set(

        selected_image_ids

    )


    selected_predictions = [

        item

        for item in predictions

        if int(item["image_id"]) in selected_id_set

    ]


    # --------------------------------------------------------
    # 有预测框
    # --------------------------------------------------------

    if len(selected_predictions) > 0:


        coco_dt = coco_gt.loadRes(

            selected_predictions

        )


    # --------------------------------------------------------
    # 没有预测框
    # --------------------------------------------------------

    else:


        from pycocotools.coco import COCO as EmptyCOCO


        coco_dt = EmptyCOCO()


        coco_dt.dataset = {

            "images": [

                img

                for img in coco_gt.dataset["images"]

                if int(img["id"]) in selected_id_set

            ],

            "categories": coco_gt.dataset["categories"],

            "annotations": []

        }


        coco_dt.createIndex()


    # --------------------------------------------------------
    # COCOeval
    # --------------------------------------------------------

    evaluator = COCOeval(

        coco_gt,

        coco_dt,

        "bbox"

    )


    evaluator.params.imgIds = selected_image_ids


    evaluator.params.catIds = [

        int(x["id"])

        for x in categories

    ]


    evaluator.params.maxDets = [

        1,

        10,

        100

    ]


    evaluator.evaluate()

    evaluator.accumulate()

    evaluator.summarize()


    stats = evaluator.stats


    metrics = {

        "AP50-95": float(stats[0]),

        "AP50": float(stats[1]),

        "AP75": float(stats[2]),

        "APs": float(stats[3]),

        "APm": float(stats[4]),

        "APl": float(stats[5]),

        "AR@1": float(stats[6]),

        "AR@10": float(stats[7]),

        "AR@100": float(stats[8]),

        "ARs": float(stats[9]),

        "ARm": float(stats[10]),

        "ARl": float(stats[11])

    }


    print(

        f"\n{group_name}评价完成。"

    )


    return metrics


# ============================================================
# 24. 定义三组实验
# ============================================================

EXPERIMENTS = {

    "A_Original": pred_original,

    "B_Selective_Tile": pred_selective_tile,

    "C_Selective_Fusion": pred_selective_fusion

}


# ============================================================
# 25. 完整验证集及分来源评价
# ============================================================

all_results = {}


print("\n" + "=" * 65)

print("开始COCO指标评价")

print("=" * 65)


for group_name, predictions in EXPERIMENTS.items():


    print("\n\n")

    print("#" * 65)

    print("当前实验组：", group_name)

    print("#" * 65)


    # --------------------------------------------------------
    # 完整验证集
    # --------------------------------------------------------

    print("\n========== 完整验证集 ==========")


    full_metrics = evaluate_coco(

        predictions,

        image_ids,

        group_name + "_Full"

    )


    # --------------------------------------------------------
    # UAVDT
    # --------------------------------------------------------

    print("\n========== UAVDT ==========")


    uavdt_metrics = evaluate_coco(

        predictions,

        image_ids_by_source["uavdt"],

        group_name + "_UAVDT"

    )


    # --------------------------------------------------------
    # VisDrone
    # --------------------------------------------------------

    print("\n========== VisDrone ==========")


    visdrone_metrics = evaluate_coco(

        predictions,

        image_ids_by_source["visdrone"],

        group_name + "_VisDrone"

    )


    all_results[group_name] = {

        "full": full_metrics,

        "uavdt": uavdt_metrics,

        "visdrone": visdrone_metrics

    }


# ============================================================
# 26. 计算各组相对于A组的变化
# ============================================================

def calculate_delta(

    current,

    baseline

):

    result = {}


    for key in current:

        result[key] = round(

            current[key] - baseline[key],

            6

        )


    return result


baseline_full = all_results["A_Original"]["full"]

baseline_uavdt = all_results["A_Original"]["uavdt"]

baseline_visdrone = all_results["A_Original"]["visdrone"]


delta_results = {}


for group_name in EXPERIMENTS:


    delta_results[group_name] = {


        "full": calculate_delta(

            all_results[group_name]["full"],

            baseline_full

        ),


        "uavdt": calculate_delta(

            all_results[group_name]["uavdt"],

            baseline_uavdt

        ),


        "visdrone": calculate_delta(

            all_results[group_name]["visdrone"],

            baseline_visdrone

        )

    }


# ============================================================
# 27. 推理时间统计
# ============================================================

original_uavdt_time = (

    time_original_by_source["uavdt"]

)


original_visdrone_time = (

    time_original_by_source["visdrone"]

)


tile_time = time_tiled_visdrone


# A组：所有图片原图推理

time_A = (

    original_uavdt_time

    +

    original_visdrone_time

)


# B组：UAVDT原图 + VisDrone切片

time_B = (

    original_uavdt_time

    +

    tile_time

)


# C组：UAVDT原图 + VisDrone原图和切片

time_C = (

    original_uavdt_time

    +

    original_visdrone_time

    +

    tile_time

    +

    time_fusion_postprocess

)


time_results = {

    "A_Original": {

        "inference_seconds": round(

            time_A,

            4

        )

    },

    "B_Selective_Tile": {

        "inference_seconds": round(

            time_B,

            4

        )

    },

    "C_Selective_Fusion": {

        "inference_seconds": round(

            time_C,

            4

        )

    }

}


# ============================================================
# 28. 保存完整实验结果
# ============================================================

summary = {

    "experiment": "Stage6.0_Selective_Tile_Fusion",

    "model": str(WEIGHT),

    "image_count": len(records),

    "source_image_count": actual_source_counts,

    "config": {

        "imgsz": IMGSZ,

        "conf": CONF,

        "nms_iou": NMS_IOU,

        "max_det": MAX_DET,

        "overlap": OVERLAP,

        "device": DEVICE,

        "half": HALF,

        "augment": False

    },

    "inference_time": time_results,

    "wall_time_seconds": round(

        total_inference_time,

        4

    ),

    "groups": all_results,

    "delta_vs_original": delta_results

}


summary_path = OUT / "metrics_summary.json"


with open(

    summary_path,

    "w",

    encoding="utf-8"

) as f:


    json.dump(

        summary,

        f,

        indent=4,

        ensure_ascii=False,

        allow_nan=False

    )


# ============================================================
# 29. 单独保存分来源评价结果
# ============================================================

source_summary = {

    group_name: {

        "uavdt": result["uavdt"],

        "visdrone": result["visdrone"]

    }

    for group_name, result in all_results.items()

}


source_summary_path = OUT / "per_source_metrics.json"


with open(

    source_summary_path,

    "w",

    encoding="utf-8"

) as f:


    json.dump(

        source_summary,

        f,

        indent=4,

        ensure_ascii=False,

        allow_nan=False

    )


# ============================================================
# 30. 输出最终对比表
# ============================================================

print("\n\n")

print("=" * 85)

print("Stage 6.0 最终实验结果")

print("=" * 85)


KEY_METRICS = [

    "AP50",

    "AP75",

    "AP50-95",

    "APs",

    "APm",

    "APl",

    "ARs"

]


for source_name in [

    "full",

    "uavdt",

    "visdrone"

]:


    print("\n")

    print("-" * 85)

    print("评价范围：", source_name)

    print("-" * 85)


    print(

        f"{'指标':<15}"

        f"{'A组原图':>15}"

        f"{'B组切片':>15}"

        f"{'C组融合':>15}"

    )


    for metric in KEY_METRICS:


        a = all_results["A_Original"][source_name][metric]

        b = all_results["B_Selective_Tile"][source_name][metric]

        c = all_results["C_Selective_Fusion"][source_name][metric]


        print(

            f"{metric:<15}"

            f"{a:>15.4f}"

            f"{b:>15.4f}"

            f"{c:>15.4f}"

        )


# ============================================================
# 31. 输出相对于A组的变化
# ============================================================

print("\n\n")

print("=" * 85)

print("相对于A组原图推理的指标变化")

print("=" * 85)


for source_name in [

    "full",

    "uavdt",

    "visdrone"

]:


    print("\n评价范围：", source_name)


    for group_name in [

        "B_Selective_Tile",

        "C_Selective_Fusion"

    ]:


        print("\n", group_name)


        for metric in [

            "AP50",

            "AP75",

            "AP50-95",

            "APs",

            "APm",

            "APl",

            "ARs"

        ]:


            delta = delta_results[

                group_name

            ][

                source_name

            ][

                metric

            ]


            print(

                f"{metric:<12}"

                f"{delta:+.6f}"

            )


# ============================================================
# 32. 输出推理时间
# ============================================================

print("\n\n")

print("=" * 65)

print("推理时间统计")

print("=" * 65)


for group_name, result in time_results.items():


    seconds = result["inference_seconds"]


    print(

        f"{group_name:<25}"

        f"{seconds:>12.3f} 秒"

    )


    print(

        f"平均每张图片："

        f"{seconds / EXPECTED_TOTAL_IMAGES * 1000:.3f} ms"

    )


# ============================================================
# 33. 完成
# ============================================================

print("\n\n")

print("=" * 65)

print("Stage 6.0 实验全部完成")

print("=" * 65)


print("\n结果保存目录：")

print(OUT)


print("\n主要结果文件：")

print(

    OUT / "metrics_summary.json"

)

print(

    OUT / "per_source_metrics.json"

)

print(

    OUT / "per_image_features.csv"

)

print(

    OUT / "all_original_predictions.json"

)

print(

    OUT / "selective_tile_predictions.json"

)

print(

    OUT / "selective_fusion_predictions.json"

)


print("\n实验结束。")