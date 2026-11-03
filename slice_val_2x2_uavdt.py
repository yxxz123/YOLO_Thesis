
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.ops import batched_nms
from ultralytics import YOLO

# 优先使用仓库原有的COCO评价实现
try:
    from faster_coco_eval import COCO, COCOeval_faster
    USE_FASTER = True
except ImportError:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    USE_FASTER = False


# ==========================================================
# 1. 固定实验配置
# ==========================================================

WEIGHT = Path(
    "runs/detect/stage2_screening/"
    "YOLO11s-DySample/weights/best.pt"
)

DATA_ROOT = Path("datasets/HighwayVehicle-v1.0")
IMAGE_DIR = DATA_ROOT / "images" / "val"

GT_JSON = Path(
    "runs/detect/source_validation/"
    "DySample-uavdt-val/"
    "annotations_auto_detect_val.json"
)

OUT = Path(
    "runs/detect/slice_ablation/"
    "YOLO11s-DySample-UAVDT"
)

IMGSZ = 640
CONF = 0.001
NMS_IOU = 0.8
MAX_DET = 300

# 相邻切片重叠比例：15%
OVERLAP = 0.15

DEVICE = 0

PRED_ARGS = dict(
    imgsz=IMGSZ,
    conf=CONF,
    iou=NMS_IOU,
    max_det=MAX_DET,
    device=DEVICE,
    half=False,
    rect=False,
    augment=False,
    agnostic_nms=False,
    verbose=False,
    save=False,
)


# ==========================================================
# 2. 文件检查及读取
# ==========================================================

for p in [WEIGHT, GT_JSON, IMAGE_DIR]:
    if not p.exists():
        raise FileNotFoundError(f"文件不存在: {p}")

OUT.mkdir(parents=True, exist_ok=True)

with open(GT_JSON, "r", encoding="utf-8") as f:
    gt_data = json.load(f)

gt_images = gt_data["images"]
categories = sorted(
    gt_data["categories"],
    key=lambda x: x["id"]
)

# 文件名映射，避免错误匹配图片ID
image_map = {}

for p in IMAGE_DIR.rglob("*"):
    if p.is_file() and p.suffix.lower() in {
        ".jpg", ".jpeg", ".png", ".bmp"
    }:
        if p.name in image_map:
            raise RuntimeError(f"图片重名: {p.name}")
        image_map[p.name] = p

records = []

for info in gt_images:
    name = Path(info["file_name"]).name

    if name not in image_map:
        raise FileNotFoundError(f"找不到图片: {name}")

    records.append((info, image_map[name]))

if len(records) != len(gt_images):
    raise RuntimeError("图片数量与GT不一致")

print("验证图片:", len(records))
print("真实标注:", len(gt_data["annotations"]))


# ==========================================================
# 3. 加载原始DySample
# ==========================================================

model = YOLO(str(WEIGHT))

print("模型权重:", WEIGHT)
print("模型类别:", model.names)

if len(categories) != len(model.names):
    raise RuntimeError("模型类别数量与COCO标注不一致")


# ==========================================================
# 4. 提取模型预测框
# ==========================================================

def extract_result(result, offset_x=0, offset_y=0):

    if result.boxes is None or len(result.boxes) == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
    scores = result.boxes.conf.cpu().numpy().astype(np.float32)
    classes = result.boxes.cls.cpu().numpy().astype(np.int64)

    # 将切片坐标映射回原图坐标
    boxes[:, [0, 2]] += offset_x
    boxes[:, [1, 3]] += offset_y

    return boxes, scores, classes


# ==========================================================
# 5. 全局NMS
# ==========================================================

def merge_nms(boxes, scores, classes, width, height):

    if len(boxes) == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32)
    classes = np.asarray(classes, dtype=np.int64)

    # 限制预测框在原图范围内
    boxes[:, [0, 2]] = np.clip(
        boxes[:, [0, 2]], 0, width
    )
    boxes[:, [1, 3]] = np.clip(
        boxes[:, [1, 3]], 0, height
    )

    valid = (
        (boxes[:, 2] > boxes[:, 0]) &
        (boxes[:, 3] > boxes[:, 1])
    )

    boxes = boxes[valid]
    scores = scores[valid]
    classes = classes[valid]

    if len(boxes) == 0:
        return boxes, scores, classes

    keep = batched_nms(
        torch.from_numpy(boxes),
        torch.from_numpy(scores),
        torch.from_numpy(classes),
        NMS_IOU
    )

    keep = keep[:MAX_DET].cpu().numpy()

    return boxes[keep], scores[keep], classes[keep]


# ==========================================================
# 6. 生成2×2重叠切片
# ==========================================================

def make_tiles(image):

    height, width = image.shape[:2]

    # 每个切片约占原图宽高的54.05%
    tile_w = math.ceil(width / (2 - OVERLAP))
    tile_h = math.ceil(height / (2 - OVERLAP))

    tile_w = min(tile_w, width)
    tile_h = min(tile_h, height)

    # 左右、上下各两个起点
    xs = [0, max(0, width - tile_w)]
    ys = [0, max(0, height - tile_h)]

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


# ==========================================================
# 7. A组：原图推理
# ==========================================================

def predict_original(image):

    result = model.predict(
        source=image,
        batch=1,
        **PRED_ARGS
    )[0]

    boxes, scores, classes = extract_result(result)

    h, w = image.shape[:2]

    return merge_nms(
        boxes, scores, classes, w, h
    )


# ==========================================================
# 8. B组：2×2切片推理
# ==========================================================

def predict_tiled(image):

    tiles, offsets = make_tiles(image)

    # 一张原图的四个切片组成一个batch
    results = model.predict(
        source=tiles,
        batch=4,
        **PRED_ARGS
    )

    all_boxes = []
    all_scores = []
    all_classes = []

    for result, (ox, oy) in zip(results, offsets):

        boxes, scores, classes = extract_result(
            result,
            offset_x=ox,
            offset_y=oy
        )

        if len(boxes):
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_classes.append(classes)

    h, w = image.shape[:2]

    if not all_boxes:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    boxes = np.concatenate(all_boxes)
    scores = np.concatenate(all_scores)
    classes = np.concatenate(all_classes)

    # 四个切片合并后，统一进行一次全局NMS
    return merge_nms(
        boxes, scores, classes, w, h
    )


# ==========================================================
# 9. 转换为COCO预测格式
# ==========================================================

def to_coco(image_id, boxes, scores, classes):

    output = []

    for box, score, cls in zip(boxes, scores, classes):

        cls = int(cls)

        if cls >= len(categories):
            continue

        x1, y1, x2, y2 = map(float, box)

        output.append({
            "image_id": int(image_id),
            "category_id": int(categories[cls]["id"]),
            "bbox": [
                x1,
                y1,
                x2 - x1,
                y2 - y1
            ],
            "score": float(score)
        })

    return output


# ==========================================================
# 10. 两组推理
# ==========================================================

pred_original = []
pred_tiled = []

start_time = time.time()

for index, (info, image_path) in enumerate(records, 1):

    image = cv2.imread(str(image_path))

    if image is None:
        raise RuntimeError(f"图片读取失败: {image_path}")

    h, w = image.shape[:2]

    if w != info["width"] or h != info["height"]:
        raise RuntimeError(
            f"图片尺寸不一致: {image_path.name}"
        )

    # A组：原图
    b0, s0, c0 = predict_original(image)

    pred_original.extend(
        to_coco(info["id"], b0, s0, c0)
    )

    # B组：2×2切片
    b1, s1, c1 = predict_tiled(image)

    pred_tiled.extend(
        to_coco(info["id"], b1, s1, c1)
    )

    if index % 20 == 0 or index == len(records):
        print(
            f"进度: {index}/{len(records)} | "
            f"原图预测: {len(pred_original)} | "
            f"切片预测: {len(pred_tiled)}"
        )

# 保存预测文件
original_json = OUT / "original_predictions.json"
tiled_json = OUT / "tiled_2x2_predictions.json"

with open(original_json, "w", encoding="utf-8") as f:
    json.dump(pred_original, f)

with open(tiled_json, "w", encoding="utf-8") as f:
    json.dump(pred_tiled, f)

print("\n推理耗时:", round(time.time() - start_time, 2), "秒")


# ==========================================================
# 11. 使用相同COCO GT评价两组结果
# ==========================================================

def evaluate_coco(pred_path):

    coco_gt = COCO(str(GT_JSON))
    coco_dt = coco_gt.loadRes(str(pred_path))

    if USE_FASTER:

        evaluator = COCOeval_faster(
            coco_gt,
            coco_dt,
            iouType="bbox",
            print_function=print
        )

        evaluator.params.imgIds = [
            int(x["id"]) for x in coco_gt.dataset["images"]
        ]

        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

        raw = evaluator.stats_as_dict

        return {
            "AP50": float(raw["AP_50"]),
            "AP75": float(raw["AP_75"]),
            "AP50-95": float(raw["AP_all"]),
            "APs": float(raw["AP_small"]),
            "APm": float(raw["AP_medium"]),
            "APl": float(raw["AP_large"]),
        }

    else:

        evaluator = COCOeval(
            coco_gt,
            coco_dt,
            "bbox"
        )

        evaluator.params.imgIds = [
            int(x["id"]) for x in coco_gt.dataset["images"]
        ]

        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

        s = evaluator.stats

        return {
            "AP50-95": float(s[0]),
            "AP50": float(s[1]),
            "AP75": float(s[2]),
            "APs": float(s[3]),
            "APm": float(s[4]),
            "APl": float(s[5]),
        }


print("\n========== A组：原图推理 ==========")
metrics_original = evaluate_coco(original_json)

print("\n========== B组：2×2切片推理 ==========")
metrics_tiled = evaluate_coco(tiled_json)

# 计算差值
delta = {
    k: round(metrics_tiled[k] - metrics_original[k], 6)
    for k in metrics_original
}

summary = {
    "model": str(WEIGHT),
    "image_count": len(records),
    "tile_overlap": OVERLAP,
    "imgsz": IMGSZ,
    "conf": CONF,
    "nms_iou": NMS_IOU,
    "original": metrics_original,
    "tiled_2x2": metrics_tiled,
    "delta": delta
}

summary_path = OUT / "metrics_summary.json"

with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4, ensure_ascii=False)

print("\n========== 最终对比 ==========")

for key in metrics_original:

    print(
        f"{key:12s} | "
        f"原图: {metrics_original[key]:.4f} | "
        f"切片: {metrics_tiled[key]:.4f} | "
        f"变化: {delta[key]:+.4f}"
    )

print("\n结果保存目录:", OUT)
print("指标文件:", summary_path)
print("实验完成")
