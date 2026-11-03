
import math
import random
import csv
from pathlib import Path

import yaml
from PIL import Image
from ultralytics.data.utils import check_det_dataset, img2label_paths

# =====================================================
# Configuration
# =====================================================

DATA_YAML = Path(
    "datasets/HighwayVehicle-v1.0/highway_vehicle_v1.0.yaml"
)

OUT_ROOT = Path(
    "datasets/HighwayVehicle-v1.0-crop20"
)

SEED = 42
IMG_SIZE = 640

SMALL_AREA = 32 * 32

CROP_MIN = 0.65
CROP_MAX = 0.80

CROP_RATIO = 0.25
MIN_VISIBLE = 0.60

IMG_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

random.seed(SEED)


# =====================================================
# Helpers
# =====================================================

def collect_images(directory):
    return sorted([
        p for p in Path(directory).rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    ])


def parse_labels(label_path, width, height):

    objects = []

    if not Path(label_path).exists():
        return objects

    for line in Path(label_path).read_text().splitlines():

        parts = line.split()

        if len(parts) < 5:
            continue

        cls = int(float(parts[0]))
        xc, yc, bw, bh = map(float, parts[1:5])

        x1 = (xc - bw / 2) * width
        y1 = (yc - bh / 2) * height
        x2 = (xc + bw / 2) * width
        y2 = (yc + bh / 2) * height

        objects.append({
            "cls": cls,
            "box": (x1, y1, x2, y2)
        })

    return objects


def is_small(box, width, height):

    x1, y1, x2, y2 = box

    scale = min(
        IMG_SIZE / width,
        IMG_SIZE / height
    )

    area = (
        (x2 - x1) * scale *
        (y2 - y1) * scale
    )

    return area < SMALL_AREA


def link_file(src, dst):

    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    if not dst.exists() and not dst.is_symlink():
        dst.symlink_to(Path(src).resolve())


# =====================================================
# Main
# =====================================================

def main():

    if OUT_ROOT.exists() and any(OUT_ROOT.iterdir()):
        raise RuntimeError(
            "输出目录非空。请先检查已有裁剪数据，"
            "不要直接覆盖。"
        )

    data = check_det_dataset(str(DATA_YAML))

    train_dir = Path(data["train"])

    if not train_dir.is_dir():
        raise RuntimeError(
            "当前脚本要求 train 指向 images/train 目录。"
            "请先检查数据 YAML。"
        )

    images = collect_images(train_dir)

    labels = img2label_paths(
        [str(p) for p in images]
    )

    if not images:
        raise RuntimeError("没有找到训练图片")

    candidates = []
    all_small_areas = []

    # -------------------------------------------------
    # 1. Find images containing small objects
    # -------------------------------------------------

    for image_path, label_path in zip(images, labels):

        with Image.open(image_path) as im:
            width, height = im.size

        objects = parse_labels(
            label_path, width, height
        )

        small_indices = []

        for i, obj in enumerate(objects):

            if is_small(obj["box"], width, height):
                small_indices.append(i)

                x1, y1, x2, y2 = obj["box"]

                scale = min(
                    IMG_SIZE / width,
                    IMG_SIZE / height
                )

                area = (
                    (x2 - x1) *
                    (y2 - y1) *
                    scale * scale
                )

                all_small_areas.append(area)

        if small_indices:

            candidates.append({
                "image": image_path,
                "label": Path(label_path),
                "objects": objects,
                "small_indices": small_indices
            })

    if not candidates:
        raise RuntimeError("没有发现小目标候选图片")

    # -------------------------------------------------
    # 2. Preserve all original images and labels
    # -------------------------------------------------

    for image_path, label_path in zip(images, labels):

        rel = image_path.relative_to(train_dir)

        out_img = OUT_ROOT / "images" / "train" / rel
        out_lab = (
            OUT_ROOT / "labels" / "train"
            / rel.with_suffix(".txt")
        )

        link_file(image_path, out_img)

        if Path(label_path).exists():
            link_file(label_path, out_lab)

    # -------------------------------------------------
    # 3. Select crop images
    # -------------------------------------------------

    target_count = math.ceil(
        len(images) * CROP_RATIO
    )

    selected = random.sample(
        candidates,
        min(target_count, len(candidates))
    )

    manifest = []

    # -------------------------------------------------
    # 4. Generate target-centered crops
    # -------------------------------------------------

    for idx, item in enumerate(selected):

        image_path = item["image"]
        objects = item["objects"]

        with Image.open(image_path) as im:
            image = im.convert("RGB")
            width, height = image.size

        target_idx = random.choice(
            item["small_indices"]
        )

        tx1, ty1, tx2, ty2 = (
            objects[target_idx]["box"]
        )

        ratio = random.uniform(
            CROP_MIN, CROP_MAX
        )

        crop_w = int(width * ratio)
        crop_h = int(height * ratio)

        # Ensure the selected target is fully included.

        left_min = max(
            0, math.ceil(tx2 - crop_w)
        )

        left_max = min(
            width - crop_w, math.floor(tx1)
        )

        top_min = max(
            0, math.ceil(ty2 - crop_h)
        )

        top_max = min(
            height - crop_h, math.floor(ty1)
        )

        if left_min > left_max or top_min > top_max:
            continue

        left = random.randint(left_min, left_max)
        top = random.randint(top_min, top_max)

        right = left + crop_w
        bottom = top + crop_h

        crop = image.crop(
            (left, top, right, bottom)
        )

        new_labels = []

        # -------------------------------------------------
        # 5. Transform all bounding boxes
        # -------------------------------------------------

        for obj in objects:

            x1, y1, x2, y2 = obj["box"]

            nx1 = max(x1, left) - left
            ny1 = max(y1, top) - top

            nx2 = min(x2, right) - left
            ny2 = min(y2, bottom) - top

            original_area = max(
                (x2 - x1) * (y2 - y1), 1e-9
            )

            visible_area = max(
                0, nx2 - nx1
            ) * max(
                0, ny2 - ny1
            )

            visible_ratio = (
                visible_area / original_area
            )

            if visible_ratio < MIN_VISIBLE:
                continue

            cx = (nx1 + nx2) / 2 / crop_w
            cy = (ny1 + ny2) / 2 / crop_h

            bw = (nx2 - nx1) / crop_w
            bh = (ny2 - ny1) / crop_h

            new_labels.append(
                f"{obj['cls']} "
                f"{cx:.6f} {cy:.6f} "
                f"{bw:.6f} {bh:.6f}"
            )

        if not new_labels:
            continue

        # -------------------------------------------------
        # 6. Save crop image and labels
        # -------------------------------------------------

        rel = image_path.relative_to(train_dir)

        crop_name = (
            rel.stem + f"__crop20_{idx:05d}"
            + rel.suffix.lower()
        )

        crop_rel = rel.with_name(crop_name)

        out_img = (
            OUT_ROOT / "images" / "train" / crop_rel
        )

        out_lab = (
            OUT_ROOT / "labels" / "train"
            / crop_rel.with_suffix(".txt")
        )

        out_img.parent.mkdir(
            parents=True, exist_ok=True
        )

        out_lab.parent.mkdir(
            parents=True, exist_ok=True
        )

        if crop_rel.suffix.lower() in {".jpg", ".jpeg"}:
            crop.save(out_img, quality=95)
        else:
            crop.save(out_img)

        out_lab.write_text(
            "\n".join(new_labels) + "\n"
        )

        manifest.append({
            "source_image": str(image_path),
            "crop_image": str(out_img),
            "crop_ratio": ratio,
            "selected_small_object": target_idx,
            "kept_boxes": len(new_labels)
        })

    # -------------------------------------------------
    # 7. Create dataset YAML
    # -------------------------------------------------

    new_data = {
        "path": str(OUT_ROOT.resolve()),
        "train": "images/train",
        "val": data["val"],
        "nc": data["nc"],
        "names": data["names"]
    }

    if "test" in data:
        new_data["test"] = data["test"]

    yaml_path = (
        OUT_ROOT / "highway_vehicle_crop20.yaml"
    )

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            new_data,
            f,
            allow_unicode=True,
            sort_keys=False
        )

    # -------------------------------------------------
    # 8. Save crop manifest
    # -------------------------------------------------

    manifest_path = (
        OUT_ROOT / "crop_manifest.csv"
    )

    with open(
        manifest_path, "w",
        newline="", encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_image",
                "crop_image",
                "crop_ratio",
                "selected_small_object",
                "kept_boxes"
            ]
        )

        writer.writeheader()
        writer.writerows(manifest)

    total = len(images) + len(manifest)

    print("\n========== Crop Dataset Summary ==========")
    print("Original images:", len(images))
    print("Small-object candidate images:", len(candidates))
    print("Generated crops:", len(manifest))
    print("Final images:", total)
    print(
        "Crop proportion:",
        round(len(manifest) / total * 100, 2), "%"
    )
    print("Small objects:", len(all_small_areas))
    print("Dataset YAML:", yaml_path)
    print("Manifest:", manifest_path)


if __name__ == "__main__":
    main()
