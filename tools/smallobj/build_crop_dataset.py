import csv
import json
import math
import random
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

from ultralytics.data.utils import (
    check_det_dataset,
    img2label_paths
)


# ============================================================
# 1. Experiment configuration
# ============================================================

DATA_YAML = Path(
    "datasets/HighwayVehicle-v1.0/"
    "highway_vehicle_v1.0.yaml"
)

OUT_ROOT = Path(
    "datasets/HighwayVehicle-v1.0-crop20"
)

SEED = 42
IMG_SIZE = 640

# COCO-style area thresholds at 640 input scale
TINY_AREA = 16 ** 2
MID_AREA = 24 ** 2
SMALL_AREA = 32 ** 2

# Crop window ratio
CROP_MIN = 0.60
CROP_MAX = 0.70

# 25% additional crops relative to original images
# Final crop proportion = 25 / 125 = 20%
CROP_RATIO = 0.25

# Selected target must be completely preserved
TARGET_VISIBLE = 0.999

# Other objects: retain if visible area >= 70%
MIN_VISIBLE = 0.70

# Target allocation
TARGET_WEIGHTS = {
    "tiny": 0.75,
    "mid": 0.20,
    "normal_small": 0.05
}

PREVIEW_LIMIT = 50

IMG_EXTS = {
    ".jpg", ".jpeg", ".png",
    ".bmp", ".webp"
}

random.seed(SEED)


# ============================================================
# 2. Basic helpers
# ============================================================

def get_train_directory(data):

    train = data["train"]

    if isinstance(train, (list, tuple)):
        raise RuntimeError(
            "当前程序要求 train 指向单一图片目录，"
            "不支持多个训练目录或图片列表。"
        )

    path = Path(train)

    if not path.is_absolute():
        path = Path(data["path"]) / path

    path = path.resolve()

    if not path.is_dir():
        raise FileNotFoundError(
            f"训练图片目录不存在：{path}"
        )

    return path


def collect_images(directory):

    return sorted([
        p for p in Path(directory).rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMG_EXTS
    ])


def bucket_for_area(area):

    if area < TINY_AREA:
        return "tiny"

    elif area < MID_AREA:
        return "mid"

    elif area < SMALL_AREA:
        return "normal_small"

    return None


def absolute_dataset_path(value, base):

    if isinstance(value, (list, tuple)):
        return [
            absolute_dataset_path(v, base)
            for v in value
        ]

    path = Path(value)

    if not path.is_absolute():
        path = Path(base) / path

    return str(path.resolve())


def link_file(src, dst):

    src = Path(src)
    dst = Path(dst)

    dst.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if dst.exists() or dst.is_symlink():
        raise FileExistsError(dst)

    dst.symlink_to(src.resolve())


# ============================================================
# 3. Read YOLO labels and calculate object areas
# ============================================================

def parse_labels(label_path, width, height, nc):

    objects = []
    invalid_count = 0

    if not Path(label_path).exists():
        return objects, 0

    for line in Path(label_path).read_text().splitlines():

        parts = line.split()

        if len(parts) < 5:
            invalid_count += 1
            continue

        try:

            cls_float = float(parts[0])

            xc, yc, bw, bh = map(
                float, parts[1:5]
            )

        except ValueError:

            invalid_count += 1
            continue

        values = [cls_float, xc, yc, bw, bh]

        if not all(math.isfinite(v) for v in values):
            invalid_count += 1
            continue

        cls = int(cls_float)

        if cls_float != cls or not (0 <= cls < nc):
            invalid_count += 1
            continue

        if bw <= 0 or bh <= 0:
            invalid_count += 1
            continue

        if (
            xc < -1e-4 or xc > 1.0001
            or yc < -1e-4 or yc > 1.0001
            or bw > 1.0001 or bh > 1.0001
        ):
            invalid_count += 1
            continue

        x1 = max(
            0.0,
            (xc - bw / 2) * width
        )

        y1 = max(
            0.0,
            (yc - bh / 2) * height
        )

        x2 = min(
            float(width),
            (xc + bw / 2) * width
        )

        y2 = min(
            float(height),
            (yc + bh / 2) * height
        )

        if x2 <= x1 or y2 <= y1:
            invalid_count += 1
            continue

        scale = min(
            IMG_SIZE / width,
            IMG_SIZE / height
        )

        area640 = (
            (x2 - x1)
            * (y2 - y1)
            * scale ** 2
        )

        bucket = bucket_for_area(area640)

        objects.append({
            "cls": cls,
            "box": (x1, y1, x2, y2),
            "area640": area640,
            "bucket": bucket
        })

    return objects, invalid_count


# ============================================================
# 4. Stratified image selection
# ============================================================

def select_candidates(candidates, target_count):

    quotas = {
        "tiny": round(target_count * 0.75),
        "mid": round(target_count * 0.20)
    }

    quotas["normal_small"] = (
        target_count
        - quotas["tiny"]
        - quotas["mid"]
    )

    actual = {
        "tiny": 0,
        "mid": 0,
        "normal_small": 0
    }

    selected = []
    used_images = set()

    # Select rare categories first.
    # This prevents common tiny-object images
    # from consuming images needed by other groups.

    for bucket in [
        "normal_small",
        "mid",
        "tiny"
    ]:

        eligible = [
            item for item in candidates
            if item["bucket_indices"][bucket]
            and str(item["image"]) not in used_images
        ]

        random.shuffle(eligible)

        need = quotas[bucket]

        for item in eligible[:need]:

            target_idx = random.choice(
                item["bucket_indices"][bucket]
            )

            selected.append({
                "item": item,
                "target_idx": target_idx,
                "planned_bucket": bucket
            })

            used_images.add(str(item["image"]))
            actual[bucket] += 1

    # Fill any remaining quota from unused images.
    # Prefer the category with the largest remaining deficit.

    while len(selected) < target_count:

        remaining = [
            item for item in candidates
            if str(item["image"]) not in used_images
        ]

        if not remaining:
            break

        available = {}

        for bucket in [
            "tiny",
            "mid",
            "normal_small"
        ]:

            deficit = quotas[bucket] - actual[bucket]

            if deficit <= 0:
                continue

            pool = [
                item for item in remaining
                if item["bucket_indices"][bucket]
            ]

            if pool:
                available[bucket] = pool

        if available:

            bucket = max(
                available,
                key=lambda b: quotas[b] - actual[b]
            )

            item = random.choice(available[bucket])

            target_idx = random.choice(
                item["bucket_indices"][bucket]
            )

        else:

            item = random.choice(remaining)

            target_idx = random.choice(
                item["small_indices"]
            )

            bucket = item["objects"][target_idx]["bucket"]

        selected.append({
            "item": item,
            "target_idx": target_idx,
            "planned_bucket": bucket
        })

        used_images.add(str(item["image"]))

        if bucket in actual:
            actual[bucket] += 1

    random.shuffle(selected)

    return selected, quotas, actual


# ============================================================
# 5. Target-centered crop geometry
# ============================================================

def sample_crop_rect(width, height, box):

    x1, y1, x2, y2 = box

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    for _ in range(30):

        ratio = random.uniform(
            CROP_MIN,
            CROP_MAX
        )

        crop_w = max(
            2,
            min(width - 1, round(width * ratio))
        )

        crop_h = max(
            2,
            min(height - 1, round(height * ratio))
        )

        if (
            crop_w < x2 - x1
            or crop_h < y2 - y1
        ):
            continue

        # Feasible interval guarantees
        # the selected target remains fully inside.

        left_min = max(
            0,
            math.ceil(x2 - crop_w)
        )

        left_max = min(
            width - crop_w,
            math.floor(x1)
        )

        top_min = max(
            0,
            math.ceil(y2 - crop_h)
        )

        top_max = min(
            height - crop_h,
            math.floor(y1)
        )

        if (
            left_min > left_max
            or top_min > top_max
        ):
            continue

        # Keep the selected object near the center.
        # Randomize its position between 40% and 60%.

        pos_x = random.uniform(0.40, 0.60)
        pos_y = random.uniform(0.40, 0.60)

        left = round(cx - pos_x * crop_w)
        top = round(cy - pos_y * crop_h)

        left = max(
            left_min,
            min(left_max, left)
        )

        top = max(
            top_min,
            min(top_max, top)
        )

        return (
            left,
            top,
            left + crop_w,
            top + crop_h,
            ratio
        )

    return None


# ============================================================
# 6. Main dataset generation
# ============================================================

def main():

    if not DATA_YAML.is_file():
        raise FileNotFoundError(DATA_YAML)

    OUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    if any(OUT_ROOT.iterdir()):
        raise RuntimeError(
            f"输出目录非空：{OUT_ROOT}\n"
            "请先检查原有数据，不要直接覆盖。"
        )

    data = check_det_dataset(str(DATA_YAML))

    train_dir = get_train_directory(data)

    images = collect_images(train_dir)

    if not images:
        raise RuntimeError("没有找到训练图片")

    nc = int(data["nc"])

    label_paths = img2label_paths(
        [str(p) for p in images]
    )

    print("========== Original Dataset ==========")
    print("Training images:", len(images))
    print("Training directory:", train_dir)

    # --------------------------------------------------------
    # A. Parse all labels and find candidates
    # --------------------------------------------------------

    candidates = []

    total_objects = 0
    total_small = 0
    invalid_labels = 0

    size_counts = Counter()
    image_small_counts = Counter()

    for image_path, label_path in zip(
        images, label_paths
    ):

        with Image.open(image_path) as im:
            width, height = im.size

        objects, invalid = parse_labels(
            label_path,
            width,
            height,
            nc
        )

        invalid_labels += invalid
        total_objects += len(objects)

        bucket_indices = {
            "tiny": [],
            "mid": [],
            "normal_small": []
        }

        small_indices = []

        for i, obj in enumerate(objects):

            bucket = obj["bucket"]

            if bucket is None:
                size_counts["large"] += 1
                continue

            small_indices.append(i)
            total_small += 1
            size_counts[bucket] += 1

            bucket_indices[bucket].append(i)

        image_small_counts[
            sum(len(v) for v in bucket_indices.values())
        ] += 1

        if small_indices:

            candidates.append({
                "image": image_path,
                "label": Path(label_path),
                "objects": objects,
                "small_indices": small_indices,
                "bucket_indices": bucket_indices
            })

    print("\n========== Object Statistics ==========")
    print("Total objects:", total_objects)
    print("Small objects:", total_small)
    print("Invalid label lines:", invalid_labels)
    print("Candidate images:", len(candidates))

    print("\nSize groups:")

    for bucket in [
        "tiny",
        "mid",
        "normal_small"
    ]:
        print(
            bucket,
            size_counts[bucket]
        )

    if not candidates:
        raise RuntimeError("没有找到小目标候选图片")

    # --------------------------------------------------------
    # B. Calculate crop amount
    # --------------------------------------------------------

    target_count = math.ceil(
        len(images) * CROP_RATIO
    )

    target_count = min(
        target_count,
        len(candidates)
    )

    selected, quotas, actual = select_candidates(
        candidates,
        target_count
    )

    print("\n========== Sampling Plan ==========")
    print("Requested crops:", target_count)
    print("Target quotas:", quotas)
    print("Initial selected distribution:", actual)

    # --------------------------------------------------------
    # C. Preserve all original images and labels
    # --------------------------------------------------------

    print("\nLinking original training data...")

    for image_path, label_path in zip(
        images, label_paths
    ):

        rel = image_path.relative_to(train_dir)

        out_img = (
            OUT_ROOT / "images" / "train" / rel
        )

        out_label = (
            OUT_ROOT / "labels" / "train"
            / rel.with_suffix(".txt")
        )

        link_file(image_path, out_img)

        if Path(label_path).is_file():
            link_file(label_path, out_label)

        else:
            out_label.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            out_label.write_text("")

    # --------------------------------------------------------
    # D. Generate crop images and labels
    # --------------------------------------------------------

    manifest = []

    preview_dir = OUT_ROOT / "preview"
    preview_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    skipped = 0

    actual_generated = Counter()

    for idx, record in enumerate(selected):

        item = record["item"]
        planned_idx = record["target_idx"]

        image_path = item["image"]
        objects = item["objects"]

        with Image.open(image_path) as im:
            image = im.convert("RGB")

        width, height = image.size

        # Try the planned target first.
        # If its geometry is unsuitable, try another
        # small target in the same image.

        target_candidates = [planned_idx]

        same_bucket = item["bucket_indices"][
            record["planned_bucket"]
        ]

        target_candidates.extend([
            i for i in same_bucket
            if i != planned_idx
        ])

        target_candidates.extend([
            i for i in item["small_indices"]
            if i not in target_candidates
        ])

        crop_rect = None
        target_idx = None

        for candidate_idx in target_candidates:

            rect = sample_crop_rect(
                width,
                height,
                objects[candidate_idx]["box"]
            )

            if rect is not None:

                crop_rect = rect
                target_idx = candidate_idx
                break

        if crop_rect is None:

            skipped += 1
            continue

        left, top, right, bottom, ratio = crop_rect

        crop = image.crop(
            (left, top, right, bottom)
        )

        crop_w = right - left
        crop_h = bottom - top

        target_bucket = objects[
            target_idx
        ]["bucket"]

        new_labels = []
        kept_boxes = []

        retained_groups = Counter()

        scale_crop = min(
            IMG_SIZE / crop_w,
            IMG_SIZE / crop_h
        )

        # ----------------------------------------------------
        # Transform every original bounding box
        # ----------------------------------------------------

        for obj_idx, obj in enumerate(objects):

            x1, y1, x2, y2 = obj["box"]

            nx1 = max(x1, left)
            ny1 = max(y1, top)

            nx2 = min(x2, right)
            ny2 = min(y2, bottom)

            inter_w = max(0, nx2 - nx1)
            inter_h = max(0, ny2 - ny1)

            original_area = max(
                (x2 - x1) * (y2 - y1),
                1e-12
            )

            visible_area = inter_w * inter_h

            visible_ratio = (
                visible_area / original_area
            )

            if visible_ratio < MIN_VISIBLE:
                continue

            if inter_w <= 0.5 or inter_h <= 0.5:
                continue

            # Selected target must be fully retained.
            if obj_idx == target_idx:

                if visible_ratio < TARGET_VISIBLE:
                    raise RuntimeError(
                        "中心目标没有被完整保留"
                    )

            local_x1 = nx1 - left
            local_y1 = ny1 - top
            local_x2 = nx2 - left
            local_y2 = ny2 - top

            cx = (
                (local_x1 + local_x2) / 2
            ) / crop_w

            cy = (
                (local_y1 + local_y2) / 2
            ) / crop_h

            bw = (
                local_x2 - local_x1
            ) / crop_w

            bh = (
                local_y2 - local_y1
            ) / crop_h

            new_labels.append(
                f"{obj['cls']} "
                f"{cx:.6f} {cy:.6f} "
                f"{bw:.6f} {bh:.6f}"
            )

            kept_boxes.append((
                local_x1,
                local_y1,
                local_x2,
                local_y2
            ))

            area640 = (
                (local_x2 - local_x1)
                * (local_y2 - local_y1)
                * scale_crop ** 2
            )

            bucket = bucket_for_area(area640)

            if bucket is not None:
                retained_groups[bucket] += 1

        if not new_labels:
            skipped += 1
            continue

        if len(kept_boxes) <= 0:
            skipped += 1
            continue

        # ----------------------------------------------------
        # Save crop image and labels
        # ----------------------------------------------------

        rel = image_path.relative_to(train_dir)

        crop_name = (
            f"{rel.stem}__crop20_{idx:05d}.jpg"
        )

        crop_rel = rel.parent / crop_name

        out_img = (
            OUT_ROOT / "images" / "train" / crop_rel
        )

        out_label = (
            OUT_ROOT / "labels" / "train"
            / crop_rel.with_suffix(".txt")
        )

        out_img.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        out_label.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        crop.save(
            out_img,
            format="JPEG",
            quality=95
        )

        out_label.write_text(
            "\n".join(new_labels) + "\n"
        )

        # ----------------------------------------------------
        # Save visual previews
        # ----------------------------------------------------

        if len(manifest) < PREVIEW_LIMIT:

            preview = crop.copy()
            draw = ImageDraw.Draw(preview)

            for box in kept_boxes:

                draw.rectangle(
                    box,
                    outline="lime",
                    width=2
                )

            preview.save(
                preview_dir / f"{idx:05d}.jpg",
                quality=95
            )

        # ----------------------------------------------------
        # Record experiment information
        # ----------------------------------------------------

        target_box = objects[
            target_idx
        ]["box"]

        target_cx = (
            (target_box[0] + target_box[2]) / 2
            - left
        ) / crop_w

        target_cy = (
            (target_box[1] + target_box[3]) / 2
            - top
        ) / crop_h

        actual_generated[target_bucket] += 1

        manifest.append({

            "source_image": str(image_path),

            "crop_image": str(out_img),

            "planned_bucket": record["planned_bucket"],

            "actual_bucket": target_bucket,

            "target_area640": round(
                objects[target_idx]["area640"], 2
            ),

            "crop_ratio": round(ratio, 4),

            "target_x_percent": round(
                target_cx * 100, 2
            ),

            "target_y_percent": round(
                target_cy * 100, 2
            ),

            "objects_before": len(objects),

            "objects_after": len(new_labels),

            "retained_tiny": retained_groups["tiny"],

            "retained_mid": retained_groups["mid"],

            "retained_normal_small": (
                retained_groups["normal_small"]
            )
        })

    # --------------------------------------------------------
    # E. Save crop manifest
    # --------------------------------------------------------

    if not manifest:
        raise RuntimeError(
            "没有成功生成裁剪图片，请检查目标框和裁剪参数。"
        )

    manifest_path = (
        OUT_ROOT / "crop_manifest.csv"
    )

    with open(
        manifest_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(manifest[0].keys())
        )

        writer.writeheader()
        writer.writerows(manifest)

    # --------------------------------------------------------
    # F. Create new dataset YAML
    # --------------------------------------------------------

    base_path = data.get("path", ".")

    new_data = {

        "path": str(OUT_ROOT.resolve()),

        "train": "images/train",

        "val": absolute_dataset_path(
            data["val"],
            base_path
        ),

        "nc": data["nc"],

        "names": data["names"]
    }

    if "test" in data:

        new_data["test"] = absolute_dataset_path(
            data["test"],
            base_path
        )

    yaml_path = (
        OUT_ROOT / "highway_vehicle_crop20.yaml"
    )

    with open(
        yaml_path,
        "w",
        encoding="utf-8"
    ) as f:

        yaml.safe_dump(
            new_data,
            f,
            allow_unicode=True,
            sort_keys=False
        )

    # --------------------------------------------------------
    # G. Validate generated labels
    # --------------------------------------------------------

    crop_images = list(
        (OUT_ROOT / "images" / "train").rglob(
            "*__crop20_*.jpg"
        )
    )

    crop_label_count = 0
    invalid_crop_labels = 0

    for crop_image in crop_images:

        label = (
            OUT_ROOT / "labels" / "train"
            / crop_image.relative_to(
                OUT_ROOT / "images" / "train"
            ).with_suffix(".txt")
        )

        if not label.is_file():
            invalid_crop_labels += 1
            continue

        crop_label_count += 1

        for line in label.read_text().splitlines():

            parts = line.split()

            if len(parts) != 5:
                invalid_crop_labels += 1
                continue

            try:
                values = list(
                    map(float, parts[1:5])
                )
            except ValueError:
                invalid_crop_labels += 1
                continue

            if not all(
                math.isfinite(v)
                and 0 <= v <= 1
                for v in values
            ):
                invalid_crop_labels += 1

    # --------------------------------------------------------
    # H. Save final summary
    # --------------------------------------------------------

    final_images = len(images) + len(manifest)

    summary = {

        "original_images": len(images),

        "candidate_images": len(candidates),

        "requested_crops": target_count,

        "generated_crops": len(manifest),

        "skipped_crops": skipped,

        "final_images": final_images,

        "crop_proportion": (
            len(manifest) / final_images
        ),

        "planned_quotas": quotas,

        "generated_distribution": dict(
            actual_generated
        ),

        "invalid_source_label_lines": invalid_labels,

        "crop_label_files": crop_label_count,

        "invalid_crop_labels": invalid_crop_labels,

        "yaml": str(yaml_path),

        "seed": SEED
    }

    with open(
        OUT_ROOT / "crop_summary.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=4
        )

    print("\n========== FINAL CROP SUMMARY ==========")

    for key, value in summary.items():
        print(f"{key}: {value}")

    if invalid_crop_labels:
        raise RuntimeError(
            "裁剪标签检查未通过，禁止开始训练。"
        )

    if skipped:
        print(
            "\nWARNING: 部分候选裁剪未生成，"
            "请检查 crop_summary.json 和 crop_manifest.csv。"
        )

    print("\nCrop dataset generation finished.")


if __name__ == "__main__":
    main()