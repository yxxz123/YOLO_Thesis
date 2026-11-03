
from pathlib import Path
import csv

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

from ultralytics.data.utils import (
    check_det_dataset,
    img2label_paths
)

DATA = "datasets/HighwayVehicle-v1.0/highway_vehicle_v1.0.yaml"

OUT = Path("runs/detect/stage5_smallobj_crop/analysis")

IMG_SIZE = 640
SMALL_AREA = 32 ** 2

IMG_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}


def main():

    OUT.mkdir(parents=True, exist_ok=True)

    data = check_det_dataset(DATA)
    train_dir = Path(data["train"])

    if not train_dir.is_dir():
        raise RuntimeError("train路径不是图片目录")

    images = sorted([
        p for p in train_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    ])

    labels = img2label_paths(
        [str(p) for p in images]
    )

    image_records = []
    all_areas = []
    examples = 0

    for image_path, label_path in zip(images, labels):

        with Image.open(image_path) as im:
            width, height = im.size
            visual = im.convert("RGB").copy()

        scale = min(
            IMG_SIZE / width,
            IMG_SIZE / height
        )

        small_count = 0
        total_count = 0
        small_areas = []

        if Path(label_path).exists():

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

                area = (
                    (x2 - x1) *
                    (y2 - y1) *
                    scale ** 2
                )

                total_count += 1
                all_areas.append(area)

                if area < SMALL_AREA:

                    small_count += 1
                    small_areas.append(area)

                    # Draw small-object boxes
                    draw = ImageDraw.Draw(visual)

                    draw.rectangle(
                        (x1, y1, x2, y2),
                        outline="red",
                        width=2
                    )

        image_records.append({
            "image": str(image_path),
            "width": width,
            "height": height,
            "total_objects": total_count,
            "small_objects": small_count,
            "min_small_area": (
                min(small_areas) if small_areas else ""
            )
        })

        if small_count > 0 and examples < 30:

            visual.save(
                OUT / "smallobj_examples"
                / f"{examples:03d}_{image_path.name}"
            ) if (OUT / "smallobj_examples").exists() else None

            if not (OUT / "smallobj_examples").exists():
                (OUT / "smallobj_examples").mkdir(
                    parents=True, exist_ok=True
                )
                visual.save(
                    OUT / "smallobj_examples"
                    / f"{examples:03d}_{image_path.name}"
                )

            examples += 1

    # Per-image statistics
    with open(
        OUT / "image_smallobj_count.csv",
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(image_records[0].keys())
        )

        writer.writeheader()
        writer.writerows(image_records)

    # Area distribution
    bins = [
        0, 16**2, 24**2, 32**2,
        48**2, 64**2, 96**2, float("inf")
    ]

    bin_names = [
        "0-16^2",
        "16^2-24^2",
        "24^2-32^2",
        "32^2-48^2",
        "48^2-64^2",
        "64^2-96^2",
        ">96^2"
    ]

    distribution = []

    for i, name in enumerate(bin_names):

        count = sum(
            bins[i] <= a < bins[i + 1]
            for a in all_areas
        )

        distribution.append({
            "area_range": name,
            "count": count
        })

    with open(
        OUT / "train_size_distribution.csv",
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=["area_range", "count"]
        )

        writer.writeheader()
        writer.writerows(distribution)

    # Histogram
    plt.figure(figsize=(9, 5))
    plt.hist(
        [a for a in all_areas if a < 5000],
        bins=30
    )
    plt.xlabel("Object area after 640 scaling")
    plt.ylabel("Number of objects")
    plt.title("Training Object Size Distribution")
    plt.tight_layout()

    plt.savefig(
        OUT / "small_object_area_hist.png",
        dpi=200
    )

    plt.close()

    small_total = sum(
        1 for a in all_areas if a < SMALL_AREA
    )

    candidate_images = sum(
        r["small_objects"] > 0
        for r in image_records
    )

    print("\n========== Analysis Summary ==========")
    print("Training images:", len(images))
    print("Total objects:", len(all_areas))
    print("Small objects:", small_total)
    print("Small-object image count:", candidate_images)
    print(
        "Small-object proportion:",
        round(small_total / max(len(all_areas), 1) * 100, 2),
        "%"
    )
    print("Output:", OUT)


if __name__ == "__main__":
    main()
