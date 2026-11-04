import os
import math
import argparse
from pathlib import Path

# ============================================================
# Stage 5: Small Object Crop Fine-tuning
# YOLO11s-DySample
#
# Control: Original dataset, 40 epochs
# Crop20: Original + crop dataset, 32 epochs
#
# Direct model.train(), no auto screening.
# ============================================================

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
import ultralytics

from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset


# ============================================================
# 1. Paths
# ============================================================

ROOT = Path(__file__).resolve().parent

BASE_WEIGHTS = (
    ROOT / "runs/detect/stage2_screening/"
    "YOLO11s-DySample/weights/best.pt"
)

ORIGINAL_YAML = (
    ROOT / "datasets/HighwayVehicle-v1.0/"
    "highway_vehicle_v1.0.yaml"
)

CROP_YAML = (
    ROOT / "datasets/HighwayVehicle-v1.0-crop20/"
    "highway_vehicle_crop20.yaml"
)

PROJECT = ROOT / "runs/detect/stage5_smallobj_crop"


# ============================================================
# 2. Fixed experiment parameters
# ============================================================

IMG_SIZE = 640
BATCH_SIZE = 16
WORKERS = 4

CONTROL_EPOCHS = 40
CROP_EPOCHS = 32

CONTROL_WARMUP = 3.0
CROP_WARMUP = 2.4

LR0 = 0.001
LRF = 0.01

MOMENTUM = 0.937
WEIGHT_DECAY = 0.0005
NBS = 64

SEED = 42

ORIGINAL_EXPECTED = 5048
CROP_EXPECTED = 6310

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png",
    ".bmp", ".webp"
}


# ============================================================
# 3. Dataset utilities
# ============================================================

def resolve_paths(value):

    values = value if isinstance(
        value, (list, tuple)
    ) else [value]

    result = []

    for item in values:

        p = Path(str(item))

        if not p.is_absolute():
            p = ROOT / p

        result.append(p.resolve())

    return result


def count_images(directory):

    directory = Path(directory)

    if not directory.is_dir():
        raise FileNotFoundError(directory)

    return sum(
        1 for p in directory.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_dataset(yaml_path):

    if not yaml_path.is_file():
        raise FileNotFoundError(yaml_path)

    return check_det_dataset(str(yaml_path))


# ============================================================
# 4. Verify datasets
# ============================================================

def verify_datasets():

    print("\n========== DATASET VERIFICATION ==========")

    original = load_dataset(ORIGINAL_YAML)
    crop = load_dataset(CROP_YAML)

    original_train = resolve_paths(
        original["train"]
    )

    crop_train = resolve_paths(
        crop["train"]
    )

    original_val = resolve_paths(
        original["val"]
    )

    crop_val = resolve_paths(
        crop["val"]
    )

    n_original = sum(
        count_images(p) for p in original_train
    )

    n_crop = sum(
        count_images(p) for p in crop_train
    )

    print("Original images:", n_original)
    print("Crop20 images:", n_crop)

    print("Original validation:", original_val)
    print("Crop20 validation:", crop_val)

    print("Original classes:", original["names"])
    print("Crop20 classes:", crop["names"])

    if n_original != ORIGINAL_EXPECTED:
        raise RuntimeError(
            f"Original image count error: {n_original}"
        )

    if n_crop != CROP_EXPECTED:
        raise RuntimeError(
            f"Crop image count error: {n_crop}"
        )

    if original["nc"] != crop["nc"]:
        raise RuntimeError("Class number mismatch.")

    if original["names"] != crop["names"]:
        raise RuntimeError("Class names mismatch.")

    if original_val != crop_val:
        raise RuntimeError(
            "Control and Crop20 validation sets differ."
        )

    expected_val = (
        ROOT / "datasets/HighwayVehicle-v1.0/images/val"
    ).resolve()

    if expected_val not in original_val:
        raise RuntimeError(
            "Unexpected validation dataset path."
        )

    print("\nDataset verification PASSED.")

    return n_original, n_crop


# ============================================================
# 5. Experiment configuration
# ============================================================

def get_config(group):

    original_batches = math.ceil(
        ORIGINAL_EXPECTED / BATCH_SIZE
    )

    crop_batches = math.ceil(
        CROP_EXPECTED / BATCH_SIZE
    )

    if group == "control":

        return {
            "data": str(ORIGINAL_YAML),
            "epochs": CONTROL_EPOCHS,
            "warmup_epochs": CONTROL_WARMUP,
            "close_mosaic": 3,
            "batches_per_epoch": original_batches,
            "name": "control_original_ft"
        }

    if group == "crop":

        return {
            "data": str(CROP_YAML),
            "epochs": CROP_EPOCHS,
            "warmup_epochs": CROP_WARMUP,
            "close_mosaic": 2,
            "batches_per_epoch": crop_batches,
            "name": "crop20_ft"
        }

    raise ValueError(group)


# ============================================================
# 6. Training
# ============================================================

def train_model(group, check_only=False):

    print("\n========== ENVIRONMENT ==========")

    print("Ultralytics:", ultralytics.__version__)
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable.")

    print("GPU:", torch.cuda.get_device_name(0))

    if not BASE_WEIGHTS.is_file():
        raise FileNotFoundError(
            f"Baseline weights not found: {BASE_WEIGHTS}"
        )

    n_original, n_crop = verify_datasets()

    cfg = get_config(group)

    epochs = cfg["epochs"]
    warmup_epochs = cfg["warmup_epochs"]
    close_mosaic = cfg["close_mosaic"]

    total_batches = (
        epochs * cfg["batches_per_epoch"]
    )

    run_dir = PROJECT / cfg["name"]

    print("\n========== STAGE 5 CONFIGURATION ==========")

    print("Experiment:", group)
    print("Initial weights:", BASE_WEIGHTS)
    print("Dataset:", cfg["data"])

    print("Epochs:", epochs)
    print("Batch:", BATCH_SIZE)
    print("Image size:", IMG_SIZE)

    print("Batches per epoch:", cfg["batches_per_epoch"])
    print("Total batch iterations:", total_batches)

    print("Optimizer: AdamW")
    print("LR0:", LR0)
    print("LRF:", LRF)
    print("Momentum:", MOMENTUM)
    print("Weight decay:", WEIGHT_DECAY)
    print("NBS:", NBS)

    print("Warmup epochs:", warmup_epochs)
    print("Cosine LR:", True)

    # Match original DySample augmentation
    print("Mosaic:", 1.0)
    print("Close Mosaic:", close_mosaic)
    print("Scale:", 0.5)

    print("Translate:", 0.1)
    print("Degrees:", 0.0)
    print("Shear:", 0.0)
    print("Perspective:", 0.0)

    print("Flip LR:", 0.5)
    print("Flip UD:", 0.0)

    print("Mixup:", 0.0)
    print("Copy Paste:", 0.0)

    print("Seed:", SEED)
    print("Deterministic:", True)
    print("Workers:", WORKERS)
    print("AMP:", True)

    print("Output:", run_dir)

    if check_only:
        print("\nCHECK ONLY: training not started.")
        return

    if run_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {run_dir}"
        )

    # Both groups start from the SAME original checkpoint.
    model = YOLO(str(BASE_WEIGHTS))

    if model.task != "detect":
        raise RuntimeError(
            f"Unexpected model task: {model.task}"
        )

    print("\nOriginal YOLO11s-DySample loaded.")

    # --------------------------------------------------------
    # Direct model.train()
    # --------------------------------------------------------

    model.train(

        # Dataset
        data=cfg["data"],

        # Training duration
        epochs=epochs,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,

        # Hardware
        device=0,
        workers=WORKERS,

        # Optimizer
        optimizer="AdamW",
        lr0=LR0,
        lrf=LRF,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        nbs=NBS,

        # Learning rate schedule
        cos_lr=True,

        warmup_epochs=warmup_epochs,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,

        # Augmentation: match original args.yaml
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,

        fliplr=0.5,
        flipud=0.0,

        mosaic=1.0,
        close_mosaic=close_mosaic,

        mixup=0.0,
        cutmix=0.0,
        copy_paste=0.0,
        bgr=0.0,

        # Reproducibility
        seed=SEED,
        deterministic=True,

        # Training stability
        amp=True,
        cache=False,

        # Force completion of all epochs
        patience=epochs + 1,

        # Validation
        val=True,
        split="val",
        plots=True,
        save_json=False,
        auto_coco_eval=False,

        conf=None,
        iou=0.7,
        max_det=300,
        half=False,
        augment=False,

        # Data loading
        rect=False,
        multi_scale=0.0,
        fraction=1.0,

        # Saving
        save=True,
        save_period=-1,

        # Output
        project=str(PROJECT),
        name=cfg["name"],
        exist_ok=False,

        resume=False,
        verbose=True
    )

    # --------------------------------------------------------
    # Final output verification
    # --------------------------------------------------------

    print("\n========== TRAINING FINISHED ==========")

    for p in [
        run_dir / "weights/best.pt",
        run_dir / "weights/last.pt",
        run_dir / "results.csv"
    ]:

        print(
            p,
            "EXISTS" if p.exists() else "MISSING"
        )


# ============================================================
# 7. Main
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="YOLO11s-DySample Stage5 Fine-tuning"
    )

    parser.add_argument(
        "--group",
        required=True,
        choices=["control", "crop"]
    )

    parser.add_argument(
        "--check-only",
        action="store_true"
    )

    args = parser.parse_args()

    train_model(
        group=args.group,
        check_only=args.check_only
    )