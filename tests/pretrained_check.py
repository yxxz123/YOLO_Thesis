import torch
from ultralytics import YOLO


yaml_path = (
    "ultralytics/cfg/models/improve/upsample/"
    "yolo11/yolo11s-DySample.yaml"
)

weights_path = "yolo11s.pt"


# =========================================================
# 1. Build DySample model from YAML
# =========================================================
model = YOLO(yaml_path)

print("\n========== BEFORE LOAD ==========")

w_before = (
    model.model.model[0]
    .conv.weight
    .detach()
    .clone()
)

print(
    "First conv mean:",
    w_before.mean().item()
)

print(
    "First conv std:",
    w_before.std().item()
)


# =========================================================
# 2. Load official YOLO11s pretrained weights
# =========================================================
print("\n========== LOAD PRETRAINED ==========")

model.load(weights_path)


# =========================================================
# 3. Check first convolution
# =========================================================
w_after = (
    model.model.model[0]
    .conv.weight
    .detach()
    .clone()
)

diff = (
    w_after - w_before
).abs().sum().item()

print("\n========== CHECK ==========")

print(
    "Weight difference:",
    diff
)

print(
    "After mean:",
    w_after.mean().item()
)

print(
    "After std:",
    w_after.std().item()
)


if diff > 0:
    print(
        "\nSUCCESS: pretrained weights were loaded."
    )
else:
    print(
        "\nWARNING: pretrained weights may NOT have loaded."
    )


# =========================================================
# 4. Check DySample itself
# =========================================================
print("\n========== DYSAMPLE ==========")

print(model.model.model[11])