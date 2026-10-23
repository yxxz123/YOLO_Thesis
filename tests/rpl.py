import torch
from ultralytics import YOLO


yaml_path = (
    "ultralytics/cfg/models/improve/yolo11s-DySample-RPL.yaml"
)


# ==========================================================
# 1. Build model
# ==========================================================
model = YOLO(yaml_path)

net = model.model


# ==========================================================
# 2. Check detection head
# ==========================================================
print("\n========== LAST HEAD ==========")

print("Head type:")
print(type(net.model[-1]))

print("\nHead structure:")
print(net.model[-1])


# ==========================================================
# 3. Model information
# ==========================================================
print("\n========== MODEL INFO ==========")

net.info()


# ==========================================================
# 4. Device check
# ==========================================================
device = next(net.parameters()).device

print("\n========== DEVICE ==========")
print("Model device:", device)


# ==========================================================
# 5. Forward test
# ==========================================================
print("\n========== FORWARD TEST ==========")

net.eval()

# IMPORTANT:
# input tensor must be on the same device as the model
x = torch.randn(
    1,
    3,
    640,
    640,
    device=device
)

with torch.no_grad():
    y = net(x)


print("Forward success!")
print("Output type:", type(y))

if isinstance(y, tuple):

    print("Output tuple length:", len(y))

    if torch.is_tensor(y[0]):
        print("Output[0] shape:", y[0].shape)

    else:
        print("Output[0] type:", type(y[0]))

else:

    if torch.is_tensor(y):
        print("Output shape:", y.shape)

    else:
        print(y)


# ==========================================================
# 6. RPL zero initialization check
# ==========================================================
print("\n========== RPL ZERO INIT TEST ==========")

head = net.model[-1]

for i, branch in enumerate(head.cv2):

    weight_sum = (
        branch.refine_pred.weight
        .detach()
        .abs()
        .sum()
        .item()
    )

    if branch.refine_pred.bias is not None:

        bias_sum = (
            branch.refine_pred.bias
            .detach()
            .abs()
            .sum()
            .item()
        )

    else:

        bias_sum = 0.0

    alpha = branch.alpha.detach().item()

    print(
        f"P{i + 3}: "
        f"refine weight abs sum = {weight_sum:.8f}, "
        f"bias abs sum = {bias_sum:.8f}, "
        f"alpha = {alpha:.6f}"
    )


print("\n========== TEST FINISHED ==========")