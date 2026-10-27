from ultralytics import YOLO


model_path = (
    "runs/detect/stage3_targeted/"
    "YOLO11s-DySample-Pretrained-LR003/"
    "weights/best.pt"
)

data_path = (
    "datasets/HighwayVehicle-v1.0/"
    "highway_vehicle_v1.0.yaml"
)

model = YOLO(model_path)


for imgsz in [640, 768, 896]:

    print("\n" + "=" * 60)
    print(f"Validation imgsz = {imgsz}")
    print("=" * 60)

    model.val(
        data=data_path,
        imgsz=imgsz,
        batch=16,
        device=0,
        workers=4,

        project="resolution_screening",
        name=f"LR003-val-{imgsz}",
    )