import warnings, os, sys

# 硬件安全配置：指定使用第一张显卡
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')

from ultralytics import YOLO

if __name__ == '__main__':

    yaml_path = 'ultralytics/cfg/models/improve/upsample/yolo11/yolo11s-DySample.yaml'

    # 初始化 YOLO 模型，从 yaml 配置文件构建网络结构
    model = YOLO(yaml_path)
    model.train(
    data='datasets/HighwayVehicle-v1.0/highway_vehicle_v1.0.yaml',
    epochs=150,
    imgsz=640,
    batch=16,
    device=0,
    workers=4,
    optimizer="AdamW",
    cos_lr=True,
    mosaic=0.5,
    close_mosaic=10,
    amp=True,
    cache=False,
    patience=150,
    seed=42,
    deterministic=True,
    # 保存中间权重
    save=True,
    save_period=10,
    project="stage4_smallobj",
    name="YOLO11s-DySample-Mosaic050",
)