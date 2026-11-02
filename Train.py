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
    # model.load("yolo11s.pt")
    model.train(
    data='datasets/HighwayVehicle-v1.0/highway_vehicle_v1.0.yaml',
    epochs=150,
    imgsz=768,
    batch=16,
    device=0,
    workers=4,
    optimizer="AdamW",
    cos_lr=True,
    close_mosaic=10,
    amp=True,
    cache=False,
    patience=150,
    seed=42,
    deterministic=True,
    project="stage3_fast_screen",
    name="YOLO11s-DySample-768",
)

    # model.train(
    #     # ---------- 基础环境与数据配置 ----------
    #     data='highway_vehicle.yaml',  # 指向你处理好的 VisDrone 巡检数据集配置文件
    #     cache=False,  # 保持 False，防止服务器内存爆满
    #     imgsz=640,  # 统一输入分辨率，保证对比实验的公平性
    #     epochs=30,  # 论文标准迭代轮数
    #     batch=16,  # YOLO11n 是轻量级模型，RTX 3090 显存绝对安全，16 跑得又快又稳
    #     close_mosaic=10,  # 建议最后 10 轮关闭 Mosaic，让模型看看真实且未被切割的巡检图像，提点效果很好
    #     device=0,
    #     patience=50,
    #     optimizer='AdamW',  # 核心修改 2：YOLO11n 官方推荐使用 AdamW 优化器，收敛更平滑
    #     amp=False,  # 核心修改 3：强制关闭混合精度，彻底杜绝之前坑过你的 CUBLAS 报错和显存翻倍问题
    #     cos_lr=True,  # 开启余弦退火学习率，让模型在后期寻找最优解时更细腻
    #     project='Baseline_Results',
    #     name='yolo11+repvit+carfe',
    #     # ---------- 核心创新：无人机视角专属 Loss 组合 ----------
    #     # 核心修改 4：结合你的研究方案，量身定制的损失函数
    #     # cls_loss='varifocal',  # 分类：变焦损失。强力压制 VisDrone 中海量的天空/公路背景负样本
    #     # iou_loss='wiseiou',  # 回归：Shape-IoU。针对车辆目标自身形状特征进行优化，完美契合你的预定研究方案
    #     # iou_aux='nwd',  # 辅助：NWD。将微小目标转化为二维高斯分布计算距离，高速巡检小目标检测的顶级杀器
    #     # iou_aux_ratio=0.5,
    # )