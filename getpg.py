from ultralytics import YOLO

# 加载你刚刚训练好的最优权重
model = YOLO("Baseline_Results/RT_DETR_l_Baseline/weights/best.pt")
# 打印模型信息
model.info()
