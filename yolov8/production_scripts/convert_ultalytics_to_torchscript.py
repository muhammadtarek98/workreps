from ultralytics import YOLO
weight_path=r"D:\graval detection project\mareim runs\segment\train_UW_1080\weights\best.pt"
task="segment"
model=YOLO(weight_path,task=task)
model.export(format="onnx",optimize=False,nms=True)