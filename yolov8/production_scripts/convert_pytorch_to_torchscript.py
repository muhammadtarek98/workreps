import torch
#import convert_ultralytics_to_pytorch
from ultralytics import YOLO

weights_path=r"D:\graval detection project\mareim runs\segment\train_UW_1080\weights\best.pt"

task="segment"
model=YOLO(weights_path,task=task)
pytorch_model_version = model.model


saved_dir=r"D:\graval detection project\mareim runs\segment\train_UW_1080\best_cpu.torchscript"
imgsz=1088
#pytorch_model_version.load_state_dict(torch.load(weights_path, map_location="cuda" if torch.cuda.is_available() else "cpu"))
#print(type(model))
#model=torch.jit.load(saved_dir,map_location="cpu")
#traced_script_module = torch.jit.trace(model,torch.randn((1,3,imgsz,imgsz)))
