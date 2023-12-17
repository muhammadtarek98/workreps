from ultralytics import YOLO
import torch
import os
"""
def convert_ultralytics_to_pytorch(weights_path,task,savin_dir):

    model=YOLO(model=weights_path,task=task)
    pytorch_model_version=model.model
    #print(type(pytorch_model_version))
    if os.path.exists(savin_dir):
        os.remove(savin_dir)
    torch.save(pytorch_model_version.state_dict(), f=savin_dir)
    model=torch.load(savin_dir,map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model


#save="cashe.pth"
#convert_ultralytics_to_pytorch(weights_path,savin_dir=save,task="segment")"""
weights_path=r"D:\graval detection project\mareim runs\segment\train_UW_1080\weights\best.pt"
model=torch.load(f=weights_path,map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
model_architecture=model["model"]
training_args=model["train_args"]
print(training_args)