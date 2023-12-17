import torch
import cv2
imgsz=1088
weights_path=r"D:\graval detection project\mareim runs\segment\train_UW_1080\weights\best.torchscript"
device="cuda" if torch.cuda.is_available() else "cpu"
model=torch.jit.load(f=weights_path,map_location=device)
img=cv2.imread(r"D:\graval detection project\datasets\under water dataset\images\test\35_left_2023_10_22_10_44_35.jpg")
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img=cv2.resize(src=img,dsize=(1088,1088))
prediction=model(img)
