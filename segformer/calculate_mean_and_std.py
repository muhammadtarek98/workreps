import os
from pathlib import Path
import cv2
import numpy as np

dirs=["/home/cplus/projects/m.tarek_master/gravel_2D/graval_detection_project/datasets/115351AA.mp4_",
      "/home/cplus/projects/m.tarek_master/gravel_2D/graval_detection_project/datasets/under_water_masks_dataset/train/images",
      "/home/cplus/projects/m.tarek_master/gravel_2D/graval_detection_project/datasets/under_water_masks_dataset/val/images",
      "/home/cplus/projects/m.tarek_master/gravel_2D/graval_detection_project/datasets/under_water_masks_dataset/test/images"]

mean = np.array([0., 0., 0.])
stdTemp = np.array([0., 0., 0.])
std = np.array([0., 0., 0.])
files=[]
for dir in dirs:
    for file in os.listdir(dir):
        if file.endswith(".jpg") or file.endswith(".png"):
            files.append(os.path.join(dir,file))
print(len(files))
for i in range(len(files)):
    print(files[i])
    im = cv2.imread(str(files[i]))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im.astype(float) / 255.

    for j in range(3):
        mean[j] += np.mean(im[:, :, j])

mean = (mean / len(files))

for i in range(len(files)):
    print(files[i])
    im = cv2.imread(str(files[i]))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im.astype(float) / 255.
    for j in range(3):
        stdTemp[j] += ((im[:, :, j] - mean[j]) ** 2).sum() / (im.shape[0] * im.shape[1])

std = np.sqrt(stdTemp / len(files))

print(mean)
print(std)