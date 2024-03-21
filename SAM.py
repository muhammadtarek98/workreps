import segment_anything
import cv2
import torch
import torchvision
import sys
from matplotlib import pyplot as plt
import numpy as np
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)
    #return img
sys.path.append("..")
sam_checkpoint = "/home/cplus/projects/m.tarek_master/graval_detection_3D/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
image=cv2.imread("/home/cplus/projects/m.tarek_master/graval_detection_project/115351AA.mp4_/0_left.jpg")
image=cv2.cvtColor(src=image,code=cv2.COLOR_BGR2RGB)
#image=Image.open("/home/cplus/projects/m.tarek_master/graval_detection_project/115351AA.mp4_/11_left.jpg")
sam = segment_anything.sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
#cv2.imshow(mat=image,winname="test")
#cv2.waitKey(0)
mask_generator = segment_anything.SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(image)
print(len(masks))
plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.show() 