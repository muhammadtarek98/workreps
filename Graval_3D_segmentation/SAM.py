import segment_anything
import cv2
import torch
import torchvision
from matplotlib import pyplot as plt
import numpy as np
import os, sys
ci_build_and_not_headless = False
try:
    from cv2.version import ci_build, headless
    ci_and_not_headless = ci_build and not headless
except:
    pass
if sys.platform.startswith("linux") and ci_and_not_headless:
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
if sys.platform.startswith("linux") and ci_and_not_headless:
    os.environ.pop("QT_QPA_FONTDIR")
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
#image=cv2.cvtColor(src=image,code=cv2.COLOR_BGR2RGB)
#image=Image.open("/home/cplus/projects/m.tarek_master/graval_detection_project/115351AA.mp4_/11_left.jpg")
sam = segment_anything.sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = segment_anything.SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(image)
mask_generator_2 = segment_anything.SamAutomaticMaskGenerator(
    model=sam,points_per_side=32,pred_iou_thresh=0.6,stability_score_thresh=0.92,
    crop_n_layers=1,crop_n_points_downscale_factor=2,min_mask_region_area=100)
masks_2=mask_generator_2.generate(image=image)

print(len(masks))
print(len(masks_2))

plt.figure(figsize=(20,20))
plt.imshow(image)
#show_anns(masks)
show_anns(masks_2)
plt.axis('off')
plt.show() 