import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torchvision
out_feature: list = [64, 128, 256, 512]
input_shape: tuple = (3, 720, 720)
batch_size: int = 64
transform = A.Compose(is_check_shapes=False,transforms=[
    A.Resize(height=720, width=720),
    A.ToFloat(),
    #A.Normalize(mean=[0.24472233,0.50500972,0.4443582],std=[0.20768219,0.24286765,0.2468539 ],always_apply=True),
    A.pytorch.ToTensorV2()],
    additional_targets={'hr_image': 'image'},)
