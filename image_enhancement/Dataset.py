from random import shuffle
from torch.utils.data import Dataset
import os
import cv2
import torch
import numpy as np
import torchvision


class CustomDataset(Dataset):
    def __init__(self, images_dir: str,
                 device: torch.device,
                 transform: torchvision.transforms = None):
        super().__init__()
        self.root_image_dir = images_dir

        self.lr_images = os.path.join(self.root_image_dir, "blurred_images")
        self.hr_images = os.path.join(self.root_image_dir, "sharped_images")
        self.lr_images_list = []
        self.hr_images_list = []
        for file in os.listdir(self.lr_images):
            self.lr_images_list.append(
                os.path.join(self.root_image_dir, self.lr_images, file)
            )
        for file in os.listdir(self.hr_images):
            self.hr_images_list.append(
                os.path.join(self.root_image_dir, self.hr_images, file)
            )
        self.hr_length = len(self.hr_images_list)
        self.lr_length = len(self.lr_images_list)
        self.data_set_length = max(self.lr_length, self.hr_length)
        self.transform = transform
        self.device = device
        shuffle(self.hr_images_list)
        shuffle(self.lr_images_list)

    def __len__(self) -> int:
        return self.data_set_length

    def __getitem__(self, idx:int):
        lr_image_file = self.lr_images_list[idx]
        hr_image_file = self.hr_images_list[idx]
        lr_image = cv2.cvtColor(src=cv2.imread(lr_image_file), code=cv2.COLOR_BGR2RGB)
        hr_image = cv2.cvtColor(src=cv2.imread(hr_image_file), code=cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            lr_image= self.transform(lr_image)
            hr_image = self.transform(hr_image)

        return [lr_image, hr_image]

"""
root_dir = "/home/cplus/projects/m.tarek_master/Image_enhancement/dataset/underwater_imagenet"
data_set = CustomDataset(images_dir=root_dir, device=device)
print(data_set.__len__())
"""