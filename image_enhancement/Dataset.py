from random import shuffle
from torch.utils.data import Dataset
import os
import cv2
import torch
import numpy as np
import torchvision
import albumentations
from UW_CycleGAN import Configs


def prepare_image(image_dir):
    image = cv2.imread(filename=image_dir)
    image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB)
    return image


class CustomDataset(Dataset):
    def __init__(self, images_dir: str,
                 device: torch.device = None,
                 transform=None):
        super().__init__()
        self.root_image_dir:str = images_dir

        self.lr_images:str = os.path.join(self.root_image_dir, "trainA")
        self.hr_images:str = os.path.join(self.root_image_dir, "trainB")
        self.external_dataset:str = "/home/cplus/projects/m.tarek_master/Image_enhancement/Enhancement_Dataset/"

        self.lr_images_list = []
        self.hr_images_list = []
        for file in os.listdir(self.lr_images):
            self.lr_images_list.append(
                os.path.join(self.root_image_dir, self.lr_images, file)
            )
        for file in os.listdir(self.external_dataset):
            if file.endswith(".png") or file.endswith(".jpg"):
                self.lr_images_list.append(
                    os.path.join(self.external_dataset,file))
        for file in os.listdir(self.hr_images):
            self.hr_images_list.append(
                os.path.join(self.root_image_dir, self.hr_images, file)
            )
        self.lr_images_list = sorted(self.lr_images_list)
        self.hr_images_list = sorted(self.hr_images_list)
        self.hr_length = len(self.hr_images_list)
        self.lr_length = len(self.lr_images_list)
        self.data_set_length = max(self.lr_length, self.hr_length)
        self.transform = transform
        self.device = device

    def __len__(self) -> int:
        return self.data_set_length

    def __getitem__(self, idx: int):
        lr_image_file = self.lr_images_list[idx % self.lr_length]
        hr_image_file = self.hr_images_list[idx % self.hr_length]
        lr_image = cv2.cvtColor(src=cv2.imread(lr_image_file), code=cv2.COLOR_BGR2RGB)
        hr_image = cv2.cvtColor(src=cv2.imread(hr_image_file), code=cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            aug = self.transform(image=lr_image, hr_image=hr_image)
            lr_image = aug["image"]
            hr_image = aug["hr_image"]
            #print(lr_image.shape)
            #print(hr_image.shape)
            #print(type(hr_image))
            #print(type(lr_image))
        return lr_image, hr_image


"""
root_dir = "/home/cplus/projects/m.tarek_master/Image_enhancement/dataset/underwater_imagenet"
transform = Configs.transform
data_set = CustomDataset(images_dir=root_dir, transform=transform)
print(data_set[0])
"""
