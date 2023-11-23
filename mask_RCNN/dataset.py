import json
import os
import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from shapely.geometry import Polygon
from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F


class CustomDataSet(Dataset):
    def __init__(self, csv_file_dir: str, data_dir: str, images_dir: str, masks_dir, json_dirs: str,
                 transforms: torchvision.transforms.Compose = None):
        super(CustomDataSet, self).__init__()
        self.df = pd.read_csv(csv_file_dir)
        self.data_dir = data_dir
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transforms = transforms
        self.json_dirs = json_dirs

    def __len__(self) -> int:
        return len(self.df)

    @staticmethod
    def open_json_file(file_dir: str):
        with open(file_dir) as f:
            file = json.load(f)
        return file

    @staticmethod
    def mask_to_box(polygon):
        mask_polygon = Polygon(shell=polygon)
        min_x, min_y, max_x, max_y = mask_polygon.bounds
        area = mask_polygon.area
        return float(min_x), float(min_y), float(max_x), float(max_y), area

    def __getitem__(self, index):
        # load images and masks
        img_path = os.path.join(self.images_dir, self.df["images_dir"].iloc[index])
        mask_dir = os.path.join(self.masks_dir, self.df["masks_dir"].iloc[index])
        img = read_image(img_path)
        mask = read_image(mask_dir)
        # instances are encoded as different colors
        obj_ids = torch.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        num_objs = len(obj_ids)
        masks = (mask == obj_ids[:, None, None]).to(dtype=torch.uint8)

        # get bounding box coordinates for each mask
        boxes = masks_to_boxes(masks)

        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)

        image_id = index
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Wrap sample and targets into torchvision tv_tensors:
        img = tv_tensors.Image(img)

        target = dict()
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img)).squeeze(1)
        target["masks"] = tv_tensors.Mask(masks)
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)

        return img, [target]


"""
    def __getitem__(self, index):
        # load images and masks
        img_path = os.path.join(self.images_dir, self.df["images_dir"].iloc[index])
        mask_dir = os.path.join(self.masks_dir, self.df["masks_dir"].iloc[index])
        img = read_image(img_path)
        mask = read_image(mask_dir)
        # instances are encoded as different colors
        obj_ids = torch.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        num_objs = len(obj_ids)
        masks = (mask == obj_ids[:, None, None]).to(dtype=torch.uint8)

        # get bounding box coordinates for each mask
        boxes = masks_to_boxes(masks)

        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)

        image_id = index
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Wrap sample and targets into torchvision tv_tensors:
        img = tv_tensors.Image(img)

        target = dict()
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img))
        target["masks"] = tv_tensors.Mask(masks)
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)

        return img, [target]"""


def get_transform(train):
    transforms = []
    if train:
        transforms.append(torchvision.transforms.RandomHorizontalFlip(0.5))
    transforms.append(torchvision.transforms.v2.ToDtype(torch.float, scale=True))
    transforms.append(torchvision.transforms.v2.ToPureTensor())
    return torchvision.transforms.Compose(transforms)
