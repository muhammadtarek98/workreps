from torch.utils.data import Dataset, DataLoader
from PIL import Image
from matplotlib import pyplot as plt
from transformers import AdamW
import torch
from torch import nn
from sklearn.metrics import accuracy_score
from tqdm.notebook import tqdm
import torchvision
import os
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor, SegformerFeatureExtractor, \
    SegformerConfig
import pandas as pd
import cv2
import albumentations as aug
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchinfo import summary
import numpy as np
from datasets import load_metric
import evaluate
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import BasePredictionWriter


class ImageSegmentationDataset(Dataset):
    """Image segmentation dataset."""

    def __init__(self, root_dir, feature_extractor, id2label, transforms=None, train=True):
        super(ImageSegmentationDataset, self).__init__()
        self.root_dir = root_dir
        self.id2label = id2label
        self.feature_extractor = feature_extractor
        self.train = train
        self.transforms = transforms
        self.img_dir = os.path.join(self.root_dir, "images")
        self.ann_dir = os.path.join(self.root_dir, "pngmasks")
        image_file_names = []
        for root, dirs, files in os.walk(self.img_dir):
            image_file_names.extend(files)
        self.images = sorted(image_file_names)
        # read annotations
        annotation_file_names = []
        for root, dirs, files in os.walk(self.ann_dir):
            annotation_file_names.extend(files)
        self.annotations = sorted(annotation_file_names)
        assert len(self.images) == len(self.annotations) or len(
            self.images) == 0, "There must be as many images as there are segmentation maps"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = cv2.imread(os.path.join(self.img_dir, self.images[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        segmentation_map = cv2.imread(os.path.join(self.ann_dir, self.annotations[idx]))
        segmentation_map = cv2.cvtColor(segmentation_map, cv2.COLOR_BGR2GRAY)
        # randomly crop + pad both image and segmentation map to same size
        encoded_inputs = self.feature_extractor(image, segmentation_map, return_tensors="pt")
        for k, v in encoded_inputs.items():
            encoded_inputs[k].squeeze_()  # remove batch dimension
        return encoded_inputs

class ImageSegmentationDatasetInfernce(Dataset):
    """Image segmentation dataset."""
    def __init__(self, image_dir,feature_extractor):
        super(ImageSegmentationDatasetInfernce,self).__init__()
        self.img_dir = image_dir
        self.feature_extractor=feature_extractor
        image_file_names = []
        for root, dirs, files in os.walk(self.img_dir):
            image_file_names.extend(files)
        self.images = sorted(image_file_names)
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        image = cv2.imread(os.path.join(self.img_dir, self.images[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encoded_inputs = self.feature_extractor(image, return_tensors="pt")
        for k,v in encoded_inputs.items():
          encoded_inputs[k].squeeze_() # remove batch dimension
        return encoded_inputs