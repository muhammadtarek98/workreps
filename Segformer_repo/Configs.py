import cv2
import numpy as np
import torchvision.transforms
from transformers import SegformerImageProcessor
from DataSet import ImageSegmentationDatasetInference, ImageSegmentationDataset
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import torch
from matplotlib import pyplot as plt
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor

up_sampling = "bilinear"
num_devices = 1
num_epochs = 200
batch_size = 1
learning_rate = 2e-07
metrics_interval = 1
device_type = 'gpu'
use_pytorch_imp = False
h = 1080
w = 1080
reduce_labels = False
# dataset
mean_1 = [0.0362148, 0.05997756, 0.03632534]
std_1 = [0.18304984, 0.23559486, 0.18306885]
#combination
mean_2 = [0.27962195, 0.32185032, 0.33909433]
std_2 = [0.16759151, 0.16447153, 0.15949133]
#real data
mean_3 = [0.40227084, 0.43002412, 0.4154681]
std_3 = [0.12789169, 0.11065691, 0.10975399]
mean_dataset_v2 = [0.42912965, 0.44123175, 0.43052802]
std_dataset_v2 = [0.12035021, 0.11250368, 0.11089015]
patience = 20
transform = torchvision.transforms.Compose([
    torchvision.transforms.Normalize(mean=mean_1, std=std_1)
])
classes = ["background", "stone"]
id2label = {0: classes[0], 1: classes[1]}
label2id = {v: k for k, v in id2label.items()}

id2color = {
    0: (0, 0, 0),
    1: (255, 0, 0)}
train_dir = "/home/cplus/projects/m.tarek_master/gravel_2D/graval_detection_project/datasets/under_water_masks_dataset_v3/training"
valid_dir = "/home/cplus/projects/m.tarek_master/gravel_2D/graval_detection_project/datasets/under_water_masks_dataset_v3/validation"
test_dir = "/home/cplus/projects/m.tarek_master/gravel_2D/graval_detection_project/datasets/under_water_masks_dataset_v3/testing"
inference_dir = "/home/cplus/projects/m.tarek_master/gravel_2D/graval_detection_project/transformers_for_CV/Segformer_repo/new_version"
model_name = "nvidia/mit-b1"


def create_feature_extractor(mean=None, std=None, model_name: str = "nvidia/mit-b1"):
    feature_extractor = SegformerImageProcessor.from_pretrained(model_name)
    feature_extractor.size["height"] = h
    feature_extractor.size["width"] = w
    feature_extractor.do_normalize = False
    feature_extractor.do_reduce_labels = reduce_labels
    #feature_extractor.image_std = std
    #feature_extractor.image_mean = mean
    return feature_extractor


#print(feature_extractor)
train_dataset = ImageSegmentationDataset(root_dir=train_dir,
                                         feature_extractor=create_feature_extractor(mean=mean_dataset_v2,
                                                                                    std=std_dataset_v2),
                                         id2label=id2label,
                                         transforms=None, train=True)
valid_dataset = ImageSegmentationDataset(root_dir=valid_dir,
                                         feature_extractor=create_feature_extractor(mean=mean_dataset_v2,
                                                                                    std=std_dataset_v2),
                                         id2label=id2label)
test_dataset = ImageSegmentationDataset(root_dir=test_dir,
                                        feature_extractor=create_feature_extractor(mean=mean_dataset_v2,
                                                                                   std=std_dataset_v2),
                                        id2label=id2label)
inference_dataset = ImageSegmentationDataset(
    root_dir="/home/cplus/projects/m.tarek_master/gravel_2D/graval_detection_project/transformers_for_CV/Segformer_repo/new_version",
    transforms=None,
    feature_extractor=create_feature_extractor(mean=mean_dataset_v2,
                                               std=std_dataset_v2),
    id2label=id2label)
train_dataloader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0)
valid_dataloader = DataLoader(valid_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=0)
test_dataloader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=0)
inference_dataloader = DataLoader(inference_dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=0)
lr_monitor = LearningRateMonitor(logging_interval='step')
logger = TensorBoardLogger(save_dir="dataset_v3_logs",
                           name="segformer_logs_b1_without_normalization",
                           log_graph=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00,
                                    patience=patience, verbose=False, mode="min")
checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_loss")

"""
for batch in train_dataloader:
    image, labels = batch["pixel_values"], batch["labels"]
    print(image.shape)
    print(labels.shape)
    #image,labels=feature_extractor(image,labels, return_tensors="pt")
    #print(labels)
    #print(labels)
    print(torch.unique(labels))
    plt.imshow(labels.squeeze_(), cmap="binary")
    plt.show()
    break
"""
