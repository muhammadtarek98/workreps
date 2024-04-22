from transformers import SegformerImageProcessor
from DataSet import ImageSegmentationDatasetInfernce, ImageSegmentationDataset
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import BasePredictionWriter
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import torch

classes = ["background", "stone"]
num_devices = 1
num_epochs=100
device_type = 'gpu'
id2label = {0: classes[0], 1: classes[1]}
label2id = {v: k for k, v in id2label.items()}
train_dir = "/home/cplus/projects/m.tarek_master/gravel_2D/graval_detection_project/datasets/under_water_masks_dataset/train"
valid_dir = "/home/cplus/projects/m.tarek_master/gravel_2D/graval_detection_project/datasets/under_water_masks_dataset/val"
test_dir = "/home/cplus/projects/m.tarek_master/gravel_2D/graval_detection_project/datasets/under_water_masks_dataset/test"
inference_dir = "/home/cplus/projects/m.tarek_master/gravel_2D/graval_detection_project/115351AA.mp4_"
model_name = "nvidia/mit-b1"
feature_extractor = SegformerImageProcessor.from_pretrained(model_name)
feature_extractor.do_reduce_labels = False
feature_extractor.size = 1080
batch_size = 1
train_dataset = ImageSegmentationDataset(root_dir=train_dir, feature_extractor=feature_extractor, id2label=id2label,
                                         transforms=None, train=True)
valid_dataset = ImageSegmentationDataset(root_dir=valid_dir, feature_extractor=feature_extractor, id2label=id2label)
test_dataset = ImageSegmentationDataset(root_dir=test_dir, feature_extractor=feature_extractor, id2label=id2label)
inference_dataset = ImageSegmentationDatasetInfernce(image_dir=inference_dir, feature_extractor=feature_extractor)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
inference_dataloader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
logger = TensorBoardLogger(save_dir="logs", name="segformer_logs_b1")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=100, verbose=False, mode="min")
checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_loss")
