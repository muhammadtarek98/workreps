from torch.utils.data import Dataset, DataLoader
from transformers import AdamW
import torch
from torch import nn
from sklearn.metrics import accuracy_score
from tqdm.notebook import tqdm
import os
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor ,SegformerFeatureExtractor
import pandas as pd
import cv2
import numpy as np
import albumentations as aug
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchinfo import summary
import numpy as np
from datasets import load_metric
import evaluate
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

torch.set_float32_matmul_precision('high')

class ImageSegmentationDataset(Dataset):
    """Image segmentation dataset."""
    def __init__(self, root_dir, feature_extractor, transforms=None, train=True):
        super(ImageSegmentationDataset,self).__init__()
        self.root_dir = root_dir
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
        assert len(self.images) == len(self.annotations) or len(self.images)==0 ,  "There must be as many images as there are segmentation maps"
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        image = cv2.imread(os.path.join(self.img_dir, self.images[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        segmentation_map = cv2.imread(os.path.join(self.ann_dir, self.annotations[idx]))
        segmentation_map = cv2.cvtColor(segmentation_map, cv2.COLOR_BGR2GRAY)
        if self.transforms is not None:
            augmented = self.transforms(image=image, mask=segmentation_map)
            encoded_inputs = self.feature_extractor(augmented['image'], augmented['mask'], return_tensors="pt")
        else:
            encoded_inputs = self.feature_extractor(image, segmentation_map, return_tensors="pt")
        for k,v in encoded_inputs.items():
            encoded_inputs[k].squeeze_()

        return encoded_inputs

class SegformerFinetuner(pl.LightningModule):
    
    def __init__(self, id2label, train_dataloader=None, val_dataloader=None, test_dataloader=None, metrics_interval=100):
        super(SegformerFinetuner, self).__init__()
        self.id2label = id2label
        self.metrics_interval = metrics_interval
        self.train_dl = train_dataloader
        self.val_dl = val_dataloader
        self.test_dl = test_dataloader
        self.num_classes = len(id2label.keys())
        self.label2id = {v:k for k,v in self.id2label.items()}
        self.model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b5", ignore_mismatched_sizes=True,
                                                         reshape_last_stage=True)
        self.train_mean_iou = evaluate.load("mean_iou")
        self.val_mean_iou = evaluate.load("mean_iou")
        self.test_mean_iou = evaluate.load("mean_iou")
        
    def forward(self, images, masks):
        outputs = self.model(pixel_values=images, labels=masks)
        return(outputs)
    
    def training_step(self, batch, batch_nb):
        images, masks = batch['pixel_values'], batch['labels']
        outputs = self(images, masks)
        loss, logits = outputs[0], outputs[1]
        upsampled_logits = nn.functional.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        predicted = upsampled_logits.argmax(dim=1)
        self.train_mean_iou.add_batch(predictions=predicted.detach().cpu().numpy(), references=masks.detach().cpu().numpy())
        if batch_nb % self.metrics_interval == 0:
            metrics = self.train_mean_iou.compute(num_labels=self.num_classes, ignore_index=255, reduce_labels=False,)
            metrics = {'loss': loss, "mean_iou": metrics["mean_iou"], "mean_accuracy": metrics["mean_accuracy"]}
            for k,v in metrics.items():
                self.log(k,v)
            return(metrics)
        else:
            return({'loss': loss})
    def validation_step(self, batch, batch_nb):
        images, masks = batch['pixel_values'], batch['labels']
        outputs = self(images, masks)
        loss, logits = outputs[0], outputs[1]
        upsampled_logits = nn.functional.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        predicted = upsampled_logits.argmax(dim=1)
        self.val_mean_iou.add_batch(predictions=predicted.detach().cpu().numpy(), references=masks.detach().cpu().numpy())
        val_metrics = self.val_mean_iou.compute(num_labels=self.num_classes, ignore_index=255, reduce_labels=False)
        val_metrics = {'val_loss': loss, "val_mean_iou": val_metrics["mean_iou"], "val_mean_accuracy": val_metrics["mean_accuracy"]}
        
        for k,v in val_metrics.items():
                self.log(k,v)
        return(val_metrics)
    
    def test_step(self, batch, batch_nb):
        images, masks = batch['pixel_values'], batch['labels']
        outputs = self(images, masks)
        loss, logits = outputs[0], outputs[1]
        upsampled_logits = nn.functional.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        predicted = upsampled_logits.argmax(dim=1)
        test_metircs=self.test_mean_iou.add_batch(predictions=predicted.detach().cpu().numpy(), references=masks.detach().cpu().numpy())
        test_metircs = {'test_loss': loss, "test_mean_iou": test_metircs["mean_iou"], "test_mean_accuracy":test_metircs["mean_accuracy"]}
        for k,v in test_metircs.items():
                self.log(k,v)
        return(test_metircs)
    
    
    def configure_optimizers(self):
        return torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=2e-05, eps=1e-08)
    
    def train_dataloader(self):
        return self.train_dl
    
    def val_dataloader(self):
        return self.val_dl
    
    def test_dataloader(self):
        return self.test_dl

transform = aug.Compose([aug.Flip(p=0.5),],is_check_shapes=False)

train_dir ="/home/cplus/projects/m.tarek_master/graval_detection_project/datasets/under_water_masks_dataset/train"
valid_dir="/home/cplus/projects/m.tarek_master/graval_detection_project/datasets/under_water_masks_dataset/val"
test_dir="/home/cplus/projects/m.tarek_master/graval_detection_project/datasets/under_water_masks_dataset/test"
feature_extractor = SegformerImageProcessor.from_pretrained ("nvidia/mit-b0")
train_dataset = ImageSegmentationDataset(root_dir=train_dir, feature_extractor=feature_extractor, transforms=transform)
valid_dataset = ImageSegmentationDataset(root_dir=valid_dir, feature_extractor=feature_extractor, transforms=None, train=False)
test_dataset = ImageSegmentationDataset(root_dir=test_dir, feature_extractor=feature_extractor, transforms=None, train=False)

classes = ["stone"]
print(classes)
id2label = {0:classes[0]}
print(id2label)
label2id = {v: k for k, v in id2label.items()}
print(label2id)

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=31)
valid_dataloader = DataLoader(valid_dataset, batch_size=4,shuffle=False,num_workers=4)
test_dataloader  = DataLoader(test_dataset,batch_size=1,shuffle=False,num_workers=0)

dataiter = iter(train_dataloader)
SegformerFineTuner=SegformerFinetuner(id2label=id2label,train_dataloader=train_dataloader,val_dataloader=valid_dataloader,test_dataloader=test_dataloader,metrics_interval=10)
SegformerFineTuner.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=3, verbose=False, mode="min")
checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_loss")
trainer=pl.Trainer(max_epochs=10,val_check_interval=len(valid_dataloader),accelerator="gpu",devices=1,callbacks=[early_stop_callback, checkpoint_callback])

trainer.fit(model=SegformerFineTuner,val_dataloaders=valid_dataloader,train_dataloaders=train_dataloader)