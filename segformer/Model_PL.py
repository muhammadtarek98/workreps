import evaluate
import pytorch_lightning as pl
import torch
import torchvision
from torch import nn
from transformers import AdamW
from transformers import SegformerForSemanticSegmentation
from Model_Pytorch import Model
from Configs import *


class SegformerFinetuner(pl.LightningModule):
    def __init__(self, ):
        super(SegformerFinetuner, self).__init__()
        self.id2label = id2label
        self.metrics_interval = metrics_interval
        self.train_dl = train_dataloader
        self.val_dl = valid_dataloader
        self.test_dl = test_dataloader
        self.num_classes = len(id2label.keys())
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.model_name=model_name

        self.model = SegformerForSemanticSegmentation.from_pretrained(self.model_name,
                                                                      ignore_mismatched_sizes=True,
                                                                      reshape_last_stage=True)

        self.model_torch_class = Model()
        self.train_mean_iou = evaluate.load("mean_iou")
        self.val_mean_iou = evaluate.load("mean_iou")
        self.test_mean_iou = evaluate.load("mean_iou")

    def forward(self, images, masks=None):
        outputs = self.model(images, masks)
        return outputs

    def training_step(self, batch, batch_nb):
        images, masks = batch['pixel_values'], batch['labels']
        outputs = self(images, masks)
        loss, logits = outputs[0], outputs[1]
        upsampled_logits = nn.functional.interpolate(logits, size=masks.shape[-2:], mode="nearest-exact")
        predicted = upsampled_logits.argmax(dim=1)
        self.train_mean_iou.add_batch(predictions=predicted.detach().cpu().numpy(),
                                      references=masks.detach().cpu().numpy())
        if batch_nb % self.metrics_interval == 0:
            metrics = self.train_mean_iou.compute(num_labels=self.num_classes, ignore_index=255, reduce_labels=False, )
            metrics = {'loss': loss, "mean_iou": metrics["mean_iou"], "mean_accuracy": metrics["mean_accuracy"]}
            for k, v in metrics.items():
                self.log(k, v, enable_graph=True, prog_bar=True)
            self.log_predictions_to_tensorboard(images, masks, predicted, 'train')
            return metrics
        else:
            return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        images, masks = batch['pixel_values'], batch['labels']
        outputs = self(images, masks)
        loss, logits = outputs[0], outputs[1]
        upsampled_logits = nn.functional.interpolate(logits, size=masks.shape[-2:], mode="nearest-exact")
        predicted = upsampled_logits.argmax(dim=1)
        self.val_mean_iou.add_batch(predictions=predicted.detach().cpu().numpy(),
                                    references=masks.detach().cpu().numpy())
        val_metrics = self.val_mean_iou.compute(num_labels=self.num_classes, ignore_index=255, reduce_labels=False)
        val_metrics = {'val_loss': loss, "val_mean_iou": val_metrics["mean_iou"],
                       "val_mean_accuracy": val_metrics["mean_accuracy"]}

        for k, v in val_metrics.items():
            self.log(k, v, enable_graph=True, prog_bar=True)
        self.log_predictions_to_tensorboard(images, masks, predicted, 'val')
        return val_metrics

    def test_step(self, batch, batch_nb):
        images, masks = batch['pixel_values'], batch['labels']
        outputs = self(images, masks)
        loss, logits = outputs[0], outputs[1]
        upsampled_logits = nn.functional.interpolate(logits, size=masks.shape[-2:], mode="nearest-exact")
        predicted = upsampled_logits.argmax(dim=1)
        self.test_mean_iou.add_batch(predictions=predicted.detach().cpu().numpy(),
                                     references=masks.detach().cpu().numpy())
        test_metrics = self.test_mean_iou.compute(num_labels=self.num_classes, ignore_index=255, reduce_labels=False)
        test_metrics = {'test_loss': loss, "test_mean_iou": test_metrics["mean_iou"],
                        "test_mean_accuracy": test_metrics["mean_accuracy"]}
        for k, v in test_metrics.items():
            self.log(k, v, enable_graph=True, prog_bar=True)
        self.log_predictions_to_tensorboard(images, masks, predicted, 'test')
        return test_metrics

    def configure_optimizers(self):
        return AdamW([p for p in self.parameters() if p.requires_grad], lr=2e-06, eps=1e-08)

    def train_dataloader(self):
        return self.train_dl

    def val_dataloader(self):
        return self.val_dl

    def test_dataloader(self):
        return self.test_dl

    def log_predictions_to_tensorboard(self, images, masks, predictions, mode='train'):
        img_grid = torchvision.utils.make_grid(images)
        mask_grid = torchvision.utils.make_grid(masks.unsqueeze(1))  # Assuming masks are single-channel
        pred_grid = torchvision.utils.make_grid(predictions.unsqueeze(1))  # Assuming predictions are single-channel
        self.logger.experiment.add_image(f'{mode}_images', img_grid, self.current_epoch)
        self.logger.experiment.add_image(f'{mode}_masks', mask_grid, self.current_epoch)
        self.logger.experiment.add_image(f'{mode}_predictions', pred_grid, self.current_epoch)
"""
test = SegformerFinetuner(use_pytorch_implementation=use_pytorch_imp)
x = torch.rand((1, 3, 1080, 1080))
output = test(x)
print(output)
"""
