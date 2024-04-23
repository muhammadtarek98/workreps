import evaluate
import pytorch_lightning as pl
import torch
import torchvision
from torch import nn
from transformers import AdamW
from transformers import SegformerForSemanticSegmentation
import Configs
from Model_Pytorch import Model
import torchinfo


class SegformerFinetuner(pl.LightningModule):
    def __init__(self):
        super(SegformerFinetuner, self).__init__()
        self.save_hyperparameters()
        self.id2label = Configs.id2label
        self.metrics_interval = Configs.metrics_interval
        self.train_dl = Configs.train_dataloader
        self.val_dl = Configs.valid_dataloader
        self.test_dl = Configs.test_dataloader
        self.num_classes = len(self.id2label.keys())
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.model_name = "nvidia/mit-b1"

        self.model = SegformerForSemanticSegmentation.from_pretrained(pretrained_model_name_or_path=self.model_name,
                                                                      ignore_mismatched_sizes=True,
                                                                      reshape_last_stage=True, )
        self.model.config.num_labels = self.num_classes
        self.model.config.id2label = self.id2label
        self.model.config.label2id = self.label2id
        #print(self.model.config.num_labels)
        #self.model.
        #self.model.config.
        #self.model_torch_class = Model()
        #self.model_torch_class.model.config.num_labels = self.num_classes
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
        upsampled_logits = nn.functional.interpolate(logits,
                                                     size=(Configs.h, Configs.w),
                                                     mode=Configs.up_sampling,
                                                     align_corners=True)
        predicted = upsampled_logits.argmax(dim=1)
        self.train_mean_iou.add_batch(predictions=predicted.detach().cpu().numpy(),
                                      references=masks.detach().cpu().numpy())
        if batch_nb % self.metrics_interval == 0:
            metrics = self.train_mean_iou.compute(num_labels=self.num_classes,
                                                  ignore_index=255,
                                                  reduce_labels=Configs.reduce_labels)
            metrics = {'loss': loss,
                       "mean_iou": metrics["mean_iou"],
                       "mean_accuracy": metrics["mean_accuracy"]}
            for k, v in metrics.items():
                self.log(k, v,
                         enable_graph=True,
                         prog_bar=True)
            self.log_predictions_to_tensorboard(images, masks, predicted, mode='train')
            return metrics
        else:
            return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        images, masks = batch['pixel_values'], batch['labels']
        outputs = self(images, masks)
        loss, logits = outputs[0], outputs[1]
        upsampled_logits = nn.functional.interpolate(logits,
                                                     size=(Configs.h, Configs.w),
                                                     mode=Configs.up_sampling)
        predicted = upsampled_logits.argmax(dim=1)
        self.val_mean_iou.add_batch(predictions=predicted.detach().cpu().numpy(),
                                    references=masks.detach().cpu().numpy())
        val_metrics = self.val_mean_iou.compute(num_labels=self.num_classes,
                                                ignore_index=255,
                                                reduce_labels=Configs.reduce_labels)
        val_metrics = {'val_loss': loss, "val_mean_iou": val_metrics["mean_iou"],
                       "val_mean_accuracy": val_metrics["mean_accuracy"]}

        for k, v in val_metrics.items():
            self.log(k, v, enable_graph=True, prog_bar=True)
        self.log_predictions_to_tensorboard(images, masks, predicted, mode='val')
        return val_metrics

    def test_step(self, batch, batch_nb):
        images, masks = batch['pixel_values'], batch['labels']
        outputs = self(images, masks)
        loss, logits = outputs[0], outputs[1]
        upsampled_logits = nn.functional.interpolate(logits,
                                                     size=(Configs.h, Configs.w),
                                                     mode=Configs.up_sampling)
        predicted = upsampled_logits.argmax(dim=1)
        self.test_mean_iou.add_batch(predictions=predicted.detach().cpu().numpy(),
                                     references=masks.detach().cpu().numpy())
        test_metrics = self.test_mean_iou.compute(num_labels=self.num_classes,
                                                  ignore_index=255,
                                                  reduce_labels=True)
        test_metrics = {'test_loss': loss,
                        "test_mean_iou": test_metrics["mean_iou"],
                        "test_mean_accuracy": test_metrics["mean_accuracy"]}
        for k, v in test_metrics.items():
            self.log(k, v,
                     enable_graph=True,
                     prog_bar=True)
        self.log_predictions_to_tensorboard(images,
                                            masks,
                                            predicted,
                                            mode='test')
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


test = SegformerFinetuner()
image, labels = None, None
for batch in Configs.train_dataloader:
    image, mask = batch["pixel_values"], batch["labels"]
    output = test(image)
    print(output)
    torchinfo.summary(model=test.model, input_data=image)
    break
