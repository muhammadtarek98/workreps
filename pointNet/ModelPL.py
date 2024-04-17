from Configs import *
import torch
import pytorch_lightning as pl
import torchinfo
from Model import PointNetSegmentation
import torchmetrics


class ModelTuner(pl.LightningModule):
    def __init__(self, num_points: int, dim: int, labels: int,
                 num_global_feature, metrics_interval: int = 10, train_data=None, validation_data=None, test_data=None,
                 lr: float = 1e-3):
        super(ModelTuner, self).__init__()
        self.save_hyperparameters(logger=True)

        self.num_points = num_points
        self.dim = dim
        self.labels = labels
        self.lr = lr
        self.num_global_feature = num_global_feature
        self.metric_intervals = metrics_interval
        self.train_dl = train_data
        self.validation_dl = validation_data
        self.test_dl = test_data
        self.model = PointNetSegmentation(num_points=self.num_points,
                                          dim=self.dim,
                                          labels=self.labels,
                                          num_global_feature=self.num_global_feature)

    def step(self, batch, reduction="mean"):
        points, target, _, _, _, _ = batch
        logits,_,_ = self.model(points)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(input=logits, target=target, reduction=reduction)
        pred = torch.argmax(logits, dim=1)
        return loss, pred, target

    def forward(self, x):
        x = self.model(x)
        return x

    def compute_iou(self, targets, predictions):
        targets = targets.reshape(-1)
        predictions = predictions.reshape(-1)

        intersection = torch.sum(predictions == targets)  # true positives
        union = len(predictions) + len(targets) - intersection

        return intersection / union

    def metrics(self, pred, target):
        acc = torchmetrics.functional.accuracy(preds=pred, target=target, num_labels=self.labels)
        precision = torchmetrics.functional.precision(preds=pred, target=target, num_labels=self.labels)
        recall = torchmetrics.functional.recall(preds=pred, target=target, num_labels=self.labels)
        f1_score = torchmetrics.functional.f1_score(preds=pred, target=target, num_labels=self.labels)
        mean_iou = self.compute_iou(targets=target, predictions=pred)
        return acc, precision, recall, f1_score, mean_iou

    def training_step(self, batch, batch_dx):

        loss, pred, target = self.step(batch)
        acc, precision, recall, f1_score, iou = self.metrics(pred, target)
        training_metrics = {"loss": loss, "acc": acc, "precision": precision, "recall": recall, "f1_score": f1_score}
        self.log(name="training_precision", value=precision, on_step=True, on_epoch=True, prog_bar=True)
        self.log(name="training_accuracy", value=acc, on_epoch=True, on_step=True, prog_bar=True)
        self.log(name="training_loss", value=loss, on_epoch=True, on_step=True, prog_bar=True)
        self.log(name="training_recall", value=recall, on_step=True, on_epoch=True, prog_bar=True)
        self.log(name="training_f1_score", value=f1_score, on_step=True, on_epoch=True, prog_bar=True)
        self.log(name="training_iou", value=iou, on_step=True, on_epoch=True, prog_bar=True)
        return training_metrics

    def validation_step(self, batch, batch_idx):
        loss, pred, target = self.step(batch)
        acc, precision, recall, f1_score, iou = self.metrics(pred, target)
        validation_metrics = {"loss": loss, "acc": acc, "precision": precision, "recall": recall, "f1_score": f1_score}
        self.log(name="validation_iou", value=iou, on_step=True, on_epoch=True, prog_bar=True)
        self.log(name="validation_precision", value=precision, on_step=True, on_epoch=True, prog_bar=True)
        self.log(name="validation_accuracy", value=acc, on_epoch=True, on_step=True, prog_bar=True)
        self.log(name="validation_loss", value=loss, on_epoch=True, on_step=True, prog_bar=True)
        self.log(name="validation_recall", value=recall, on_step=True, on_epoch=True, prog_bar=True)
        self.log(name="validation_f1_score", value=f1_score, on_step=True, on_epoch=True, prog_bar=True)
        return validation_metrics

    def test_step(self, batch, batch_idx):
        loss, pred, target = self.step(batch)
        acc, precision, recall, f1_score, iou = self.metrics(pred, target)
        test_metrics = {"loss": loss, "acc": acc, "precision": precision, "recall": recall, "f1_score": f1_score}
        self.log(name="test_iou", value=iou, on_step=True, on_epoch=True, prog_bar=True)
        self.log(name="test_precision", value=precision, on_step=True, on_epoch=True, prog_bar=True)
        self.log(name="test_accuracy", value=acc, on_epoch=True, on_step=True, prog_bar=True)
        self.log(name="test_loss", value=loss, on_epoch=True, on_step=True, prog_bar=True)
        self.log(name="test_recall", value=recall, on_step=True, on_epoch=True, prog_bar=True)
        self.log(name="test_f1_score", value=f1_score, on_step=True, on_epoch=True, prog_bar=True)
        return test_metrics

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.parameters(), lr=self.lr
        )

    def train_dataloader(self):
        return self.train_dl

    def val_dataloader(self):
        return self.validation_dl

    def test_dataloader(self):
        return self.test_dl


#model = ModelTuner(lr=lr, dim=dim, labels=labels, num_points=num_points, num_global_feature=num_global_feature)
#torchinfo.summary(model=model)
