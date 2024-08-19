import pytorch_lightning as pl
import torchmetrics.image
import torch
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS, OptimizerLRScheduler

from UW_CycleGAN.Discriminator import Discriminator
from UW_CycleGAN.Generator import Generator, model


class GAN(pl.LightningModule):
    def __init__(self, device:torch.device,
                 train_loader=None,
                 val_loader=None,
                 lr: float = 2e-3,
                 input_shape=None, ):
        super().__init__()
        self.save_hyperparameters()
        self.generator_lr = Generator()
        self.generator_hr = Generator()
        self.discriminator_lr = Discriminator()
        self.discriminator_hr = Discriminator()
        self.learning_rate = lr
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.generator_optimizer = None
        self.discriminator_optimizer = None
        self.l1_loss=torch.nn.L1Loss()
        self.mse_loss=torch.nn.MSELoss()

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self.val_loader

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self.train_loader

    def configure_optimizers(self) -> OptimizerLRScheduler:
        self.discriminator_optimizer=torch.optim.Adam(lr=self.learning_rate,
                                                      params=list(self.discriminator_lr.parameters())+list(self.discriminator_hr.parameters()))
        self.generator_optimizer=torch.optim.Adam(lr=self.learning_rate,
                                                  params=list(self.generator_lr.parameters()+list(self.generator_lr.parameters())))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model=GAN(device=device)
