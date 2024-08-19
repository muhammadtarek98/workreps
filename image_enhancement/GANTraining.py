from typing import Any

import pytorch_lightning as pl
import torchmetrics.image
import torch
from dask.array import optimize
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS, OptimizerLRScheduler, STEP_OUTPUT

from Discriminator import Discriminator
from Generator import Generator


class GAN(pl.LightningModule):
    def __init__(self,
                 train_loader=None,
                 val_loader=None,
                 lr: float = 2e-3,
                 input_shape=None,
                 cycle_lambda=10,
                 identity_lambda=0.5):
        super().__init__()
        self.input_shape=input_shape
        self.cycle_lambda=cycle_lambda
        self.identity_lambda=identity_lambda
        self.save_hyperparameters()
        self.generator_lr = Generator()
        self.generator_hr = Generator()
        self.discriminator_lr = Discriminator()
        self.discriminator_hr = Discriminator()
        self.learning_rate = lr
        self.train_loader = train_loader
        self.val_loader = val_loader
        #self.generator_optimizer = None
        #self.discriminator_optimizer = None
        self.l1_loss=torch.nn.L1Loss()
        self.mse_loss=torch.nn.MSELoss()
        self.automatic_optimization = False
    def forward(self, lr,hr):
        fake_lr=self.generator_lr(hr)
        fake_hr=self.generator_hr(lr)
        return fake_hr,fake_lr

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self.val_loader
    def discriminator_loss(self,discriminator,generator,fake,real):
        gen=generator(fake)
        real=discriminator(real)
        fake=discriminator(gen.detach())
        real_loss=self.mse_loss(real,torch.ones_like(real))
        fake_loss=self.mse_loss(fake,torch.zeros_like(fake))
        total_loss=real_loss+fake_loss
        return total_loss
    def generator_loss(self,discriminator,fake):
        disc_fake=discriminator(fake)
        loss=self.mse_loss(disc_fake,torch.ones_like(disc_fake))
        return loss
    def cycle_loss(self,real,cycled):
        loss=self.l1_loss(real,cycled)
        return loss
    def identity_loss(self,real,identity):
        loss=self.l1_loss(real,identity)
        return loss

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self.train_loader
    def training_step(self, batch,batch_idx) :
        lr_image,hr_image=batch
        fake_hr,fake_lr=self(lr_image,hr_image)
        discriminator_optimizer=self.optimizers()[0]
        generator_optimizer=self.optimizers()[1]
        discriminator_optimizer.zero_grad()
        lr_disc_loss=self.discriminator_loss(discriminator=self.discriminator_hr,generator=self.generator_hr,
                                        fake=lr_image,real=hr_image)
        hr_disc_loss=self.discriminator_loss(discriminator=self.discriminator_lr,generator=self.generator_lr,
                                        fake=hr_image,real=lr_image)
        total_disc_loss=hr_disc_loss+lr_disc_loss
        self.manual_backward(total_disc_loss)
        discriminator_optimizer.step()

        generator_optimizer.zero_grad()
        lr_gen_loss=self.generator_loss(self.discriminator_lr,fake_lr)
        hr_gen_loss=self.generator_loss(self.discriminator_hr,fake_hr)
        cycled_lr=self.generator_lr(fake_hr)
        cycled_hr=self.generator_hr(fake_lr)
        cycled_lr_loss=self.cycle_loss(real=lr_image,cycled=cycled_lr)
        cycled_hr_loss=self.cycle_loss(real=hr_image,cycled=cycled_hr)
        lr_identity=self.generator_lr(lr_image)
        hr_identity=self.generator_hr(hr_image)

        lr_identity_loss=self.identity_loss(real=lr_image,identity=lr_identity)
        hr_identity_loss=self.identity_loss(real=hr_image,identity=hr_identity)
        total_gen_loss=lr_gen_loss+hr_gen_loss+(cycled_hr_loss*self.cycle_lambda)+(cycled_lr_loss*self.cycle_lambda)+(lr_identity_loss*self.identity_lambda)+(hr_identity_loss*self.identity_lambda)
        self.manual_backward(total_gen_loss)
        generator_optimizer.step()
        self.log(name="total_discriminator_loss", value=total_disc_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(name="lr_discriminator_loss", value=lr_disc_loss, prog_bar=True, logger=True, on_step=True,
                 on_epoch=True)
        self.log(name="hr_discriminator_loss", value=hr_disc_loss, prog_bar=True, logger=True, on_step=True,
                 on_epoch=True)

        self.log(name="total_gen_loss", value=total_gen_loss, prog_bar=True, logger=True, on_step=True,
                     on_epoch=True)
        self.log(name="lr_identity_loss", value=lr_identity_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(name="hr_identity_loss", value=hr_identity_loss, prog_bar=True, logger=True, on_step=True,
                     on_epoch=True)
        self.log(name="lr_cycle_loss", value=cycled_lr_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(name="hr_cycle_loss", value=cycled_hr_loss, prog_bar=True, logger=True, on_step=True,
                     on_epoch=True)
        self.log(name="lr_generator_loss", value=lr_gen_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(name="hr_generator_loss", value=hr_gen_loss, prog_bar=True, logger=True, on_step=True,
                     on_epoch=True)

        return {"total_generator_loss":total_gen_loss,"total_discriminator_loss":total_disc_loss}


    def configure_optimizers(self):
        self.hparams.lr = self.learning_rate
        discriminator_optimizer=torch.optim.Adam(lr=self.learning_rate,
                                                      params=list(self.discriminator_lr.parameters())+list(self.discriminator_hr.parameters()))
        generator_optimizer=torch.optim.Adam(lr=self.learning_rate,
                                                  params=list(self.generator_lr.parameters())+list(self.generator_lr.parameters()))
        return [discriminator_optimizer, generator_optimizer]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


