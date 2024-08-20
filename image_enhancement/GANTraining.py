import pytorch_lightning as pl
import torchmetrics.image
import torch, torchvision
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS, OptimizerLRScheduler, STEP_OUTPUT

from Discriminator import Discriminator
from Generator import Generator


def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            torch.nn.init.normal_(m.weight, 1.0, init_gain)
            torch.nn.init.constant_(m.bias, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>


class GAN(pl.LightningModule):
    def __init__(self,
                 train_loader=None,
                 val_loader=None,
                 lr: float = 1e-4,
                 input_shape=None,
                 cycle_lambda=10,
                 identity_lambda=0.5):
        super().__init__()
        self.input_shape = input_shape
        self.cycle_lambda = cycle_lambda
        self.identity_lambda = identity_lambda
        self.save_hyperparameters()
        self.generator_lr = Generator()
        self.generator_hr = Generator()
        self.discriminator_lr = Discriminator()
        self.discriminator_hr = Discriminator()
        self.learning_rate = lr
        init_weights(self.discriminator_hr)
        init_weights(self.discriminator_lr)
        init_weights(self.generator_hr)
        init_weights(self.generator_lr)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.l1_loss = torch.nn.L1Loss()
        self.mse_loss = torch.nn.MSELoss()
        self.automatic_optimization = False

    def forward(self, lr, hr):
        fake_lr = self.generator_lr(hr)
        fake_hr = self.generator_hr(lr)
        return fake_hr, fake_lr

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self.val_loader

    def discriminator_loss(self, discriminator, generator, fake, real):
        gen = generator(fake)
        real = discriminator(real)
        fake = discriminator(gen.detach())
        real_loss = self.mse_loss(real, torch.ones_like(real))
        fake_loss = self.mse_loss(fake, torch.zeros_like(fake))
        total_loss = (real_loss + fake_loss) / 2
        return total_loss

    def generator_loss(self, discriminator, fake):
        disc_fake = discriminator(fake)
        loss = self.mse_loss(disc_fake, torch.ones_like(disc_fake))
        return loss

    def cycle_loss(self, real, cycled):
        loss = self.l1_loss(real, cycled)
        return loss

    def identity_loss(self, real, identity):
        loss = self.l1_loss(real, identity)
        return loss

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self.train_loader

    def training_step(self, batch, batch_idx):
        lr_image, hr_image = batch[0], batch[1]
        fake_hr, fake_lr = self(lr_image, hr_image)
        hr_discriminator_optimizer = self.optimizers()[0]
        lr_discriminator_optimizer = self.optimizers()[1]
        lr_generator_optimizer = self.optimizers()[2]
        hr_generator_optimizer = self.optimizers()[3]
        hr_discriminator_optimizer.zero_grad()
        lr_discriminator_optimizer.zero_grad()
        lr_disc_loss = self.discriminator_loss(discriminator=self.discriminator_hr, generator=self.generator_hr,
                                               fake=lr_image, real=hr_image)
        hr_disc_loss = self.discriminator_loss(discriminator=self.discriminator_lr, generator=self.generator_lr,
                                               fake=hr_image, real=lr_image)
        total_disc_loss = hr_disc_loss + lr_disc_loss

        self.manual_backward(total_disc_loss, retain_graph=True)
        hr_discriminator_optimizer.step()
        lr_discriminator_optimizer.step()

        lr_generator_optimizer.zero_grad()
        hr_generator_optimizer.zero_grad()
        lr_gen_loss = self.generator_loss(self.discriminator_lr, fake_lr)
        hr_gen_loss = self.generator_loss(self.discriminator_hr, fake_hr)
        cycled_lr = self.generator_lr(fake_hr)
        cycled_hr = self.generator_hr(fake_lr)
        cycled_lr_loss = self.cycle_loss(real=lr_image, cycled=cycled_lr)
        cycled_hr_loss = self.cycle_loss(real=hr_image, cycled=cycled_hr)
        lr_identity = self.generator_lr(lr_image)
        hr_identity = self.generator_hr(hr_image)

        lr_identity_loss = self.identity_loss(real=lr_image, identity=lr_identity)
        hr_identity_loss = self.identity_loss(real=hr_image, identity=hr_identity)
        total_gen_loss = lr_gen_loss + hr_gen_loss + (cycled_hr_loss * self.cycle_lambda) + (
                cycled_lr_loss * self.cycle_lambda) + (lr_identity_loss * self.identity_lambda) + (
                                 hr_identity_loss * self.identity_lambda)
        self.manual_backward(total_gen_loss, retain_graph=True)
        lr_generator_optimizer.step()
        hr_generator_optimizer.step()
        self.log(name="total_discriminator_loss", value=total_disc_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(name="lr_discriminator_loss",
                 value=lr_disc_loss,
                 prog_bar=True, logger=True,
                 on_step=True,
                 on_epoch=True)
        self.log(name="hr_discriminator_loss",
                 value=hr_disc_loss,
                 prog_bar=True,
                 logger=True, on_step=True,
                 on_epoch=True)
        self.log_predictions_to_tensorboard(lr_image=lr_image,
                                            gen_lr=fake_hr,
                                            real_hr_image=hr_image,
                                            cycled_hr=cycled_hr,
                                            cycled_lr=cycled_lr)

        self.log(name="total_gen_loss",
                 value=total_gen_loss,
                 prog_bar=True, logger=True,
                 on_step=True,
                 on_epoch=True)
        self.log(name="lr_identity_loss",
                 value=lr_identity_loss,
                 on_step=True, on_epoch=True,
                 prog_bar=True)
        self.log(name="hr_identity_loss",
                 value=hr_identity_loss,
                 prog_bar=True, logger=True, on_step=True,
                 on_epoch=True)
        self.log(name="lr_cycle_loss",
                 value=cycled_lr_loss,
                 on_step=True, on_epoch=True,
                 prog_bar=True)
        self.log(name="hr_cycle_loss",
                 value=cycled_hr_loss, prog_bar=True, logger=True, on_step=True,
                 on_epoch=True)
        self.log(name="lr_generator_loss",
                 value=lr_gen_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(name="hr_generator_loss",
                 value=hr_gen_loss,
                 prog_bar=True, logger=True,
                 on_step=True,
                 on_epoch=True)

        return {"total_generator_loss": total_gen_loss, "total_discriminator_loss": total_disc_loss}

    def configure_optimizers(self):
        self.hparams.lr = self.learning_rate
        hr_discriminator_optimizer = torch.optim.Adam(lr=self.learning_rate,
                                                      params=self.discriminator_hr.parameters())
        lr_discriminator_optimizer = torch.optim.Adam(lr=self.learning_rate,
                                                      params=self.discriminator_lr.parameters())
        lr_generator_optimizer = torch.optim.Adam(lr=self.learning_rate,
                                                  params=self.generator_lr.parameters())
        hr_generator_optimizer = torch.optim.Adam(lr=self.learning_rate,
                                                  params=self.generator_hr.parameters())
        return [hr_discriminator_optimizer, lr_discriminator_optimizer, lr_generator_optimizer, hr_generator_optimizer]

    def denormalize(self,tensor):
        return (tensor +1.0)*255.0
    def log_predictions_to_tensorboard(self, cycled_hr,
                                       cycled_lr, lr_image, gen_lr, real_hr_image, mode='train'):
        lr_image = torchvision.utils.make_grid(tensor=lr_image, normalize=False)
        real_hr_image = torchvision.utils.make_grid(tensor=real_hr_image, normalize=False)
        gen_lr = torchvision.utils.make_grid(tensor=gen_lr, normalize=False)
        cycled_lr = torchvision.utils.make_grid(tensor=cycled_lr, normalize=False)
        cycled_hr = torchvision.utils.make_grid(tensor=cycled_hr, normalize=False)

        self.logger.experiment.add_image(f'{mode}_low_quality_image', lr_image,
                                         self.current_epoch)
        self.logger.experiment.add_image(f'{mode}_real_high_quality_image', real_hr_image,
                                         self.current_epoch)
        self.logger.experiment.add_image(f'{mode}_generated_high_quality_image', gen_lr,
                                         self.current_epoch)
        self.logger.experiment.add_image(f'{mode}_cycle_low_quality_image', cycled_lr,
                                         self.current_epoch)
        self.logger.experiment.add_image(f'{mode}_cycle_high_quality_image', cycled_hr,
                                         self.current_epoch)
