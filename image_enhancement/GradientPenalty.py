import torch
import numpy as np
from UW_CycleGAN.Discriminator import Discriminator
from UW_CycleGAN.Generator import Generator
from UW_CycleGAN import Configs
from UW_CycleGAN.Dataset import CustomDataset


class GradientPenalty(torch.nn.Module):
    def __init__(self, eps, real: torch.Tensor, fake: torch.Tensor, discriminator: torch.nn.Module):
        super(GradientPenalty, self).__init__()
        self.real: torch.Tensor = real
        self.fake: torch.Tensor = fake
        self.discriminator = discriminator
        self.batch_size, self.channels, self.height, self.width = self.real.shape
        self.x_hat = eps * self.real + (1 - eps) * self.fake

    def forward(self):
        D_hat = self.discriminator(self.x_hat)
        gradient = torch.autograd.grad(inputs=self.x_hat,
                                       outputs=D_hat,
                                       retain_graph=True,
                                       create_graph=True,
                                       grad_outputs=torch.ones_like(D_hat))[0]
        gradient_norm = gradient.norm(gradient.shape[0], dim=1)
        gradient_penality = torch.mean((gradient_norm - 1) ** 2)
        return gradient_penality


random_input = torch.randn(size=(2, 3, 128, 128))
gen=Generator(num_residual_blocks=5)
disc = Discriminator()
root_dir = "/home/cplus/projects/m.tarek_master/Image_enhancement/dataset/underwater_imagenet"
transform = Configs.transform
data_set = CustomDataset(images_dir=root_dir, transform=transform)

data_loader = torch.utils.data.DataLoader(dataset=data_set,
                                          batch_size=Configs.batch_size,
                                          shuffle=True,
                              num_workers=4)
iter=iter(data_loader)
lr,hr=next(iter)
fake=gen(lr)
eps=torch.randn()
print(eps)
gp=GradientPenalty(eps=eps)
