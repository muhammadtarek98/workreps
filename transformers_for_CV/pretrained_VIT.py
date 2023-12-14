import torch
import torchvision
from hyperparamters import *
from torchinfo import summary
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
tensor = torch.randn(size=(1, 3, height, width))
pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
pretrained_vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights).to(device)


for parameter in pretrained_vit.parameters():
    parameter.requires_grad = True

pretrained_vit.heads = torch.nn.Linear(in_features=768, out_features=10).to(device)

summary(pretrained_vit)