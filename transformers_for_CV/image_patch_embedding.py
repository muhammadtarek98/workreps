import torch
import torch.nn as nn


class Patch_Embedding(nn.Module):
    def __init__(self, in_channels:int, patch_size:int, embedding_dim:int):
        super(Patch_Embedding, self).__init__()

        self.in_channels = in_channels
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.conv2d = nn.Conv2d(in_channels=self.in_channels,
                                out_channels=self.embedding_dim,
                                stride=self.patch_size,
                                padding=0,
                                kernel_size=self.patch_size)
        self.activation=nn.ReLU(inplace=True)
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)

    def forward(self, x: torch.tensor):
        x = self.conv2d(x)
        #x=self.activation(x)
        #x=self.pool(x)
        x = self.flatten(x)
        x = x.permute(0, 2, 1)
        return x


"""
test=torch.randn((1,3,224,224))
in_channels=3
img_sz=224
patch_size=16
out_channels=768
print(out_channels)
model=Patch_embedding(in_channels=in_channels,embedding_dim=out_channels,patch_size=patch_size)
result=model(test)
print(result.shape)
"""
