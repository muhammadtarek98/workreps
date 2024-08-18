import torch, torchinfo
import Configs
from GeneratorEncoderBlock import ConvBlock


class GeneratorEncoder(torch.nn.Module):
    def __init__(self, in_channels: int = 3,
                 kernel_size: int = 4,
                 stride: int = 2,
                 padding: int = 1,
                 num_blocks: int = 4):
        super().__init__()
        self.out_channels: list = Configs.out_feature
        self.conv_blocks = torch.nn.ModuleList()
        for i in range(num_blocks):
            self.conv_blocks.append(
                ConvBlock(in_channels=in_channels if i == 0 else self.out_channels[i],
                          out_channels=self.out_channels[i + 1],
                          stride=stride,
                          padding=padding,
                          use_activation=False if i == 0 else True,
                          kernel_size=kernel_size
                          )
            )

    def forward(self, x: torch.Tensor) ->dict:
        feature_maps=[]
        for block in self.conv_blocks:
            feature_maps.append(x)
            x = block(x)
        return [feature_maps,x]

"""
model = GeneratorEncoder()
x = torch.randn((1, *Configs.input_shape))
torchinfo.summary(model, input_data=x)
"""