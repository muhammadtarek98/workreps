from UW_CycleGAN.GeneratorDecoderBlock import ConvTransposeBlock
import torch, torchinfo
from UW_CycleGAN import Configs

from UW_CycleGAN.GeneratorEncoderBlock import ConvBlock


class GeneratorDecoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up_sampling = torch.nn.ModuleList(
            [
                ConvTransposeBlock(in_channels=in_channels, out_channels=128, stride=2),
                ConvTransposeBlock(in_channels=128, out_channels=64, stride=2),
            ]
        )
        self.final_block = ConvBlock(in_channels=64,
                                     out_channels=out_channels,
                                     kernel_size=4, stride=1, padding=3, use_bn=False, use_activation=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.up_sampling:
            x = layer(x)
        x = self.final_block(x)
        return x
"""
model = GeneratorDecoder()
x = torch.randn((1, 512, 45, 45))
torchinfo.summary(model=model, input_data=x)
"""