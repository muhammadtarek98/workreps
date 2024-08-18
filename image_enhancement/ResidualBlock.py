import torch
from GeneratorEncoderBlock import ConvBlock  # Import the ConvBlock class


class ResidualBlock(torch.nn.Module):
    def __init__(self,out_channels:int,
                 in_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 use_activation: bool = False):
        super(ResidualBlock, self).__init__()
        self.conv_block = ConvBlock(use_activation=use_activation,
                                    in_channels=in_channels,
                                    out_channels=out_channels,
                                    stride=stride,
                                    padding=padding,
                                    kernel_size=kernel_size)
        self.conv_block_2=ConvBlock(use_activation=use_activation,
                                    in_channels=out_channels,
                                    out_channels=out_channels,
                                    stride=stride,
                                    padding=padding,
                                    kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out:torch.Tensor=self.conv_block(x)
        out=self.conv_block_2(out)
        return x + out
