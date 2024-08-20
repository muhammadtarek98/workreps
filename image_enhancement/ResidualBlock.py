import torch, torchinfo
from GeneratorEncoderBlock import ConvBlock


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 use_bn: bool = True,
                 use_activation: bool = False):
        super(ResidualBlock, self).__init__()
        self.conv_block = torch.nn.Sequential(
            ConvBlock(in_channels=channels,
                      out_channels=channels,
                      use_activation=True,
                      use_bn=use_bn,
                      kernel_size=kernel_size,
                      padding=padding, stride=stride),
            ConvBlock(in_channels=channels,
                      out_channels=channels,
                      use_activation=False,
                      use_bn=use_bn,
                      kernel_size=kernel_size, padding=padding,
                      stride=stride
                      )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.conv_block(x)
        return out + x

"""
model = ResidualBlock(channels=3,
                      kernel_size=3, stride=1,
                      use_activation=True, padding=1)
x = torch.randn(size=(1, 3, 720, 720))
torchinfo.summary(model=model, input_data=x)
"""