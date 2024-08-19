import torch
import torchinfo


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 padding: int,
                 use_bn:bool,
                 use_activation: bool):
        super().__init__()
        self.use_activation = use_activation
        self.conv = torch.nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    bias=True,
                                    padding_mode="reflect")
        self.activation = None
        self.bn=None
        if self.use_activation:
            self.activation = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            self.activation = torch.nn.Identity()
        if use_bn:
            self.bn = torch.nn.InstanceNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.use_activation:
            x = self.activation(x)
        return x

"""
model = ConvBlock(in_channels=3, out_channels=64,
                  kernel_size=4, stride=1,
                  use_activation=True, padding=1)
x = torch.randn(size=(1, 3, 720, 720))
torchinfo.summary(model=model, input_data=x)
"""