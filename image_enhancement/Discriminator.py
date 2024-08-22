import torch, torchinfo
from UW_CycleGAN.ResidualBlock import ResidualBlock
from UW_CycleGAN.GeneratorEncoderBlock import ConvBlock


class Discriminator(torch.nn.Module):
    def __init__(self, in_channels: int = 3, kernel_size: int = 4, stride: int = 2, padding: int = 1,
                 out_channels=[64, 128, 256, 512]):
        super().__init__()
        self.initial_layer = torch.nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=out_channels[0],
                      stride=stride, padding=padding,
                      kernel_size=kernel_size,
                      use_activation=True, use_bn=False)

        )
        in_channels = out_channels[0]
        layers = []
        for channel in out_channels[1:]:
            layers.append(ConvBlock(in_channels=in_channels,
                                    out_channels=channel,
                                    stride=1 if channel == out_channels[-1] else 2,
                                    use_bn=True,
                                    use_activation=True,
                                    kernel_size=kernel_size,
                                    padding=padding))
            in_channels = channel
        layers.append(
            ConvBlock(in_channels=in_channels,
                      out_channels=1,
                      kernel_size=kernel_size,
                      stride=1,
                      padding=1,
                      use_activation=False,
                      use_bn=False, ))
        self.disc = torch.nn.Sequential(*layers)
        self.activation = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial_layer(x)
        return self.activation(self.disc(x))

""""
model = Discriminator()
x = torch.randn(size=(5, 3, 720, 720))
pred = model(x)
print(pred.shape)
torchinfo.summary(model=model, input_data=x)
"""