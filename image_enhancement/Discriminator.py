import torch, torchinfo

class ConvBlock(torch.nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size):

class Discriminator(torch.nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 64):
        super().__init__()
        self.b1 = ResidualBlock(in_channels=in_channels,
                                out_channels=out_channels,
                                use_activation=True,
                                kernel_size=4,
                                stride=1,
                                padding=2)
        self.b2 = ResidualBlock(in_channels=in_channels * 2,
                                out_channels=out_channels * 2,
                                kernel_size=4,
                                use_activation=True,
                                stride=1,
                                padding=1)
        self.b3 = ResidualBlock(in_channels=out_channels * 2,
                                out_channels=out_channels * 4,
                                kernel_size=4,
                                use_activation=True,
                                stride=1,
                                padding=0
                                )
        self.b4 = ResidualBlock(in_channels=out_channels * 4,
                                out_channels=out_channels * 8,
                                kernel_size=4,
                                stride=1,
                                padding=1,
                                use_activation=True)
        self.conv = torch.nn.Conv2d(in_channels=out_channels * 8,
                                    out_channels=1,
                                    kernel_size=1,
                                    stride=1)

    def forward(self, x: torch.Tensor):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        return self.conv(x)


model = Discriminator()
x = torch.randn(size=(1, 3, 720, 720))
torchinfo.summary(model=model, input_data=x)
