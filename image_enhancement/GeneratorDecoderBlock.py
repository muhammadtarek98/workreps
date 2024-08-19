import torch
import torchinfo


class ConvTransposeBlock(torch.nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 stride: int,
                 kernel_size: int = 3,
                 padding: int = 1,
                 last_layer: bool = False):
        super().__init__()
        self.convtranspose = torch.nn.ConvTranspose2d(in_channels=in_channels,
                                                      out_channels=out_channels,
                                                      kernel_size=kernel_size,
                                                      padding=padding,
                                                      stride=stride)
        self.bn = torch.nn.InstanceNorm2d(out_channels)
        self.activation = None
        if last_layer:
            self.activation = torch.nn.Tanh()  # Output image needs to be between -1 and 1
        else:
            self.activation = torch.nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convtranspose(x)
        x = self.bn(x)
        return self.activation(x)
"""
model = ConvTransposeBlock(in_channels=64, out_channels=3, stride=2, last_layer=True)
x = torch.randn((1,64, 720, 720))
pred=model(x)
print(pred.shape)
torchinfo.summary(model=model, input_data=x)
"""