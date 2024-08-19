from GeneratorDecoderBlock import ConvTransposeBlock
import torch, torchinfo, Configs


class GeneratorDecoder(torch.nn.Module):
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 stride: int = 2,
                 padding: int = 1,
                 kernel_size: int = 4,
                 image_channels: int = 3):
        super().__init__()
        self.deconv_blocks = torch.nn.ModuleList([
            ConvTransposeBlock(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,padding=padding),
            ConvTransposeBlock(in_channels=out_channels,
                               out_channels=out_channels//2,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding)]
        )
        self.output_layer=ConvTransposeBlock(last_layer=True,
                                             in_channels=out_channels//2,
                                             out_channels=image_channels,
                                             stride=stride,
                                             kernel_size=7,
                                             padding=3)
    def forward(self,x:torch.Tensor)->torch.Tensor:
        for layer in self.deconv_blocks:
            x=layer(x)
        return self.output_layer(x)

"""
model = GeneratorDecoder()
x = torch.randn((1, 512, 45, 45))
torchinfo.summary(model=model, input_data=x)
"""