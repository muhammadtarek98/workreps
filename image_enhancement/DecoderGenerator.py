from GeneratorDecoderBlock import ConvTransposeBlock
import torch, torchinfo, Configs


class GeneratorDecoder(torch.nn.Module):
    def __init__(self,
                 stride: int = 2,
                 padding: int = 1,
                 kernel_size: int = 4,
                 num_blocks: int = 4):
        super().__init__()
        self.deconv_blocks = torch.nn.ModuleList()
        for i in reversed(range(num_blocks)):
            self.deconv_blocks.append(
                ConvTransposeBlock(in_channels=Configs.out_feature[i],
                                   out_channels=3 if i == 0 else Configs.out_feature[i - 1],
                                   stride=stride,
                                   padding=padding,
                                   kernel_size=kernel_size,
                                   last_layer=True if i == 0 else False)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, block in enumerate(self.deconv_blocks):
            #skip = skips[-(i + 1)]  # Get the corresponding skip connection
            #x = torch.cat([x, skip], dim=1)  # Concatenate along the channel dimension
            x = block(x)
        return x

"""
model = GeneratorDecoder()
x = torch.randn((1, 512, 45, 45))
torchinfo.summary(model=model, input_data=x)
"""