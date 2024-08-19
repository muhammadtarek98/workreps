import torch, torchinfo
import Configs
from GeneratorEncoderBlock import ConvBlock
from ResidualBlock import ResidualBlock
from UW_CycleGAN.Configs import out_feature

class GeneratorEncoder(torch.nn.Module):
    def __init__(self,
                 num_residual_blocks: int,
                 in_channels: int = 3,
                 out_channels: int = 64
                 ):
        super().__init__()
        self.initial_block = ConvBlock(use_activation=True, use_bn=False,
                                       in_channels=in_channels,
                                       out_channels=out_feature[0],
                                       kernel_size=7, stride=1, padding=3)
        self.down_sampling = torch.nn.ModuleList(
            [
                ConvBlock(out_channels=out_feature[1],
                          in_channels=out_feature[0],
                          kernel_size=3, stride=2, padding=1,
                          use_activation=True, use_bn=True),

                ConvBlock(out_channels=out_feature[2],
                          in_channels=out_feature[1],
                          kernel_size=3, stride=2, padding=1,
                          use_activation=True, use_bn=True)
            ]
        )
        self.residual_block = torch.nn.Sequential(
            *[ResidualBlock(channels=out_feature[2], use_bn=True, use_activation=True) for _ in
              range(num_residual_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial_block(x)
        print(x.shape)
        for layer in self.down_sampling:
            x = layer(x)

        return self.residual_block(x)


"""
model = GeneratorEncoder(num_residual_blocks=9)
x = torch.randn((1,3,720,720))
torchinfo.summary(model, input_data=x)
"""
