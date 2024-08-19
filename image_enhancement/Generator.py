
import torch, torchinfo

from GeneratorEncoder import GeneratorEncoder
from DecoderGenerator import GeneratorDecoder
from UW_CycleGAN.ResidualBlock import ResidualBlock


class Generator(torch.nn.Module):
    def __init__(self, num_residual_blocks: int = 9):
        super().__init__()
        self.encoder = GeneratorEncoder(num_residual_blocks=num_residual_blocks)
        self.decoder = GeneratorDecoder(in_channels=256,out_channels=256//2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x


model = Generator()
x = torch.randn(size=(1, 3, 720, 720))
torchinfo.summary(model=model, input_data=x)