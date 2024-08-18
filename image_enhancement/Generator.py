from pyexpat import features

import torch,torchinfo

from GeneratorEncoder import GeneratorEncoder
from DecoderGenerator import GeneratorDecoder
from ResidualBlock import ResidualBlock


class Generator(torch.nn.Module):
    def __init__(self, num_residual_blocks:int =9):
        super().__init__()
        self.encoder = GeneratorEncoder()
        #self.skip_connection=ResidualBlock()

        self.decoder = GeneratorDecoder()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)[1]
        x = self.decoder(x)
        return x



model = Generator()
x = torch.randn(size=(1, 3, 720, 720))
pred=model(x)
print(x.shape)
print(pred.shape)
#torchinfo.summary(model=model, input_data=x.shape)
