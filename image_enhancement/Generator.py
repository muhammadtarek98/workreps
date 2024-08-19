
import torch, torchinfo

from GeneratorEncoder import GeneratorEncoder
from DecoderGenerator import GeneratorDecoder


class Generator(torch.nn.Module):
    def __init__(self, num_residual_blocks: int = 9):
        super().__init__()
        self.encoder = GeneratorEncoder(num_residual_blocks=num_residual_blocks)
        self.decoder = GeneratorDecoder(in_channels=256, out_channels=3)  # Adjusted to ensure output has 3 channels (same as input)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = Generator()
x = torch.randn(size=(1, 3, 720, 720))
torchinfo.summary(model=model, input_data=x)
"""# Assuming a 256x256 RGB image
input_tensor = torch.randn(1, 3, 256, 256)  # Batch size 1, 3 channels (RGB), 256x256 image
model = Generator(num_residual_blocks=9)
output_tensor = model(input_tensor)

print(f"Input shape: {input_tensor.shape}")
print(f"Output shape: {output_tensor.shape}")"""
