import torch.nn as nn
import torch
from hyperparamters import *
import class_with_image_patch_embedding
import transformer_encoder
import image_patch_embedding
import torch,torchvision

class ViT(nn.Module):
    """Creates a Vision Transformer architecture with ViT-Base hyperparameters by default."""
    def __init__(self,
                 img_size: int,  # Training resolution from Table 3 in ViT paper
                 in_channels: int,  # Number of channels in input image
                 patch_size: int,  # Patch size
                 num_transformer_layers: int,  # Layers from Table 1 for ViT-Base
                 embedding_dim: int,  # Hidden size D from Table 1 for ViT-Base
                 mlp_size: int,  # MLP size from Table 1 for ViT-Base
                 num_heads: int,  # Heads from Table 1 for ViT-Base
                 attn_dropout: float,  # Dropout for attention projection
                 mlp_dropout: float,  # Dropout for dense/MLP layers
                 embedding_dropout: float,  # Dropout for patch and position embeddings
                 num_classes: int):  # Default for ImageNet but can customize this
        super().__init__()


        # 4. Calculate number of patches (height * width/patch^2)
        self.num_patches = (img_size * img_size) // patch_size ** 2

        # 5. Create learnable class embedding (needs to go at front of sequence of patch embeddings)
        self.class_embedding = nn.Parameter(data=torch.randn(1, 1, embedding_dim),
                                            requires_grad=True)

        # 6. Create learnable position embedding
        self.position_embedding = nn.Parameter(data=torch.randn(1, self.num_patches + 1, embedding_dim),
                                               requires_grad=True)

        # 7. Create embedding dropout value
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        # 8. Create patch embedding layer
        self.patch_embedding = image_patch_embedding.Patch_Embedding(in_channels=in_channels,
                                                                     patch_size=patch_size,
                                                                     embedding_dim=embedding_dim)
        self.transformer_encoder = nn.Sequential(
            *[transformer_encoder.Transformer_Encoder(embedding_dim=hyperparamters.embedding_size,
                                                      num_heads=num_heads,
                                                      mlp_size=mlp_size,
                                                      mlp_dropout=mlp_dropout) for _ in range(num_transformer_layers)])

        # 10. Create classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim,
                      out_features=num_classes)
        )

    def forward(self, x):
        batch_size = x.shape[0]

        # 13. Create class token embedding and expand it to match the batch size (equation 1)
        class_token = self.class_embedding.expand(batch_size, -1,
                                                  -1)

        # 14. Create patch embedding (equation 1)
        x = self.patch_embedding(x)

        # 15. Concat class embedding and patch embedding (equation 1)
        x = torch.cat((class_token, x), dim=1)

        # 16. Add position embedding to patch embedding (equation 1)
        x = self.position_embedding + x


        x = self.embedding_dropout(x)

        # 18. Pass patch, position and class embedding through transformer encoder layers (equations 2 & 3)
        x = self.transformer_encoder(x)


        x = self.classifier(x[:, 0])

        return x

"""
#tensor = torch.randn(size=(hyperparamters.batch_size, hyperparamters.channels, hyperparamters.height, hyperparamters.width))
model = ViT(
    img_size=hyperparamters.height,
    patch_size=hyperparamters.patch_size,
    embedding_dim=hyperparamters.embedding_size,
    in_channels=hyperparamters.channels,
    num_classes=hyperparamters.num_classes,
    embedding_dropout=hyperparamters.embedding_dropout, num_transformer_layers=12,
    mlp_size=hyperparamters.mlp_size, mlp_dropout=hyperparamters.mlp_dropout,
    num_heads=hyperparamters.num_heads, attn_dropout=0.1
)
#result=model(tensor)
#print(result)
"""


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
tensor = torch.randn(size=(1, 3, height, width))
pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
pretrained_vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights).to(device)


for parameter in pretrained_vit.parameters():
    parameter.requires_grad = False

pretrained_vit.heads = nn.Linear(in_features=768, out_features=10).to(device)