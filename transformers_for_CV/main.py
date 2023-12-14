import torch
from VIT import ViT
import hyperparamters
from torchinfo import summary
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
summary(model=model, input_size=(hyperparamters.batch_size, hyperparamters.channels, hyperparamters.height, hyperparamters.width), device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"])