import torch.nn as nn
from multi_head_self_attention import MSA_Block
from MLP import MLP
import hyperparamters


class Transformer_Encoder(nn.Module):
    def __init__(self, embedding_dim:int=hyperparamters.embedding_size, num_heads:int=hyperparamters.num_heads,
                 mlp_size:int=hyperparamters.mlp_size, mlp_dropout:float=hyperparamters.mlp_dropout):
        super(Transformer_Encoder, self).__init__()
        self.msa_block = MSA_Block(embedding_dim=embedding_dim, num_heads=num_heads)
        self.mlp_block = MLP(embedding_dim=embedding_dim, mlp_size=mlp_size, drop_out=mlp_dropout)

    def forward(self, x):
        x = self.msa_block(x) + x
        x = self.mlp_block(x) + x
        return x


transformer_encoder = nn.TransformerEncoderLayer(d_model=hyperparamters.embedding_size,
                                                 nhead=hyperparamters.num_heads,
                                                 activation="gelu",
                                                 dropout=hyperparamters.mlp_dropout,
                                                 dim_feedforward=hyperparamters.mlp_size
                                                 )

# model=Transformer_Encoder()
# print(transformer_encoder)
# print(model)
