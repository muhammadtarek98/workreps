import torch.nn as nn
from multi_head_self_attention import MSA_Block
from MLP import MLP
import hyperparamters


class Transformer_Encoder(nn.Module):
    def __init__(self):
        super(Transformer_Encoder,self).__init__()
        self.msa_block = MSA_Block(embedding_dim=hyperparamters.e,num_heads=hyperparamters.num_heads)
        self.mlp_block = MLP(embedding_dim=hyperparamters.e,mlp_size=hyperparamters.mlp_size,drop_out=hyperparamters.mlp_dropout)

    def forward(self, x):
        x = self.msa_block(x) + x
        x = self.mlp_block(x) + x
        return x
