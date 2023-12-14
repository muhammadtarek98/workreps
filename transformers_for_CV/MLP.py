import torch.nn as nn
import multi_head_self_attention
import hyperparamters


class MLP(nn.Module):
    def __init__(self, embedding_dim: int, mlp_size: int, drop_out: float):
        super(MLP, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape=embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=mlp_size),
            nn.GELU(),
            nn.Dropout(p=drop_out),
            nn.Linear(in_features=mlp_size,
                      out_features=embedding_dim),
            nn.Dropout(p=drop_out)
        )

    def forward(self, x):
        x = self.ln(x)
        x = self.mlp(x)
        return x


model = MLP(embedding_dim=hyperparamters.embedding_size, mlp_size=hyperparamters.mlp_size, drop_out=hyperparamters.mlp_dropout)
output = model(multi_head_self_attention.test)
# print(output.shape)
