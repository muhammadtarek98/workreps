import torch
import torch.nn as nn
import class_with_image_patch_embedding
import hyperparamters


class MSA_Block(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int):
        super(MSA_Block, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape=embedding_dim)
        self.msa = nn.MultiheadAttention(embed_dim=embedding_dim,
                                         num_heads=num_heads)

    def forward(self, x):
        x = self.ln(x)
        attention_map, attention_output_weights = self.msa(
            query=x,
            value=x,
            key=x,
            need_weights=False
        )

        return attention_map


msa_block = MSA_Block(embedding_dim=hyperparamters.embedding_size, num_heads=hyperparamters.num_heads)
test = msa_block(class_with_image_patch_embedding.patch_and_positional_embeddings)

# print(class_with_image_patch_embedding.patch_and_positional_embeddings.shape)
# print(test.shape)
