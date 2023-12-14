import torch.nn as nn
import torch
from image_patch_embedding import Patch_Embedding
from hyperparamters import *


img = torch.randn(size=(channels, height,width))
img = img.unsqueeze(0)
patch_embedding_layer = Patch_Embedding(in_channels=channels, patch_size=patch_size,
                                        embedding_dim=embedding_size)

patch_embedding = patch_embedding_layer(img)
#print(patch_embedding.shape)
# create class label token
batch_size = patch_embedding.shape[0]
embedding_dim = patch_embedding.shape[-1]
class_token = nn.Parameter(data=torch.randn(size=(batch_size, 1, embedding_dim), requires_grad=True))

# add class token to patch embedding

patch_embedding_with_class_token = torch.cat(tensors=(class_token, patch_embedding), dim=1)

#print(patch_embedding_with_class_token.shape)
# create positional embedding
positional_embedding = nn.Parameter(
    torch.randn(size=(batch_size,num_patches + 1, embedding_dim), requires_grad=True))#.unsqueeze(1)

patch_and_positional_embeddings = patch_embedding_with_class_token + positional_embedding

#print(patch_and_positional_embeddings.shape)
