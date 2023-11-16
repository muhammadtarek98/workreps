import torch.nn as nn
import torch
from image_patch_embedding import Patch_Embedding
import hyperparamters
#print(e)
img=torch.randn(size=(hyperparamters.c,hyperparamters.h,hyperparamters.w))
img=img.unsqueeze(0)
patch_embedding_layer=Patch_Embedding(in_channels=hyperparamters.c,patch_size=hyperparamters.p,embedding_dim=hyperparamters.e)
patch_embedding=patch_embedding_layer(img)
#print(patch_embedding.shape)
#create class label token
batch_size=patch_embedding.shape[0]
embedding_dim=patch_embedding.shape[-1]
class_token=nn.Parameter(data=torch.randn(size=(batch_size,1,embedding_dim),requires_grad=True))

#add class token to patch embedding

patch_embedding_with_class_token=torch.cat(tensors=(class_token,patch_embedding),dim=1)

#print(patch_embedding_with_class_token.shape)
#create positional embedding
positional_embedding=nn.Parameter(torch.randn(size=(1,hyperparamters.num_patches+1,embedding_dim),requires_grad=True))

patch_and_positional_embeddings=patch_embedding_with_class_token+positional_embedding

#print(patch_and_positional_embeddings.shape)