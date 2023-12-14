
import torch
height = 224
width = 224
channels = 3
patch_size = 16
embedding_size = 768
num_patches = int((height * width) / patch_size ** 2)
num_heads = 12
mlp_size = 3072
mlp_dropout = 0.1
batch_size = 32
num_classes = 10
embedding_dropout = 0.1
lr=1e-3
beta=(0.9,0.98)
#optimizer= torch.optim.Adam(VIT.ViT.parameters(),lr=lr,betas=beta,weight_decay=0.3)
loss=torch.nn.CrossEntropyLoss()
