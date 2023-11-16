import torch.nn as nn
import torch
import hyperparamters
import class_with_image_patch_embedding
import transformer_encoder
import image_patch_embedding
class ViT(nn.Module):
    def __init__(self,imgsz,patch_size,embedding_dropout,in_channels,embedding_dim,num_classes):
        super(ViT,self).__init__()


        self.num_patches=(imgsz*imgsz)//patch_size
        self.class_embedding = nn.Parameter(data=torch.randn(size=(1, 1, embedding_dim), requires_grad=True))

        self.positional_embedding=nn.Parameter(data=torch.randn(size=(1,1,self.num_patches+1),requires_grad=True))
        self.embedding_dropout=nn.Dropout(p=embedding_dropout)
        self.patch_embedding=image_patch_embedding.Patch_Embedding(in_channels=in_channels,patch_size=patch_size,embedding_dim=embedding_dim)
        self.transformer_encoder=nn.Sequential(*[
            transformer_encoder.Transformer_Encoder()
        ])
        self.classifier=nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim,
                      out_features=num_classes)
        )
    def forward(self,x):
        batch_size=x[0]
        class_token=self.class_embedding.expand((batch_size,1,-1))
        x=self.patch_embedding(x)
        x=torch.cat((class_token,x),dim=1)
        x=self.positional_embedding+x
        x=self.embedding_dropout(x)
        x=self.transformer_encoder(x)
        x=self.classifier(x[:,0])
        return x
tensor=torch.randn(size=(hyperparamters.batch_size,hyperparamters.h,hyperparamters.w,hyperparamters.c))
model=ViT(
    imgsz=hyperparamters.h,
    patch_size=hyperparamters.p,
    embedding_dim=hyperparamters.e,
    in_channels=hyperparamters.c,
    num_classes=hyperparamters.num_classes,
    embedding_dropout=hyperparamters.embedding_dropout
)
print(model(tensor))


