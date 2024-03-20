import kaolin
from torch_kmeans import KMeans
import torch
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print(kaolin.__version__)