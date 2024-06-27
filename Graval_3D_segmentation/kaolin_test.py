import kaolin
from torch_kmeans import KMeans
import open3d as o3d
import torch
import numpy as np
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
pcl=o3d.io.read_point_cloud("/home/cplus/projects/m.tarek_master/test_ascii.ply")

print(len(np.asarray(pcl.points)))
print(pcl.normals)
o3d.visualization.draw_geometries([pcl])

#print(kaolin.__version__)
print(torch.__version__)
