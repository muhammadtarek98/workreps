from torch_geometric.transforms import SamplePoints, KNNGraph
import torch
import torch_geometric
import networkx as nx
import torch_geometric.transforms as T
import open3d
from matplotlib import pyplot as plt
from torch_geometric.utils.convert import to_networkx
from torch_geometric.datasets import GeometricShapes
from torch_geometric.io import read_ply
import numpy as np

dataset=GeometricShapes(root="../")
s="/home/cplus/projects/m.tarek_master/graval_detection_3D/point_cloud_segmentation/regions_of_cloud_scaled/region_1/1.ply"
#pcd=open3d.io.read_point_cloud(s)
#pcd=open3d.io.read_point_cloud(s)
pcd=read_ply(path=s)
print("done")
print("start rotation and preprocessing")
print(type(pcd))
R = pcd.get_rotation_matrix_from_xyz((180, 0, 0))
pcd_rotated = pcd.rotate(R, (0, 0, 0))
points=np.asarray(pcd_rotated.points)# Generate random 3D point cloud

