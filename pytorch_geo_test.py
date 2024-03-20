import torch_geometric
import open3d as o3d
from torch_geometric.nn import aggr
#print(torch_geometric.__version__)
mesh=torch_geometric.io.read_obj(in_file="/home/cplus/projects/m.tarek_master/Ellipsoid_1.obj")
print(mesh)
o3d.visualization()