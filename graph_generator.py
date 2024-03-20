import open3d as o3d
import torch
import networkx as nx
from torch_geometric.data import Data
import torch_geometric
import numpy as np

print("read file")
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
point_cloud = o3d.io.read_point_cloud("/home/cplus/projects/m.tarek_master/graval_detection_3D/point_cloud_segmentation/regions_of_cloud_scaled/region_1/1.ply")
R = point_cloud.get_rotation_matrix_from_xyz((180, 0, 0))
pcd_rotated = point_cloud.rotate(R, (0, 0, 0))
points=torch.from_numpy(np.asarray(pcd_rotated.points))

print(type(points))
point_cloud=points.to(device)
data = Data(pos=torch.tensor(points, dtype=torch.float).to(device))

knn_graph=torch_geometric.transforms.KNNGraph(k=2,force_undirected=False,loop=True,cosine=True)
ccp=torch_geometric.transforms.LargestConnectedComponents(num_components=4)

graph=knn_graph(data)
#graph=ccp(graph)
print(graph)
g = torch_geometric.utils.to_cugraph(graph.edge_index,directed=False,relabel_nodes= True)
node_positions = {i:
                   pcd_rotated.points[i] 
                   for i in range(len(pcd_rotated.points))}

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd_rotated.points))

edges = graph.edge_index.cpu().numpy().T
G = nx.Graph()
G.add_edges_from(edges)

# Draw point cloud
o3d.visualization.draw_geometries([pcd], window_name='Point Cloud')

# Draw graph edges
lines = []
for edge in G.edges():
    start_node = node_positions[edge[0]]
    end_node = node_positions[edge[1]]
    lines.append([start_node, end_node])

line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(np.vstack(lines))
line_set.lines = o3d.utility.Vector2iVector(np.array([[i, i+1] for i in range(0, len(lines), 2)]))
o3d.visualization.draw_geometries([line_set], window_name='Graph Edges')
