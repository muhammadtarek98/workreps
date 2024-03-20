import open3d as o3d
import torch
import networkx as nx
from torch_geometric.data import Data
import torch_geometric
import numpy as np
import cugraph
from matplotlib import pyplot as plt

print("read file")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the point cloud from a PLY file
point_cloud = o3d.io.read_point_cloud("/home/cplus/projects/m.tarek_master/graval_detection_3D/point_cloud_segmentation/plys/label_0.ply")
R = point_cloud.get_rotation_matrix_from_xyz((180, 0, 0))
pcd_rotated = point_cloud.rotate(R, (0, 0, 0))
points = np.asarray(pcd_rotated.points)

# Convert the point cloud to a PyG Data object
data = Data(pos=torch.tensor(points, dtype=torch.float).to(device))

# Create the k-nearest neighbor graph
knn_graph = torch_geometric.transforms.KNNGraph(k=8, force_undirected=True, cosine=True)
graph = knn_graph(data)

# Convert PyG graph to cugraph
edge_index = graph.edge_index.cpu().numpy()
num_nodes = graph.num_nodes
edge_index = (edge_index[0].tolist(), edge_index[1].tolist())  # Convert NumPy arrays to Python lists
g = cugraph.Graph()
g.from_cudf_edgelist(edge_index[0], edge_index[1])

# Add nodes to the graph
g.add_nodes(num_nodes)

# Get node positions from the PyG Data object
node_positions = {i: data.pos[i].cpu().numpy() for i in range(len(data.pos))}

# Visualize the graph in 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Draw nodes
for node, pos in node_positions.items():
    ax.scatter(pos[0], pos[1], pos[2], color='blue')

# Draw edges
for edge in g.edges():
    start_node = node_positions[edge[0]]
    end_node = node_positions[edge[1]]
    ax.plot([start_node[0], end_node[0]], [start_node[1], end_node[1]], [start_node[2], end_node[2]], color='red')

plt.show()
