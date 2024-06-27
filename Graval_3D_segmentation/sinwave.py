import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
import open3d 

print("read pcd")
s="/home/cplus/projects/m.tarek_master/graval_detection_3D/point_cloud_segmentation/regions_of_cloud_scaled/region_1/1.ply"
pcd=open3d.io.read_point_cloud(s)
print("done")
print("start rotation and preprocessing")

R = pcd.get_rotation_matrix_from_xyz((180, 0, 0))
pcd_rotated = pcd.rotate(R, (0, 0, 0))
points=np.asarray(pcd_rotated.points)# Generate random 3D point cloud
num_points = 100

# Construct KDTree for efficient neighbor search
kdtree = KDTree(points)

# Define distance threshold for connecting points
distance_threshold = 0.1

# Construct graph
G = nx.Graph()

# Connect neighboring points within distance threshold
for i, point in enumerate(points):
    # Find neighbors within distance threshold
    neighbors = kdtree.query_radius([point], r=distance_threshold)[0]
    # Add edges between the current point and its neighbors
    for neighbor_idx in neighbors:
        if neighbor_idx != i:  # Exclude self-loops
            G.add_edge(i, neighbor_idx)

# Visualize graph
nx.draw(G, with_labels=True)
plt.show()
