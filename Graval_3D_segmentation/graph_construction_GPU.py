import open3d as o3d
import torch
import numpy as np
import cugraph
import torch.multiprocessing as mp
from clustering_and_fitting import *
if __name__ == '__main__':
    print("read file")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    point_cloud = o3d.io.read_point_cloud("/home/cplus/projects/m.tarek_master/graval_detection_3D/point_cloud_segmentation/plys/label_0.ply")
    R = point_cloud.get_rotation_matrix_from_xyz((180, 0, 0))
    pcd_rotated = point_cloud.rotate(R, (0, 0, 0))
    points = np.asarray(pcd_rotated.points)
    points = torch.tensor(points, device=device, dtype=torch.float32)
    num_processes = mp.cpu_count()
    mp.set_start_method('spawn') 
    pool = mp.Pool(num_processes)
    results = [pool.apply_async(find_nearest_neighbor, args=(u, points)) for u in range(len(points))]
    pool.close()
    pool.join()
    edges = []
    weights = []
    for result in results:
        u, nearest_vertical_neighbor, min_distance = result.get()
        if nearest_vertical_neighbor is not None:
            edges.append((u, nearest_vertical_neighbor))
            weights.append(min_distance)
    edges_np = np.array(edges)
    weights_np = np.array(weights)
    g = cugraph.Graph()
    g.add_edge_list(edges_np[:, 0], edges_np[:, 1], weights_np)
    print(g)

"""
points = np.asarray(pcd_rotated.points)
data = Data(pos=torch.tensor(points, dtype=torch.float).to(device))
knn_graph = torch_geometric.transforms.KNNGraph(k=2, force_undirected=True, cosine=True)
graph = knn_graph(data)
print(graph)
g = torch_geometric.utils.to_cugraph(graph.edge_index,directed=False,relabel_nodes= True)
#print(help(g))
print(g.is_bipartite())
print(g.has_isolated_vertices())

node_positions = {i: pcd_rotated.points[i]for i in range(len(pcd_rotated.points))}

pcd=o3d.geometry.PointCloud()
pcd.points=o3d.utility.Vector3dVector(points)
lines=[]
for edge in g.edges():
    print(edge)
    break
    u=node_positions[edge[0]]
    v=node_positions[edge[1]]
    lines.append([u,v])
line_set=o3d.geometry.LineSet()
line_set.points=pcd_rotated.points
line_set.lines=o3d.utility.Vector2iVector(np.array([[i, i+1] for i in range(0, len(lines), 2)]))
o3d.visualization.draw_geometries([line_set])
"""