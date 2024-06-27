import open3d as o3d
import numpy as np
import os
import torch
from matplotlib import pyplot as plt
print(o3d.core.cuda.is_available())

# print(help(pcd))
# pcd.compute_point_cloud_distance()
# convex_hull=pcd.compute_convex_hull()
# avg_distance=np.mean(distances)
# print(avg_distance)
# boundarys, mask = pcd.compute_boundary_points(avg_distance,100)
# boundarys = boundarys.paint_uniform_color([0.0, 0.0, 1.0])
# pcd = pcd.paint_uniform_color([0.6, 0.6, 0.6])
# o3dcuda.visualization.draw_geometries(geometry_list=[pcd,boundarys], window_name='Clustered Point Cloud')
# dists = o3d.geometry.PointCloud.compute_point_cloud_distance()
# stone=np.asarray([pcd==[1.0,1.0,1.0]])
# print(pcd_np)
# kmeans=KMeans(n_clusters=5,max_iter=1000,algorithm="lloyd")
# labels=kmeans.fit_predict(np.asarray(pcd.points))
# num_points_in_cluster = np.bincount(labels)
# print(num_points_in_cluster)
# o3d.visualization.draw_geometries(geometry_list=[pcd])
