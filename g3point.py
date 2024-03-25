import configparser
import os

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from scipy.spatial import KDTree

import Parameters, tools
from detrend import rotate_point_cloud_plane, orient_normals
from segment import segment_labels
from visualization import show_clouds

dir= "/home/cplus/projects/m.tarek_master/graval_detection_3D/point_cloud_segmentation"
cloud = os.path.join(dir, "1_copy_ascii.ply")    
cloud_detrended = os.path.join(dir, "1_copy_ascii.ply")
ini = "/home/cplus/projects/m.tarek_master/graval_detection_3D/point_cloud_segmentation/g3point_python/params.ini"

pcd_orig = o3d.io.read_point_cloud(cloud)
#R = pcd_orig.get_rotation_matrix_from_xyz((180,0, 0))
#pcd_orig = pcd_orig.rotate(R, (0, 0, 0))
#pcd_orig=pcd_orig.paint_uniform_color([0,0, 1])
xyz = np.asarray(pcd_orig.points)
params = Parameters.Parameters(ini)
axis = np.array([0, 0, 1])
#o3d.visualization.draw_geometries([pcd_orig])
xyz_detrended = rotate_point_cloud_plane(xyz, axis)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz_detrended)
print(len(np.asarray(pcd.points)))
tree = KDTree(xyz_detrended)  # build a KD tree
neighbors_distances, neighbors_indexes = tree.query(xyz_detrended, params.knn + 1)
neighbors_distances, neighbors_indexes = neighbors_distances[:, 1:], neighbors_indexes[:, 1:]

surface = np.pi * np.amin(neighbors_distances, axis=1) ** 2

pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(params.knn))
centroid = np.mean(xyz_detrended, axis=0)
sensor_center = np.array([centroid[0], centroid[1], 1000])
normals = orient_normals(xyz_detrended, np.asarray(pcd.normals), sensor_center)

labels, nlabels, labelsnpoint, stacks, ndon, sink_indexes = segment_labels(xyz_detrended, params.knn, neighbors_indexes,braun_willett=True)

colors = np.random.rand(len(stacks), 3)[labels, :]
pcd.colors = o3d.utility.Vector3dVector(colors)
pcd_sinks = o3d.geometry.PointCloud()
pcd_sinks.points = o3d.utility.Vector3dVector(xyz_detrended[sink_indexes, :])
pcd_sinks.paint_uniform_color(np.array([1., 0., 0.]))
clouds = (('pcd', pcd, None, 3),('pcd_sinks', pcd_sinks, None, 5))
show_clouds(clouds)
print(stacks)
#o3d.visualization.draw_geometries([labels])
print(nlabels," ",len(labelsnpoint)," ",len(labels)," ",len(ndon))