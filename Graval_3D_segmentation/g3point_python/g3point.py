import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from scipy.spatial import KDTree
from detrend import rotate_point_cloud_plane, orient_normals
from segment import segment_labels
from visualization import show_clouds

file=2
file_name=f"/home/cplus/projects/m.tarek_master/graval_detection_3D/point_cloud_segmentation/cloud_scaled_half_ascii/{file}/{file}.ply"
pcd_orig = o3d.io.read_point_cloud(file_name)
R = pcd_orig.get_rotation_matrix_from_xyz((180,0, 0))
pcd_orig = pcd_orig.rotate(R, (0, 0, 0))
xyz = np.asarray(pcd_orig.points)
axis = np.array([0, 0, 1])
xyz_detrended = rotate_point_cloud_plane(xyz, axis)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz_detrended)
pcd.colors=o3d.utility.Vector3dVector(pcd_orig.colors)
print(len(np.asarray(pcd.points)))
K=100
tree = KDTree(xyz_detrended)  # build a KD tree
neighbors_distances, neighbors_indexes = tree.query(xyz_detrended,
 K + 1)
neighbors_distances, neighbors_indexes = neighbors_distances[:, 1:], neighbors_indexes[:, 1:]
pcd.normals=pcd_orig.normals
centroid = np.mean(xyz_detrended, axis=0)
sensor_center = np.array([centroid[0], centroid[1], 1000])
normals = orient_normals(xyz_detrended, np.asarray(pcd.normals), sensor_center)

dp= segment_labels(xyz_detrended, K, neighbors_indexes,braun_willett=False)

labels=dp["labels"]
nlabels=dp["nlabels"]
labelsnpoint=dp["labelsnpoint"]
stacks=dp["stacks"]
ndon=dp["ndon"]
local_maximum_indexes=dp["local_maximum_indexes"]

colors = np.random.rand(len(stacks), 3)[labels, :]
pcd.colors = o3d.utility.Vector3dVector(colors)
pcd_sinks = o3d.geometry.PointCloud()
pcd_sinks.points = o3d.utility.Vector3dVector(xyz_detrended[local_maximum_indexes, :])
pcd_sinks.paint_uniform_color(np.array([1., 0., 0.]))
clouds = (('pcd', pcd, None, 3),('pcd_sinks', pcd_sinks, None, 5))
#show_clouds(clouds)
print(nlabels," ",len(labelsnpoint)," ",len(labels)," ",len(ndon)," ",len(stacks))
print(len(np.asarray(pcd_sinks.points)))
stone_1=pcd.select_by_index(stacks[0])
print(type(nlabels))

o3d.visualization.draw_geometries([pcd,pcd_sinks])