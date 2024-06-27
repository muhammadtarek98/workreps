import os

import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
import Parameters, tools
from detrend import rotate_point_cloud_plane, orient_normals
from cluster import clean_labels, cluster
from segment import segment_labels
from visualization import show_clouds

import ellipsoid

# Inputs
dir_ = "/home/cplus/projects/m.tarek_master/graval_detection_3D/point_cloud_segmentation/plys"
cloud = os.path.join(dir_, "cropped_circle_old_ascii.ply")
cloud_detrended = os.path.join(dir_, "cropped_circle_old_ascii.ply")
ini = "/home/cplus/projects/m.tarek_master/graval_detection_3D/point_cloud_segmentation/g3point_python/params.ini"

# Load data
xyz = tools.load_data(cloud)
# Remove min values
mins = np.amin(xyz, axis=0)  # WARNING: done in the Matlab code
xyz = xyz - mins

params = Parameters.Parameters(ini)

# Rotate and detrend the point cloud
if params.rot_detrend:
    # make the normal of the fitted plane aligned with the axis
    axis = np.array([0, 0, 1])
    xyz_rot = rotate_point_cloud_plane(xyz, axis)
    x, y, z = np.split(xyz_rot, 3, axis=1)
    # Remove the polynomial trend from the cloud
    least_squares_solution, residuals, rank, s = np.linalg.lstsq(np.c_[np.ones(x.shape), x, y, x ** 2, x * y, y ** 2],
                                                                 z, rcond=None)
    a0, a1, a2, a3, a4, a5 = least_squares_solution
    z_detrended = z - (a0 + a1 * x + a2 * y + a3 * x ** 2 + a4 * x * y + a5 * y ** 2)
    xyz_detrended = np.c_[x, y, z_detrended]
else:
    xyz_detrended = xyz

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)

# Find neighbors of point cloud
tree = KDTree(xyz)  # build a KD tree
neighbors_distances, neighbors_indexes = tree.query(xyz, params.knn + 1)
neighbors_distances, neighbors_indexes = neighbors_distances[:, 1:], neighbors_indexes[:, 1:]

# Determine node surface
surface = np.pi * np.amin(neighbors_distances, axis=1) ** 2

# Compute normals and force them to point towards positive Z
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(params.knn))
centroid = np.mean(xyz, axis=0)
sensor_center = np.array([centroid[0], centroid[1], 1000])
normals = orient_normals(xyz, np.asarray(pcd.normals), sensor_center)

labels, nlabels, labelsnpoint, stacks, ndon, sink_indexes = segment_labels(xyz_detrended, params.knn, neighbors_indexes)


center, radii, quaternions, rotation_matrix, ellipsoid_parameters = (
    ellipsoid.fit_ellipsoid_to_grain(xyz[stacks[1]]))

c, r, q, r_m = ellipsoid.implicit_to_explicit(ellipsoid_parameters)

matlab_rotation_matrix = np.array([[-0.9264, -0.3668, 0.0854],
                                   [-0.2170, 0.3345, -0.9171],w
                                   [0.3078, -0.8681, -0.3894]])

tools.save_data_with_colors(cloud, xyz, mins, stacks, labels, '_G3POINT')
tools.save_data_with_colors(cloud, xyz[sink_indexes, :], mins, stacks, np.arange(len(stacks)), '_G3POINT_SINKS')
