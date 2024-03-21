import open3d as o3d
import numpy as np
# Load point cloud
print(o3d.core.cuda.is_available())

device = o3d.core.Device("CUDA:0")

pcd = o3d.io.read_point_cloud(
    filename="/home/cplus/projects/m.tarek_master/graval_detection_3D/point_cloud_segmentation/plys/rec_scaled_ascii.ply",print_progress=True)

# pcd_gpu = o3d.t.geometry.PointCloud(o3d.cuda.pybind.t.geometry.PointCloud(np.asarray(pcd.points)))
#device = o3d.core.Device("CPU:0")
#dtype = o3d.core.float32

# Create an empty point cloud
# Use pcd.point to access the points' attributes
#pcd = o3d.t.geometry.PointCloud(device)

R = pcd.get_rotation_matrix_from_xyz((180, 0, 0))
pcd_rotated = pcd.rotate(R, (0, 0, 0))

pcd_rotated.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.00001, max_nn=150))

with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_rotated, depth=26)
o3d.visualization.draw_geometries(geometry_list=[mesh])

"""
import open3d as o3d
import numpy as np
import torch
import torch_geometric.transforms as T
import torch_geometric.utils as torch_utils
import cugraph
import cudf
import torch_geometric
print("read file")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
point_cloud = o3d.io.read_point_cloud("/home/cplus/projects/m.tarek_master/graval_detection_3D/point_cloud_segmentation/regions_of_cloud_scaled/region_1/1.ply")
R = point_cloud.get_rotation_matrix_from_xyz((180, 0, 0))
pcd_rotated = point_cloud.rotate(R, (0, 0, 0))
points = np.asarray(pcd_rotated.points)
cl, ind = pcd_rotated.remove_statistical_outlier(nb_neighbors=10,
                                                    std_ratio=2.0)
#downpcd = pcd_rotated.voxel_down_sample(voxel_size=0.001)
#print(cl)
#print(ind)
inlier_cloud = pcd_rotated.select_by_index(ind)
outlier_cloud = pcd_rotated.select_by_index(ind, invert=True)
o3d.visualization.draw_geometries([inlier_cloud])
print(len(pcd_rotated.points))

"""