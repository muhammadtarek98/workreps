import open3d as o3d
import numpy as np
# Load point cloud
print(o3d.core.cuda.is_available())

device = o3d.core.Device("CUDA:0")

pcd = o3d.io.read_point_cloud(
    filename="/home/cplus/projects/m.tarek_master/graval_detection_3D/point_cloud_segmentation/rec_scaled_ascii.ply",print_progress=True)

# pcd_gpu = o3d.t.geometry.PointCloud(o3d.cuda.pybind.t.geometry.PointCloud(np.asarray(pcd.points)))
#device = o3d.core.Device("CPU:0")
#dtype = o3d.core.float32

# Create an empty point cloud
# Use pcd.point to access the points' attributes
pcd = o3d.t.geometry.PointCloud(device)

R = pcd.get_rotation_matrix_from_xyz((180, 0, 0))
#o3d.t.geometry.get_rotation_matriix_xyz((180,0,0))
#pcd.extrude_normals((180,0,0))
# pcd_rotated =pcd_gpu.rotate(R, (0,0,0))
# pcd_rotated_cuda =o3d.cuda.pybind.geometry.PointCloud(pcd_rotated_cpu.points)


#down_sampled_pcd =pcd.voxel_down_sample(voxel_size=0.0001)
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.0001, max_nn=100000))
#down_sampled_pcd.estimate_normals(radius=0.0001, max_nn=100000)

with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as m:
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=12)
o3d.visualization.draw_geometries(geometry_list=[mesh])