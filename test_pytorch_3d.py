import torch
import torch.nn.functional as F
import pytorch3d
from pytorch3d.ops import sample_points_from_meshes
import open3d as o3d
# Assuming you have a point cloud as a numpy array (Nx3)
point_cloud_np = o3d.io.read_point_cloud(r"/home/cplus/projects/m.tarek_master/graval_detection_3D/kmean_results.ply")

# Convert the point cloud to a PyTorch tensor
point_cloud_tensor = torch.tensor(point_cloud_np, dtype=torch.float32)

# Create a Pointclouds object
point_clouds = Pointclouds(points=[point_cloud_tensor])

# Sample points from the point cloud to create a mesh
verts, faces = sample_points_from_meshes(point_clouds, num_samples=10000)

# Create a Meshes object
meshes = pytorch3d.structures.Meshes(verts=[verts], faces=[faces])

# Visualize the mesh (optional)
from pytorch3d.vis.plotly_vis import plot_scene
plot_scene({
    "pointcloud": point_clouds,
    "mesh": meshes
})
