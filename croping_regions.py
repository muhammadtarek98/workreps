"""import open3d as o3d

# Load the PLY file
pcd = o3d.io.read_point_cloud("/home/cplus/projects/m.tarek_master/cloud_Scaled_ascii_label_0.ply")

# Define the size of the small regions (e.g., 100 points per region)
region_size = 1000

# Iterate over the points and segment them into small regions
num_points = len(pcd.points)
num_regions = num_points // region_size

for i in range(num_regions):
    # Define the range of points for the current region
    start_idx = i * region_size
    end_idx = min((i + 1) * region_size, num_points)
    
    # Extract the points and colors for the current region
    region_points = pcd.points[start_idx:end_idx]
    region_colors = pcd.colors[start_idx:end_idx]  # Assuming colors are available
    
    # Create a new PointCloud object for the current region
    region_pcd = o3d.geometry.PointCloud()
    region_pcd.points = o3d.utility.Vector3dVector(region_points)
    region_pcd.colors = o3d.utility.Vector3dVector(region_colors)
    
    # Save the region as a PLY file
    o3d.visualization.draw_geometries(geometry_list=[region_pcd],window_name=f'cluster {i+ 1}')
"""
import os 
for i in os.listdir("/home/cplus/projects/m.tarek_master/"):
    if "region_" in i:
        os.remove(i)