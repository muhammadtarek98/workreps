import open3d as o3d
file=2
pcd=o3d.io.read_point_cloud(f"/media/cplus/ITHDD/Cloud_Gravel_Scaled/half/{file}/{file}_ascii.ply")
o3d.visualization.draw_geometries([pcd])
