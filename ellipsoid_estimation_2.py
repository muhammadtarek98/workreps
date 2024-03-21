import open3d
import os
import numpy as np
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
import cv2 as cv
import torch
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans as SKLearnKMeans
from torch_kmeans import KMeans as TorchKMeans
import torch.utils.dlpack
import pandas as pd

def fit_ellipsoid(points):
    alpha = 1
    mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(points, alpha)
    return mesh

def fit_ellipse_cv(poly):
    poly = np.array(poly, dtype=np.float32)
    try:
        ((cent_x, cent_y), (width, height), angle) = cv.fitEllipse(points=poly)
    except cv.error as e:
        print(f"Error in fitEllipse: {e}")
        return None
    r1, r2 = width / 2, height / 2
    axes = (r1, r2)
    angle = angle
    d = dict()
    d["angle"] = angle
    d["axes"] = axes
    d["center_coordinates"] = (cent_x, cent_y)
    d["r1"] = r1
    d["r2"] = r2
    return d

def calculate_volume_using_fitted_ellipse_cv(poly):
    dp=fit_ellipse_cv(poly)
    r1, r2 = dp['r1'], dp['r2']
    return (4 / 3.0) * np.pi * r1 * r2 * min(r1, r2)

def kmeans_clustering_ellipsoids(pcd):
    kmeans = SKLearnKMeans(n_clusters=1500, max_iter=2, algorithm="full", init="random", n_init="auto")
    labels = kmeans.fit_predict(np.asarray(pcd.points))
    clusters = []
    for cluster_label in range(kmeans.n_clusters):
        cluster_points = pcd.select_by_index(np.where(labels == cluster_label)[0])
        clusters.append(cluster_points)
    return clusters


def kmeans_clustering_torch(pcd, device='cuda'):
    pcd_np = np.asarray(pcd.points)
    # Subsample points to reduce the total number of points
    subsample_factor = 0.1  # Adjust this value as needed
    subsample_indices = np.random.choice(len(pcd_np), size=int(len(pcd_np) * subsample_factor), replace=False)
    subsampled_pcd_np = pcd_np[subsample_indices]
    pcd_torch = torch.from_numpy(subsampled_pcd_np).float().to(device)
    pcd_torch=pcd_torch.unsqueeze(dim=0)
    print(pcd_torch.shape)
    kmeans = TorchKMeans(n_clusters=15000, max_iter=2, device="cpu")
    labels = kmeans.fit(pcd_torch)
    clusters = []
    for cluster_label in range(kmeans.n_clusters):
        cluster_indices = subsample_indices[labels == cluster_label]
        cluster_points = pcd.select_by_index(cluster_indices)
        clusters.append(cluster_points)
    
    return clusters
def fit_mesh(points,r3):
    alpha = 1
    mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(points, alpha)
    mesh.scale
    return mesh
def fit_ball(points, radii):
    pcd = open3d.geometry.PointCloud()
    radii_list = radii.tolist()
    pcd.points = open3d.utility.Vector3dVector(points)
    pcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=2, max_nn=len(points)))
    radii_double_vector = open3d.utility.DoubleVector(radii_list)
    ball = open3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii_double_vector)
    return ball

def fit_ellipsoid(points, axes,center):
    sphere = open3d.geometry.TriangleMesh.create_sphere(radius=axes[0])
    scale_factor = np.array([axes[0], axes[1], axes[2]])
    sphere.scale(scale_factor[0]*scale_factor[1]*scale_factor[2], center=center)  # Adjust the position based on the cluster center
    return sphere

def calculate_ellipsoid_volume(mesh):
    volume = mesh.get_surface_area()
    return volume

def calculate_convex_hull_volume(pcd):
    convex_hull, _ = pcd.compute_convex_hull()
    volume = convex_hull.get_oriented_bounding_box().volume()
    return volume
if __name__ == '__main__':
        print("read_file")
        result=dict()
        result["file_name"]=[]
        result["total_volume"]=[]
        device = open3d.core.Device("CUDA:0")
        main_dir="/home/cplus/projects/m.tarek_master/graval_detection_3D/regions_of_cloud_scaled"
        regions=["region_1","region_2","region_3","region_4","region_5","region_6"]
        for i in regions:
            for j in os.listdir(os.path.join(main_dir,i)):
                pcd = open3d.io.read_point_cloud(os.path.join(main_dir,i,j))
                R = pcd.get_rotation_matrix_from_xyz((180, 0, 0))
                pcd_rotated = pcd.rotate(R, (0, 0, 0))
                alpha = 0.3
            
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")  
                ellipsoid_clusters = kmeans_clustering_ellipsoids(pcd_rotated)
                total_ellipsoid_volumes = 0.0

                for idx, ellipsoid_cluster in enumerate(ellipsoid_clusters):
                    points = np.asarray(ellipsoid_cluster.points)
                    print("PCA in progress")
                    pca = PCA(n_components=2)
                    points_2d = pca.fit_transform(points)
                    ellipsoid_volume = calculate_volume_using_fitted_ellipse_cv(points_2d)*pow(100,3)
                    center=ellipsoid_cluster.get_center()
                    print("radius calculation in progress")
                    dp=fit_ellipse_cv(points_2d)
                    r1,r2,r3=dp['r1'],dp['r2'],min(dp['r1'],dp['r2'])
                    print(r1," ",r2," ",r3)
                    ellipsoid = fit_ball(points=ellipsoid_cluster.points,radii=np.asarray([r1,r2,r3]))
                    print(f"Ellipsoid {idx + 1} results: {ellipsoid_volume:.6f} cm^3")
                    open3d.io.write_point_cloud(filename=f"/home/cplus/projects/m.tarek_master/graval_detection_3D/point_cloud_segmentation/clusters/stone{idx+1}.ply", 
                                                write_ascii=True, 
                                                pointcloud=ellipsoid_cluster)
                    open3d.io.write_triangle_mesh(filename=f"/home/cplus/projects/m.tarek_master/graval_detection_3D/point_cloud_segmentation/meshes/stone{idx+1}.ply",
                                                write_ascii=True, 
                                                mesh=ellipsoid)
                result["file_name"].append(os.path.join(main_dir,i))
                result["total_volume"].append(ellipsoid_volume)
            new = pd.DataFrame.from_dict(result)
            new.to_csv(f"results_{j}.csv",encoding='utf-8', index=False)
            print("Total Ellipsoid Volumes:", total_ellipsoid_volumes)  




            