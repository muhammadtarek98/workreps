import open3d as o3d
import os
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from sklearn.decomposition import PCA
import cv2 as cv
from matplotlib import pyplot as plt
import torch 
from scipy.spatial import KDTree
from g3point_python.segment import segment_labels
from g3point_python.detrend import rotate_point_cloud_plane,orient_normals
import cuml
from sklearn.cluster import KMeans as SKLearnKMeans
from sklearn.decomposition import PCA
import torch.utils.dlpack
import pandas as pd
import microstructpy
from microstructpy.geometry import Ellipsoid
from torch_kmeans import KMeans as TorchKMeans
def save_pcds(stacks,segmented_pcd,stones_dir,segmented_file,pcd,pcd_sinks):
    for idx,stack in enumerate(stacks):
        stone=pcd.select_by_index(stack)
        print(stone)
        stone_pcd=o3d.geometry.PointCloud()
        stone_pcd.points=stone.points
        stone_pcd.colors=stone.colors
        stone_pcd.normals=stone.normals
        stone_dir =f"{stones_dir}/{idx+1}.ply"
        o3d.io.write_point_cloud(filename=stone_dir,
                                pointcloud=stone,
                                write_ascii=True,
                                print_progress=True )
        print(f"stone {idx+1} is save")
    o3d.visualization.draw_geometries([pcd,pcd_sinks])
    o3d.io.write_point_cloud(filename=segmented_file, 
                         pointcloud=segmented_pcd,
                             write_ascii=True,
                               compressed=False,
                             print_progress=False)

def preprocess(file_name:str):
    #print("loading file.")
    pcd = o3d.io.read_point_cloud(file_name)
    #print("loaded file ..")
    R = pcd.get_rotation_matrix_from_xyz((180,0, 0))
    pcd_rotated = pcd.rotate(R, (0, 0, 0))
    #print("file preprocessed")
    return pcd_rotated


def project_2d(points:np.ndarray):
    pca= PCA(n_components=2)    
    project_2d_points=pca.fit_transform(points)
    return project_2d_points

def project_2d_GPU(points:np.ndarray):
    pca=cuml.PCA(n_components=2)
    projected_2d_points=pca.fit_transform(points)
    return projected_2d_points
def sand_fliter(pcd,file_name=None):
    #print("remove sand begins")
    kmeans=cuml.KMeans(n_clusters=2,max_iter=1000000000)
    colors=np.asarray(pcd.colors)
    labels=kmeans.fit_predict(colors)
    pcd_label_1=pcd.select_by_index(np.where(labels==1)[0])
    pcd_label_0 = pcd.select_by_index(np.where(labels == 0)[0])
    o3d.visualization.draw_geometries([pcd_label_1])
    o3d.io.write_point_cloud(filename=f"/home/cplus/projects/m.tarek_master/graval_detection_3D/point_cloud_segmentation/cloud_scaled_full_ascii/{file_name}/label_1.ply",
                          pointcloud=pcd_label_1,
                          write_ascii=True,
                          print_progress=True)
    o3d.visualization.draw_geometries([pcd_label_0])
    o3d.io.write_point_cloud(filename=f"/home/cplus/projects/m.tarek_master/graval_detection_3D/point_cloud_segmentation/cloud_scaled_full_ascii/{file_name}/label_0.ply",
                          pointcloud=pcd_label_0,
                          write_ascii=True,
                          print_progress=True)
    print("files saved")


def sand_fliter(pcd,file_name:str=None):
    #print("remove sand begins")
    kmeans=cuml.KMeans(n_clusters=2,max_iter=1000000000)
    colors=np.asarray(pcd.colors)
    labels=kmeans.fit_predict(colors)
    pcd_label_1=pcd.select_by_index(np.where(labels==1)[0])
    pcd_label_0 = pcd.select_by_index(np.where(labels == 0)[0])
    o3d.visualization.draw_geometries([pcd_label_1])
    o3d.io.write_point_cloud(filename=f"/home/cplus/projects/m.tarek_master/graval_detection_3D/point_cloud_segmentation/cloud_scaled_full_ascii/{file_name}/label_1.ply",
                          pointcloud=pcd_label_1,
                          write_ascii=True,
                          print_progress=True)
    o3d.visualization.draw_geometries([pcd_label_0])
    o3d.io.write_point_cloud(filename=f"/home/cplus/projects/m.tarek_master/graval_detection_3D/point_cloud_segmentation/cloud_scaled_full_ascii/{file_name}/label_0.ply",
                          pointcloud=pcd_label_0,
                          write_ascii=True,
                          print_progress=True)
    print("files saved")

def find_nearest_neighbor(u, points):
    min_distance = float('inf')
    nearest_vertical_neighbor = None
    for v in range(len(points)):
        if u != v:
            distance = torch.linalg.norm(points[u] - points[v])
            if distance < min_distance:
                min_distance = distance
                nearest_vertical_neighbor = v
    return (u, nearest_vertical_neighbor, min_distance)



def fit_ball(points, radii):
    pcd = o3d.geometry.PointCloud()
    radii_list = radii.tolist()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2, max_nn=len(points)))
    radii_double_vector = o3d.utility.DoubleVector(radii_list)
    ball = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii_double_vector)
    return ball



def fit_ellipse_cv(poly)->dict:
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

def fit_ellipsoid_from_sphere(points, axes,center):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=axes[0])
    scale_factor = np.array([axes[0], axes[1], axes[2]])
    sphere.scale(scale_factor[0]*scale_factor[1]*scale_factor[2], center=center)  # Adjust the position based on the cluster center
    return sphere



def fit_ellipsoid(points):
    alpha = 1
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(points, alpha)
    return mesh



def calculate_volume_using_fitted_ellipse_cv(poly):
    dp=fit_ellipse_cv(poly)
    r1, r2 = dp['r1'], dp['r2']
    return (4 / 3.0) * np.pi * r1 * r2 * min(r1, r2)
 


def calculate_volume_using_fitted_ellipse(poly):
    dp=fit_ellipse_cv(poly)
    r1, r2 = dp['r1'], dp['r2']
    return (4 / 3.0) * np.pi * r1 * r2 * min(r1, r2)



def kmeans_clustering_ellipsoids(pcd,num_cluster):
    kmeans = KMeans(n_clusters=num_cluster, max_iter=1000000000, algorithm="lloyd",init="random",n_init="auto")
    labels = kmeans.fit_predict(np.asarray(pcd.points))
    clusters = []
    for cluster_label in range(kmeans.n_clusters):
        cluster_indices = np.where(labels == cluster_label)[0]
        cluster_points = np.asarray(pcd.points)[cluster_indices]
        clusters.append(cluster_points)
    return clusters


def calculate_ellipsoid_volume(mesh):
    volume = mesh.get_surface_area()
    return volume

def calculate_convex_hull_volume(pcd):
    convex_hull, _ = pcd.compute_convex_hull()
    volume = convex_hull.get_oriented_bounding_box().volume()
    return volume

def type_and_len(x):
    if type(x)==int:
        print(type(x))
    else:
        print(type(x))
        print(len(x))
        
        
def g3_point_cloud_cluster(point_cloud:np.ndarray,KNN:int)->dict:
    axes=np.array([0,0,1])
    xyz_detrended=rotate_point_cloud_plane(xyz=point_cloud,
                                           axis=axes)
    pcd_detrended=o3d.geometry.PointCloud()
    pcd_detrended.points=o3d.utility.Vector3dVector(xyz_detrended)
    tree=KDTree(xyz_detrended)
    neighbors_distance,neighbors_indexes=tree.query(k=KNN+1,
                                                    x=xyz_detrended)
    #exculd the the top node from the stack
    neighbors_distance=neighbors_distance[:,1:]
    neighbors_indexes=neighbors_indexes[:,1:]
    surface=np.pi * np.min(a=neighbors_distance,axis=1)**2
    pcd_detrended.estimate_normals(search_param=
                                   o3d.geometry.KDTreeSearchParamKNN(KNN))
    centroid=np.mean(a=xyz_detrended,axis=0)
    sensor_center=np.array([centroid[0],centroid[1],1000])
    normals=orient_normals(points=xyz_detrended,
                           normals=np.asarray(pcd_detrended.normals),
                           sensor_center=sensor_center)
    dp=segment_labels(xyz=xyz_detrended,
                   knn=KNN,
                   neighbors_indexes=neighbors_indexes,
                   braun_willett=False)
    print(type(dp))

    labels=dp["labels"]
    nlabels=dp["nlabels"]
    labelsnpoint=dp["labelsnpoint"]
    stacks=dp["stacks"]
    ndon=dp["ndon"]
    sink_indexes=dp["local_maximum_indexes"]
    return len(stacks)


def kmeans_clustering_ellipsoids(pcd):
    kmeans = SKLearnKMeans(n_clusters=1500, max_iter=2, algorithm="full", init="random", n_init="auto")
    labels = kmeans.fit_predict(np.asarray(pcd.points))
    clusters = []
    for cluster_label in range(kmeans.n_clusters):
        cluster_points = pcd.select_by_index(np.where(labels == cluster_label)[0])
        clusters.append(cluster_points)
    return clusters

def calculate_convex_hull_volume(pcd):
    convex_hull, _ = pcd.compute_convex_hull()
    volume = convex_hull.get_oriented_bounding_box().volume()
    return volume


def distance_function(params, cloud_points):
    x_c, y_c, z_c, a, b, c = params
    distances = np.sqrt(((cloud_points[:, 0] - x_c) / a)**2 + ((cloud_points[:, 1] - y_c) / b)**2 + ((cloud_points[:, 2] - z_c) / c)**2) - 1
    return np.sum(distances**2)


def kmean_cluster_GPU(pcd,number_of_clusters):
    kmeans=cuml.KMeans(n_clusters=number_of_clusters,max_iter=1000000000)
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
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(points, alpha)
    mesh.scale
    return mesh