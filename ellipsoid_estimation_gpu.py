import cuml
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
import microstructpy
from matplotlib import pyplot as plt
from microstructpy.geometry import Ellipsoid
import math
def fit_ball(points, radii):
    pcd = open3d.geometry.PointCloud()
    radii_list = radii.tolist()
    pcd.points = open3d.utility.Vector3dVector(points)
    pcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=2, max_nn=len(points)))
    radii_double_vector = open3d.utility.DoubleVector(radii_list)
    ball = open3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii_double_vector)
    return ball
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

def fit_ellipsoid(points):
    alpha = 1
    mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(points, alpha)
    return mesh

 
def calculate_volume_using_fitted_ellipse(poly):
    dp=fit_ellipse_cv(poly)
    r1, r2 = dp['r1'], dp['r2']
    return (4 / 3.0) * np.pi * r1 * r2 * min(r1, r2)




def calculate_ellipsoid_volume(mesh):
    volume = mesh.get_surface_area()
    return volume

def calculate_convex_hull_volume(pcd):
    convex_hull, _ = pcd.compute_convex_hull()
    volume = convex_hull.get_oriented_bounding_box().volume()
    return volume
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

def distance_function(params, cloud_points):
    x_c, y_c, z_c, a, b, c = params
    distances = np.sqrt(((cloud_points[:, 0] - x_c) / a)**2 + ((cloud_points[:, 1] - y_c) / b)**2 + ((cloud_points[:, 2] - z_c) / c)**2) - 1
    return np.sum(distances**2)
def kmean_cluster_GPU(pcd,number_of_clusters):
    kmeans=cuml.KMeans(n_clusters=15000,max_iter=1000000000)
    labels = kmeans.fit_predict(np.asarray(pcd.points))
    clusters = []
    for cluster_label in range(kmeans.n_clusters):
        cluster_points = pcd.select_by_index(np.where(labels == cluster_label)[0])
        clusters.append(cluster_points)
    return clusters


if __name__ == '__main__':
        result=dict()
        result["file_name"]=[]
        result["total_volume"]=[]
        device = open3d.core.Device("CUDA:0")
        alpha = 0.3    
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        main_dir="/home/cplus/projects/m.tarek_master/graval_detection_3D/point_cloud_segmentation/regions_of_cloud_scaled"
        regions=["region_1","region_2","region_3","region_4","region_5","region_6"]
        ellipsoids=[]
        pcs=[]
        x=1
        for i in regions:
            for j in sorted(os.listdir(os.path.join(main_dir,i))):
                pcd=open3d.io.read_point_cloud(os.path.join(main_dir,i,j))
                R = pcd.get_rotation_matrix_from_xyz((180, 0, 0))
                pcd_rotated = pcd.rotate(R, (0, 0, 0))
                points=np.asarray(pcd.points)
                pcd_sel = pcd.select_by_index(np.where(points[:, 2] > 0)[0])
                print(pcd_sel)
                #print(help(pcd_sel))
                print(f"read_file {os.path.join(main_dir,i,j)}")
                ellipsoid_clusters = kmean_cluster_GPU(pcd,number_of_clusters=len(points))
                total_ellipsoid_volumes = 0.0
                alpha = 0.3
                for idx, ellipsoid_cluster in enumerate(ellipsoid_clusters):
                    print(type(ellipsoid_cluster))
                    print("PCA in progress")
                    clusters_points = np.asarray(ellipsoid_cluster.points)
                    pca = cuml.PCA(n_components=2)
                    points_2d = pca.fit_transform(points)
                    ellipsoid_volume = calculate_volume_using_fitted_ellipse_cv(points_2d)*pow(100,3)
                    center=ellipsoid_cluster.get_center()
                    print("radius calculation in progress")
                    dp=fit_ellipse_cv(points_2d)
                    r1,r2,r3=dp['r1'],dp['r2'],min(dp['r1'],dp['r2'])
                    print(r1," ",r2," ",r3)
                    angle=dp["angle"]
                    print(f"Ellipsoid {idx + 1} results: {ellipsoid_volume:.6f} cm^3")
                    total_ellipsoid_volumes+=ellipsoid_volume
                    a = np.abs(r1)
                    b = np.abs(r2)
                    c = np.abs(r3) 
                    estimated_ellipsoid = Ellipsoid(a=a,b=b,c=c)
                    #print(type(estimated_ellipsoid))
                    results=estimated_ellipsoid.best_fit(points)
                    print(results)
                    theta = np.linspace(0, angle, 1000)
                    phi = np.linspace(0,angle, 1000)
                    theta, phi = np.meshgrid(theta, phi)
                    x = a * np.sin(phi) * np.cos(theta)
                    y = b * np.sin(phi) * np.sin(theta)
                    z = c * np.cos(phi)
                    vertices = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=-1)
                    mesh = open3d.geometry.TriangleMesh()
                    mesh.vertices = open3d.utility.Vector3dVector(vertices)
                    num_vertices_per_side = len(x)
                    faces = []
                    for k in range(num_vertices_per_side - 1):
                        for l in range(num_vertices_per_side - 1):
                            idx1 = k * num_vertices_per_side + l
                            idx2 = idx1 + 1
                            idx3 = (k + 1) * num_vertices_per_side + l
                            idx4 = idx3 + 1
                            faces.append([idx1, idx2, idx3])
                            faces.append([idx2, idx4, idx3])
                    mesh.triangles = open3d.utility.Vector3iVector(faces)
                    #save pc
                    ellipsoids.append(mesh)
                    pcs.append(ellipsoid_cluster)
                    result["file_name"].append(os.path.join(main_dir,i))
                    result["total_volume"].append(ellipsoid_volume)
            new = pd.DataFrame.from_dict(result)
            new.to_csv(f"results_{x}.csv",encoding='utf-8', index=False)
            print(f"Total Ellipsoid Volumes:{total_ellipsoid_volumes} cm^3 ")
            x+=1
        for i in range(len(ellipsoids)):
         open3d.io.write_triangle_mesh(f"/home/cplus/projects/m.tarek_master/graval_detection_3D/point_cloud_segmentation/regions_of_cloud_scaled/1/ellipsoids/{i}.ply",
                                      mesh=mesh[i], 
                                      write_ascii=True, 
                                      compressed=False, 
                                      write_vertex_normals=True, 
                                      write_vertex_colors=True,
                                        write_triangle_uvs=True, 
                                        print_progress=True)
        for i in range(len(pcs)):
            open3d.io.write_point_cloud(
                filename=f"/home/cplus/projects/m.tarek_master/graval_detection_3D/point_cloud_segmentation/regions_of_cloud_scaled/1/pc/{i}.ply",
                pointcloud=pcs[i]
            )
        