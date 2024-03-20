import open3d
import os
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
import open3d as o3d
import numpy as np
from scipy.optimize import minimize
from sklearn.decomposition import PCA
import cv2 as cv
from matplotlib import pyplot as plt
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


def fit_ellipsoid(points):
    alpha = 1
    mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(points, alpha)
    return mesh

 
def calculate_volume_using_fitted_ellipse(poly):
    dp=fit_ellipse_cv(poly)
    r1, r2 = dp['r1'], dp['r2']
    return (4 / 3.0) * np.pi * r1 * r2 * min(r1, r2)


def kmeans_clustering_ellipsoids(pcd):
    kmeans = KMeans(n_clusters=150, max_iter=1000000000, algorithm="full",init="random",n_init="auto")
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




print("read_file")
pcd = open3d.io.read_point_cloud("/home/cplus/projects/m.tarek_master/graval_detection_3D/rec_scaled_ascii_label_0.ply")
print("apply rotation")
R = pcd.get_rotation_matrix_from_xyz((180, 0, 0))
pcd_rotated = pcd.rotate(R, (0, 0, 0))
print("apply kmean")
ellipsoid_clusters = kmeans_clustering_ellipsoids(pcd_rotated)
print(type(ellipsoid_clusters[0]))
total_ellipsoids_volumes_with_convex_hull = 0.0  # in m^3
total_ellipsoids_volume_calculate_ellipsoid_volume = 0.0  # in m^3
alpha = 0.3
for idx, ellipsoid_cluster in enumerate(ellipsoid_clusters):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(ellipsoid_cluster)
    ellipsoid = fit_ellipsoid(np.asarray(pcd.points))
    points = np.asarray(ellipsoid_cluster)
    print("fit pca to PCD")
    print(points.shape)
    pca = PCA(n_components=2)
    points_2d_pca = pca.fit_transform(points)
    mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(ellipsoid_cluster, alpha)
    mesh.compute_vertex_normals()
    ellipsoid_volume = calculate_ellipsoid_volume(ellipsoid)
    results = calculate_volume_using_fitted_ellipse(points_2d_pca)*pow(100,3)
    print(f"Ellipsoid {idx + 1} results: {results:.6f} cm^3")
    open3d.visualization.draw_geometries([mesh])
    #open3d.visualization.draw_geometries(geometry_list=[ellipsoid_cluster],window_name=f'cluster {idx + 1}')
    #results = calculate_volume_using_fitted_ellipse(points_2d)
    #print(f"Ellipsoid {idx + 1} results: {results:.6f} m^3")
    #ellipsoid = fit_ellipsoid(ellipsoid_cluster)
    alpha = 0.3
    #points = np.asarray(ellipsoid_cluster)
    print("fit pca to PCD")
    #pca = PCA(n_components=2)
    mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(ellipsoid, alpha)
    mesh.compute_vertex_normals()
    #open3d.visualization.draw_geometries(geometry_list=[ellipsoid_cluster], window_name=f' PCD cluster {idx + 1}')
    open3d.visualization.draw_geometries(geometry_list=[ellipsoid], window_name=f'Ellipsoid Cluster {idx + 1}')
    open3d.visualization.draw_geometries(geometry_list=[mesh],window_name=f'Ellipsoid Cluster mesh {idx + 1}')
    #points_2d = pca.fit_transform(points)
    #results=calculate_volume_using_fitted_ellipse(poly=points_2d)
    #print(f"Ellipsoid {idx + 1} resluts :{results}f")
    