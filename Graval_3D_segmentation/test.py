import open3d.ml.torch as ml3d  # or open3d.ml.tf as ml3d
# construct a dataset by specifying dataset_path
dataset = ml3d.datasets.KITTI(dataset_path='/home/cplus/projects/m.tarek_master/graval_detection_3D/kitti_dataset/dataset')
# get the 'all' split that combines training, validation and test set
all_split = dataset.get_split('all')
# print the attributes of the first datum
print(all_split.get_attr(0))
# print the shape of the first point cloud
print(all_split.get_data(0)['point'].shape)
# show the first 400 frames using the visualizer
vis = ml3d.vis.Visualizer()
vis.visualize_dataset(dataset, "training", indices=range(400))

"""
import open3d
import os
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
def kmeans_clustering_ellipsoids(pcd):
    kmeans = KMeans(n_clusters=20, max_iter=10000000, algorithm="lloyd")
    labels = kmeans.fit_predict(np.asarray(pcd.points))

    clusters = []
    for cluster_label in range(kmeans.n_clusters):
        cluster_points = pcd.select_by_index(np.where(labels == cluster_label)[0])
        clusters.append(cluster_points)

    return clusters

def fit_ellipsoid(points):
    # Fit an ellipsoid to the given points using alpha shape
    alpha =1 # Adjust the alpha parameter based on your data
    mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(points, alpha)
    return mesh

def calculate_ellipsoid_volume(mesh):
    volume = mesh.get_surface_area()
    return volume
print("read_file")
pcd = open3d.io.read_point_cloud(r"/home/cplus/projects/m.tarek_master/graval_detection_3D/point_cloud_segmentation/label_1.ply")
R= pcd.get_rotation_matrix_from_xyz((180, 90, 0))
pcd_rotated =pcd.rotate(R, (0,0,0))
print("estimate_normals")
#pcd_rotated.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.0001, max_nn=100000))

#open3d.visualization.draw_geometries(geometry_list=[pcd], window_name='Original Point Cloud')
#print("run kmean cluster")
#kmeans = KMeans(n_clusters=2,max_iter=10000,algorithm="lloyd")
#labels = kmeans.fit_predict(np.asarray(pcd_rotated.colors))
#print(labels)
#pcd_0=[]
#pcd_1=[]

#num_points_in_cluster = np.bincount(labels)
#print(num_points_in_cluster)
#colors = np.asarray(pcd.colors)
#colors[labels == 0] = [1, 0, 0]
#colors[labels == 1] = [0, 1, 0]
#separate the stones from the ground
#pcd_0 = pcd_rotated.select_by_index(np.where(labels == 0)[0])
#pcd_1=pcd_rotated.select_by_index(np.where(labels == 1)[0])

#R= pcd.get_rotation_matrix_from_xyz((180, 0, 0))
#pcd_rotated =pcd_0.rotate(R, (0,0,0))

#open3d.io.write_point_cloud(filename="label_0.ply", write_ascii=True, pointcloud=pcd_0)
#pcd_1 = pcd.select_by_index(np.where(labels == 1)[0])
#open3d.io.write_point_cloud(filename="label_1.ply", write_ascii=True, pointcloud=pcd_1)
#kmeans = KMeans(n_clusters=4,max_iter=1000,algorithm="lloyd")
#labels = kmeans.fit_predict(np.asarray(pcd_0.points))
#colors_pcd_0 = np.asarray(pcd_0.points)
#colors_pcd_0[labels == 0] = [1, 0, 0]
#colors_pcd_0[labels == 1] = [0, 1, 0]
#colors_pcd_0[labels == 2]=[0,0,1]
#colors_pcd_0[labels == 3]=[1,1,0]
#colors_pcd_0[labels == 4]=[1,0,1]
#print("fiting ellipsoid")
#ellipsoid = fit_ellipsoid(pcd_1)
#ellipsoid_volume = calculate_ellipsoid_volume(ellipsoid)
#print("Ellipsoid Volume:", ellipsoid_volume)
#print(np.unique(np.asarray(colors)))
#print("visualize ellipsoid")
open3d.visualization.draw_geometries(geometry_list=[pcd_rotated], window_name='Clustered Point Cloud with Ellipsoid')
#print(f"total volume:{calculate_ellipsoid_volume(np.asarray(pcd_rotated))}")
#open3d.io.write_point_cloud(filename="kmean_results.ply",write_ascii=True,pointcloud=pcd)
#print("Number of points in cluster 0:", num_points_in_cluster[0])
#print("Number of points in cluster 1:", num_points_in_cluster[1])

ellipsoid_clusters = kmeans_clustering_ellipsoids(pcd_rotated)

# Visualize the clustered ellipsoids
for idx, ellipsoid_cluster in enumerate(ellipsoid_clusters):
    ellipsoid = fit_ellipsoid(ellipsoid_cluster)
    ellipsoid_volume = calculate_ellipsoid_volume(ellipsoid)
    print(f"Ellipsoid {idx + 1} Volume:", ellipsoid_volume)
    open3d.visualization.draw_geometries(geometry_list=[ellipsoid_cluster], window_name=f'Ellipsoid Cluster {idx + 1}')
"""

"""import open3d
import os
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
import open3d as o3d
import numpy as np
from scipy.optimize import minimize
from sklearn.decomposition import PCA
import cv2 as cv
def fit_ellipse_with_cv(poly):
    ((cent_x, cent_y), (width, height), angle) = cv.fitEllipse(points=poly)
    r1, r2 = width, height
    axes = (int(width), int(height))
    r1 = r1 / 2
    r2 = r2 / 2
    angle = angle
    d = dict()
    d["angle"] = angle
    d["axes"] = axes
    d["center_coordinates"] = (int(cent_x), int(cent_y))
    d["r1"] = r1
    d["r2"] = r2
    return d
def calculate_volume_using_fitted_ellipse(poly):
    dp = fit_ellipse(poly)
    r1 = dp['r1']
    r2 = dp['r2']
    return (4 / 3.0) * np.PI * r1 * r2 * min(r1, r2)
def fit_ellipse(pcd):
    # Extract x, y, z coordinates from the point cloud
    points = np.asarray(pcd.points)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # Define the objective function for fitting an ellipse
    def objective(params):
        a, b, c, d, e, f = params
        return np.sum((a*x**2 + b*x*y + c*y**2 + d*x + e*y + f - z)**2)

    # Initial guess for ellipse parameters
    initial_params = [1.0, 0.0, 1.0, 0.0, 0.0, 0.0]

    # Minimize the objective function to obtain ellipse parameters
    result = minimize(objective, initial_params)

    # Check if the optimization was successful
    if result.success:
        # Extract the ellipse parameters
        a, b, c, d, e, f = result.x

        # Calculate eigenvalues of the ellipse matrix
        ellipse_matrix = np.array([[a, b/2], [b/2, c]])
        eigenvalues, _ = np.linalg.eig(ellipse_matrix)

        # Check if eigenvalues are valid
        if np.all(np.isreal(eigenvalues)):
            # Calculate radii from eigenvalues
            radii = np.sqrt(-1 / eigenvalues)

            return radii
        else:
            print("Invalid eigenvalues. Check input data.")
            return None
    else:
        print("Optimization did not converge. Check input data.")
        return None


def kmeans_clustering_ellipsoids(pcd):
    kmeans = KMeans(n_clusters=150, max_iter=10000000, algorithm="full",init="random",n_init="auto")
    labels = kmeans.fit_predict(np.asarray(pcd.points))
    clusters = []
    for cluster_label in range(kmeans.n_clusters):
        cluster_points = pcd.select_by_index(np.where(labels == cluster_label)[0])
        clusters.append(cluster_points)
    return clusters

def fit_ellipsoid(points):
    alpha = 1
    mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(points, alpha)
    return mesh

def calculate_ellipsoid_volume(mesh):
    volume = mesh.get_surface_area()
    return volume

def calculate_convex_hull_volume(pcd):
    convex_hull, _ = pcd.compute_convex_hull()
    volume = convex_hull.get_oriented_bounding_box().volume()
    return volume

print("read_file")
pcd = open3d.io.read_point_cloud(r"/home/cplus/projects/m.tarek_master/graval_detection_3D/rec_scaled_ascii_label_0.ply")
R = pcd.get_rotation_matrix_from_xyz((180,0, 0))
pcd_rotated = pcd.rotate(R, (0, 0, 0))
#alpha = 0.3
#mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd_rotated, alpha)
#mesh.compute_vertex_normals()
#open3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
#print("estimate_normals")
#open3d.visualization.draw_geometries(geometry_list=[pcd_rotated], window_name='Clustered Point Cloud with Ellipsoid')
ellipsoid_clusters = kmeans_clustering_ellipsoids(pcd_rotated)
print(type(ellipsoid_clusters[0]))
total_ellipsoids_volumes_with_convex_hull=0.0 #in m^3
total_ellipsoids_volume_calculate_ellipsoid_volume=0.0 #in m^3

for idx, ellipsoid_cluster in enumerate(ellipsoid_clusters):
    ellipsoid = fit_ellipsoid(ellipsoid_cluster)
    alpha = 0.3
    points = np.asarray(ellipsoid_cluster)
    print("fit pca to PCD")
    pca = PCA(n_components=2)
    points_2d = pca.fit_transform(points)
    results=calculate_volume_using_fitted_ellipse(poly=points_2d)
    print(f"Ellipsoid {idx + 1} resluts :{results}f")



    #mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(ellipsoid_cluster, alpha)
    #mesh.compute_vertex_normals()
    #ellipsoid_volume = calculate_ellipsoid_volume(ellipsoid)

    #print(f"Ellipsoid {idx + 1} Volume (from Ellipsoid):", ellipsoid_volume)
    #convex_hull_volume = calculate_convex_hull_volume(ellipsoid_cluster)
    #total_ellipsoids_volumes_with_convex_hull+=convex_hull_volume
    #total_ellipsoids_volume_calculate_ellipsoid_volume+=ellipsoid_volume
    #ellipsoid_radii = fit_ellipse(ellipsoid)
    #print(f"Ellipsoid {idx + 1} radii :{ellipsoid_radii}")

    #print(f"Ellipsoid {idx + 1} Volume (from Convex Hull):", convex_hull_volume)
    #open3d.visualization.draw_geometries(geometry_list=[ellipsoid_cluster], window_name=f' PCD cluster {idx + 1}')
    #open3d.visualization.draw_geometries(geometry_list=[ellipsoid], window_name=f'Ellipsoid Cluster {idx + 1}')
    #open3d.visualization.draw_geometries(geometry_list=[mesh],window_name=f'Ellipsoid Cluster mesh {idx + 1}')
    #open3d.io.write_triangle_mesh(filename=f"Ellipsoid_{idx + 1}.obj",write_ascii=True,mesh=mesh,print_progress=True)
    #break


#print(f"total volumes with convex hull {total_ellipsoids_volumes_with_convex_hull}")
#print(f"total volumes with get surface area {total_ellipsoids_volume_calculate_ellipsoid_volume} ") """