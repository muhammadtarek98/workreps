import open3d
import os
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
import open3d
import os
import numpy as np
from sklearn.cluster import KMeans
pcd = open3d.io.read_point_cloud("/home/cplus/projects/m.tarek_master/graval_detection_3D/point_cloud_segmentation/cloud_Scaled_ascii.ply")
R = pcd.get_rotation_matrix_from_xyz((180,0, 0))
pcd_rotated = pcd.rotate(R, (0, 0, 0))
#open3d.visualization.draw_geometries([pcd])
# Perform k-means clustering
kmeans = KMeans(n_clusters=2, max_iter=100000, algorithm="lloyd")
labels = kmeans.fit_predict(np.asarray(pcd_rotated.colors))

pcd_0 = pcd_rotated.select_by_index(np.where(labels == 0)[0])
open3d.io.write_point_cloud(filename="cloud_Scaled_ascii_label_0.ply", write_ascii=True, pointcloud=pcd_0)
#open3d.visualization.draw_geometries([pcd_0])
pcd_1 = pcd_rotated.select_by_index(np.where(labels == 1)[0])
open3d.io.write_point_cloud(filename="cloud_Scaled_ascii_label_1.ply", write_ascii=True, pointcloud=pcd_1)
#open3d.visualization.draw_geometries([pcd_1])

