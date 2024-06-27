import open3d
import os
import numpy as np
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
import cv2 as cv
import torch 
from scipy.spatial import KDTree
from matplotlib import pyplot as plt
import pandas as pd
from clustering_and_fitting import *
if __name__ == '__main__':
        print("read_file")
        result=dict()
        result["file_name"]=[]
        result["total_volume"]=[]
        device = open3d.core.Device("CUDA:0")
        main_dir="/home/cplus/projects/m.tarek_master/graval_detection_3D/regions_of_cloud_scaled"
        #regions=["region_1","region_2","region_3","region_4","region_5","region_6"]
        pcd = open3d.io.read_point_cloud("/media/tarek/ITHDD/graval detection project/graval_detection_3D/plys/big_circle_ascii_label_1.ply")
        R = pcd.get_rotation_matrix_from_xyz((180, 0, 0))
        pcd_rotated = pcd.rotate(R, (0, 0, 0))
        alpha = 0.3 
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")  
        total_ellipsoid_volumes = 0.0
        points=np.asarray(pcd_rotated.points)
        num_of_clusters=g3_point_cloud_cluster(point_cloud=points,KNN=200)
        ellipsoid_clusters = kmeans_clustering_ellipsoids(pcd_rotated)
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
            #result["file_name"].append(os.path.join(main_dir,i))
            #result["total_volume"].append(ellipsoid_volume)
            #new = pd.DataFrame.from_dict(result)
            #new.to_csv(f"results_{j}.csv",encoding='utf-8', index=False)
            #print("Total Ellipsoid Volumes:", total_ellipsoid_volumes)  """