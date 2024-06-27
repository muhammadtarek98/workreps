import open3d as o3d
import algorithms
import os
import numpy as np
#stones_dir="/home/cplus/projects/m.tarek_master/graval_detection_3D/data/cloud_scaled_half_ascii/1/stones"
first_radii=[]
second_radii=[]
third_radii=[]
volumes=[]
for i in range(1,13):
    stones_dir=f"/home/cplus/projects/m.tarek_master/graval_detection_3D/data/cloud_scaled_half_ascii/{i}/stones"
    for file in os.listdir(stones_dir):
        if file.endswith(".ply"):
            pcd=o3d.io.read_point_cloud(os.path.join(stones_dir,file))
            points_3d=np.asarray(pcd.points)
            points_2d=algorithms.project_2d(points=points_3d)
            dp=algorithms.fit_ellipse_cv(points_2d)
            volume = algorithms.calculate_volume_using_fitted_ellipse(poly=points_2d) * pow(100, 3)
            r1=dp["r1"]
            r2=dp["r2"]
            r3=min(r1,r2)
            first_radii.append(r1)
            second_radii.append(r2)
            third_radii.append(r3)
            volumes.append(volume)



r1=np.array(first_radii,dtype=np.float32)
r2=np.array(second_radii,dtype=np.float32)
r3=np.array(third_radii,dtype=np.float32)
v=np.array(volumes)
print(np.average(r1))
print(np.average(r2))
print(np.average(r3))



