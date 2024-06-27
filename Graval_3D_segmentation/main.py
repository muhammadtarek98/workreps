import packages_versions
import open3d as o3d
import os
from scipy.spatial import KDTree
import algorithms
import numpy as np
from g3point_python.segment import segment_labels
from g3point_python.detrend import rotate_point_cloud_plane, orient_normals

"""
first step filter out the sands
"""


def separate_the_stones() -> None:
    num_of_stones = []
    for file in range(1, 13):
        print(f"file {file} is under processing")
        file_name = f"/home/cplus/projects/m.tarek_master/graval_detection_3D/data/cloud_scaled_half_ascii/{file}/{file}.ply"
        segmented_file = f"/home/cplus/projects/m.tarek_master/graval_detection_3D/data/cloud_scaled_half_ascii/{file}/{file}_segmentated.ply"
        stones_dir = f"/home/cplus/projects/m.tarek_master/graval_detection_3D/data/cloud_scaled_half_ascii/{file}/stones"
        pcd_orig = algorithms.preprocess(file_name=file_name)
        xyz = np.asarray(pcd_orig.points)
        axis = np.array([0, 0, 1])
        xyz_detrended = rotate_point_cloud_plane(xyz, axis)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_detrended)
        pcd.colors = o3d.utility.Vector3dVector(pcd_orig.colors)
        print(len(np.asarray(pcd.points)))
        K = 100
        tree = KDTree(xyz_detrended)  # build a KD tree
        neighbors_distances, neighbors_indexes = tree.query(xyz_detrended, K + 1)
        neighbors_distances, neighbors_indexes = neighbors_distances[:, 1:], neighbors_indexes[:, 1:]
        pcd.normals = pcd_orig.normals
        centroid = np.mean(xyz_detrended, axis=0)
        sensor_center = np.array([centroid[0], centroid[1], 1000])
        normals = orient_normals(xyz_detrended, np.asarray(pcd.normals), sensor_center)
        dp = segment_labels(xyz_detrended, K, neighbors_indexes, braun_willett=False)
        labels = dp["labels"]
        nlabels = dp["nlabels"]
        labelsnpoint = dp["labelsnpoint"]
        stacks = dp["stacks"]
        ndon = dp["ndon"]
        num_of_stones.append(len(stacks))
        local_maximum_indexes = dp["local_maximum_indexes"]
        colors = np.random.rand(len(stacks), 3)[labels, :]
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd_sinks = o3d.geometry.PointCloud()
        pcd_sinks.points = o3d.utility.Vector3dVector(xyz_detrended[local_maximum_indexes, :])
        pcd_sinks.paint_uniform_color(np.array([1., 0., 0.]))
        # clouds = (('pcd', pcd, None, 3),('pcd_sinks', pcd_sinks, None, 5))
        #algorithms.save_pcds(stacks=stacks,segmented_pcd=pcd,stones_dir=stones_dir,segmented_file=segmented_file,pcd=pcd,pcd_sinks=pcd_sinks)
        print(nlabels, " ", len(np.asarray(pcd_sinks.points)), " ", " ", len(stacks))
        print(len(np.asarray(pcd_sinks.points)))

    print(num_of_stones)


num_of_stones = 0
total_volume = 0.0
l = []
volumes = []
first_radii = []
second_radii = []
third_radii = []

main_dir = "/home/cplus/projects/m.tarek_master/graval_detection_3D/data/cloud_scaled_half_ascii/"
for file in range(1, 13):
    stones_dir = f"{main_dir}{file}/stones"
    for i in os.listdir(stones_dir):
        print(f"{i} stone in batch {file}")
        pcd = algorithms.preprocess(file_name=os.path.join(stones_dir, i))
        points = np.asarray(pcd.points)
        try:
            projected_points = algorithms.project_2d(points=points)
            volume = algorithms.calculate_volume_using_fitted_ellipse(poly=projected_points) * pow(100, 3)
            dp = algorithms.fit_ellipse_cv(projected_points)
            r1 = dp["r1"] * 100
            r2 = dp["r2"] * 100
            angle = dp["angle"]
            r3 = min(r1, r2)

            first_radii.append(r1)
            second_radii.append(r2)
            third_radii.append(r3)
            #print(f"stone {i} volume is {volume} cm^3")
            total_volume += volume
            """algorithms.ellipsoid_fitting(r1=r1,
                                         r2=r2,
                                         r3=r3,
                                         points=points,
                                         angle=angle)

"""
        except:
            l.append(os.path.join(stones_dir, i))
#print(total_volume)
#print(l[0])
r1 = np.array(first_radii, dtype=np.float32)
r2 = np.array(second_radii, dtype=np.float32)
r3 = np.array(third_radii, dtype=np.float32)
#v=np.array(volumes)
print(np.average(r1))
print(np.average(r2))
print(np.average(r3))
