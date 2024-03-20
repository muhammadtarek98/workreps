import os
import numpy as np
import torch
import open3d as o3d
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d
print(o3d.core.cuda.is_available())
kitti_dataset_dir=r'/home/cplus/projects/m.tarek_master/graval_detection_3D/kitti_dataset/dataset'
cfg_model_path=r"/home/cplus/projects/m.tarek_master/graval_detection_3D/Open3D-ML/ml3d/configs/randlanet_s3dis.yml"
model_weights=r"/home/cplus/projects/m.tarek_master/graval_detection_3D/kitti_dataset/randlanet_s3dis_202201071330utc.pth"
pcd_dir=r"/home/cplus/projects/m.tarek_master/graval_detection_3D/point_cloud_segmentation/cropped_circle_old_ascii.ply"
cfg=_ml3d.utils.Config.load_from_file(filename=cfg_model_path)
model=ml3d.models.RandLANet(**cfg.model)
print(_ml3d.__version__)

cfg.dataset['dataset_path'] = kitti_dataset_dir
dataset = ml3d.datasets.SemanticKITTI(cfg.dataset.pop('dataset_path', None), **cfg.dataset)
pipeline = ml3d.pipelines.SemanticSegmentation(model, dataset=dataset, device="gpu", **cfg.pipeline)

pipeline.load_ckpt(ckpt_path=model_weights)

pcd=o3d.io.read_point_cloud(pcd_dir)
#o3d.visualization.draw_geometries([pcd])
pcd.remove_non_finite_points()
data = {
    "point" : np.asarray(pcd.points, dtype=np.float32),
    "feat" : np.asarray(pcd.colors, dtype=np.float32) * np.float32(255.),
    "label" : np.zeros((len(pcd.points),), dtype=np.int32)

}

result=pipeline.run_inference(data=data)
prediction=[{
"name":"pc",
"points":data["point"],
"labels":result["predict_labels"]
}]
#print(prediction)
ml_visualizer=ml3d.vis.Visualizer()
ml_visualizer.visualize(prediction)





