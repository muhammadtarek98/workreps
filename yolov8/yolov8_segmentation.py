from ultralytics import YOLO
import cv2
import os
import numpy as np
import torch
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
imagesize = 480


def train(weights_path,data_dir,project,single_cls_flag):
    model = YOLO(weights_path, task='segment')
    results=model.train(data=data_dir,
                epochs=500,
                imgsz=imagesize, pretrained=True, batch=1, cos_lr=True, lr0=1e-3,lrf=1e-4, workers=4, val=False,
                optimizer='Adam', single_cls=single_cls_flag, boxes=False, augment=True, save=True,project=project)
    return results


def validation(weights_path,data_dir,confidence_threshold=0.5):
    model = YOLO(weights_path)
    metrics = model.val(plots=True, batch=1, conf=confidence_threshold,data=data_dir,boxes=False,project=project,split="test")
    print(metrics.box.map)
    print(metrics.box.map50)
    print(metrics.box.map75)
    print(metrics.box.maps)
    print(metrics.seg.map)
    print(metrics.seg.map50)
    print(metrics.seg.map75)
    print(metrics.seg.maps)


def predict(weights_path, saving_flag,project_name=None, images_path=None, video_path=None, confidence_threshold=0.5,stream_flag=False):
    model = YOLO(weights_path,
                 task='segment')
    global results
    if stream_flag:
        results = model(source=images_path, stream=True)
    else:

        results = model.predict(save=saving_flag,
                            source=images_path,
                            imgsz=imagesize,
                            conf=confidence_threshold, boxes=False,stream=stream_flag,project=project_name)
    return results


if __name__ == '__main__':
    data_dir=r"D:\graval detection project\datasets\under water dataset\data.yaml"
    #project="trash_and_marine_life_detection"
    project="stone segmentation"
    #weights_dir=r"D:\graval detection project\yolov8n-seg.pt"
    #results=train(weights_path=weights_dir,data_dir=data_dir,project=project,single_cls_flag=True)
    #validation(weights_path=r"D:\graval detection project\mareim runs\train_lastdataset_update_UW\train7\weights\best.pt",data_dir=data_dir,confidence_threshold=0.7)
    #main_dir = r"D:\graval detection project\resyris\resyris\test_data_realworld\dataset\images\val"
    #video = r"D:\graval detection project\UV light\videos_3\video_20231031_121233.mp4"
    dir=r"D:\graval detection project\datasets\under water dataset\images\test"
    #image_list=[ os.path.join(r'D:\graval detection project\datasets\under water dataset\images\train',i) for i in os.listdir(r'D:\graval detection project\datasets\under water dataset\images\train')]
    predict(weights_path=r"D:\graval detection project\mareim runs\train_lastdataset_update_UW\train7\weights\best.pt", saving_flag=True,images_path=dir,confidence_threshold=0.7)



