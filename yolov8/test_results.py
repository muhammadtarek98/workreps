from ultralytics import YOLO
import cv2 as cv
import math
from PIL import Image, ImageDraw
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
imagesize = 720
scale_factor_cm_per_pixel = 0.028






def predict(weights_path, images_path, saving_flag):
    model = YOLO(weights_path,
                 task='segment')  # load a custom model
    results = model.predict(save=saving_flag,
                            source=images_path,
                            imgsz=imagesize,
                            conf=0.4, boxes=False)
    # print(results.data)
    return results


def test_fit_circle(poly):
    (x, y), radius = cv.minEnclosingCircle((poly))
    center = (int(x), int(y))
    print(x, " ", y)
    print(radius)

    img = cv.circle(obj_numpy, center, radius=int(radius), color=(0, 255, 0), thickness=2)
    return img


def fit_ellipse(poly):
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


def calculate_area_of_ellipse_in_pixels(poly):
    dp = fit_ellipse(poly)
    r1 = dp['r1']
    r2 = dp['r2']
    return math.pi * r1 * r2


def calculate_area_of_fitted_ellipse_in_cm(poly):
    dp = fit_ellipse(poly)
    r1 = dp['r1']
    r2 = dp['r2']
    r1_cm = r1 * scale_factor_cm_per_pixel
    r2_cm = r2 * scale_factor_cm_per_pixel
    print(r1_cm, " ", r2_cm)
    return math.pi * r1_cm * r2_cm


def calculate_volume_using_fitted_ellipse_in_cm(poly):
    dp = fit_ellipse(poly)
    r1 = dp['r1']
    r2 = dp['r2']
    r1_cm = r1 * scale_factor_cm_per_pixel
    r2_cm = r2 * scale_factor_cm_per_pixel
    return (4 / 3.0) * math.pi * r1_cm * r2_cm * min(r1_cm, r2_cm)


if __name__ == "__main__":

    file = r"D:\graval detection project\test radius\1.png"
    results = predict(weights_path=r"D:\graval detection project\runs\segment\train2\weights\best.pt",
                      images_path=file, saving_flag=False)
    for i in range(len(results)):
        masks = results[i].masks
        print(f"number of detect stones in image {i + 1}-th: {len(masks)}")
        for j in range(len(masks)):
            mask = masks[j]
            obj = mask.data[j].cpu()
            poly_list = mask.xy[j]
            obj_numpy = obj.numpy()
            poly = np.array(poly_list, dtype=np.float32)
            center_before_fitting=get_the_center_of_object(results[i])
            print(f"area of the segment {j + 1}-th in image {i + 1}-th: ",
                  calculate_area_of_fitted_ellipse_in_cm(poly_list), "cm^2")
            print(f"volume of the segment {j + 1}-th in image {i + 1}-th: ",
                  calculate_volume_using_fitted_ellipse_in_cm(poly_list), "cm^3")
            dp = fit_ellipse(poly)
            angle = dp["angle"]
            axes = dp["axes"]
            center_coordinates = dp["center_coordinates"]
            r1 = dp["r1"]
            r2 = dp["r2"]
            img = cv.imread(filename=file)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            x = cv.ellipse(img=img, angle=angle,
                           center=center_coordinates,
                           axes=axes, color=(255, 0, 0), thickness=-1,
                           startAngle=0, endAngle=360)

            cv.imshow("test", x)
            cv.waitKey(0)

    """
    main_dir=r"D:\graval detection project\resyris\resyris\test_data_realworld\dataset\images\test"
    paths = [dir for dir in os.listdir(main_dir)]
    for path in paths:
        results = predict(weights_path=r"D:\graval detection project\runs\segment\train2\weights\best.pt",
                          images_path=os.path.join(main_dir,path), saving_flag=False)
        for i in range(len(results)):
            masks = results[i].masks
            print(f"number of detect stones in image {i + 1}-th: {len(masks)}")
            for j in range(len(masks)):
                mask = masks[j]
                obj = mask.data[j].cpu()
                poly_list = mask.xy[j]
                obj_numpy = obj.numpy()
                poly = np.array(poly_list, dtype=np.float32)
                print(f"area of the segment {j + 1}-th in image {i + 1}-th: ",
                      calculate_area_of_fitted_ellipse_in_cm(poly_list), "cm^2")
                print(f"volume of the segment {j + 1}-th in image {i + 1}-th: ",
                      calculate_volume_using_fitted_ellipse_in_cm(poly_list), "cm^3")
                dp = fit_ellipse(poly)
                angle=dp["angle"]
                axes=dp["axes"]
                center_coordinates=dp["center_coordinates"]
                r1=dp["r1"]
                r2=dp["r2"]
                img=cv.imread(filename=path)
                img=cv.cvtColor(img,cv.COLOR_BGR2RGB)
                x = cv.ellipse(img=img, angle=angle,
                               center=center_coordinates,
                               axes=axes, color=(255, 0, 0), thickness=-1,
                               startAngle=0, endAngle=360)

                cv.imshow("test", x)
                cv.waitKey(0)

                # contours, _ = cv.findContours(obj_numpy, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                # masked_area_numpy = cv.contourArea(contours[0])
                # print(f"masked_area: {masked_area_numpy * (scale_factor_cm_per_pixel ** 2)} mm^2")

"""

"""#
print(type(poly))
circle_fit = test_fit_circle(poly=poly)


# 
# 
# 
# 

"""
