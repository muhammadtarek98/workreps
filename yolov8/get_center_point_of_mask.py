import cv2 as cv
import math
from PIL import Image, ImageDraw
import os
import numpy as np
import torch

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from yolov8_segmentation import predict
def get_the_center(result):
    dp=dict()
    xyxy_format=result.boxes.xyxy
    #print(xyxy_format.shape)
    x_min,y_min,x_max,y_max=xyxy_format[0][:4]

    center_x=float((x_min+x_max)/2)
    center_y=float((y_min+y_max)/2)
    dp["center_x"]=center_x
    dp["center_y"]=center_y
    return dp



if __name__=="__main__":
    file = r"D:\graval detection project\datasets\under water dataset\images\test\RossellidaeOther029.jpg"
    check_point=r"D:\graval detection project\mareim runs\segment\train_UW_1080\weights\best.pt"
    img=cv.imread(file,flags=cv.IMREAD_COLOR)
    #img=cv.cvtColor(img,cv.COLOR_BGR2RGB)
    results = predict(weights_path=check_point,
                      images_path=file, saving_flag=False,confidence_threshold=0.1)
    #print(results)

    center_point_of_objects=[]
    for result in results:
        img_shape=result.orig_shape
        speed=result.speed
        r=result.cpu()
        #iterate over detect objects
        for i in range(len(result)):
            r=result[i]
            dp=get_the_center(r)
            x=dp["center_x"]
            y=dp["center_y"]
            center_point_of_objects.append((x,y))
    for x,y in center_point_of_objects:
        cv.circle(img=img,center=[int(x),int(y)],radius=1,color=(0,255,0),thickness=-1)
    print(len(center_point_of_objects))
    cv.imshow("result",img)
    cv.waitKey(0)





    print(i)


