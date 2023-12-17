import cv2
import numpy as np
import math
from yolov8_segmentation import predict
from test_results import fit_ellipse

cap = cv2.VideoCapture(9)
cap.set(3, 640)
cap.set(4, 480)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
classNames = ["stone"]
while True:
    success, img = cap.read()
    results = predict(weights_path=r"D:\graval detection project\runs\segment\train_720_styled\weights\best.pt",
                      saving_flag=False, stream_flag=True, images_path=img)
    # coordinates
    for i in range(len(results)):
        masks = results[i].masks
        print(f"number of detect stones in image {i + 1}-th: {len(masks)}")
        for j in range(len(masks)):
            mask = masks[j]
            obj = mask.data[j].cpu()
            poly_list = mask.xy[j]
            obj_numpy = obj.numpy()
            poly = np.array(poly_list, dtype=np.float32)
            # print(f"area of the segment {j + 1}-th in image {i + 1}-th: ",calculate_area_of_fitted_ellipse_in_cm(poly_list), "cm^2")
            # print(f"volume of the segment {j + 1}-th in image {i + 1}-th: ",calculate_volume_using_fitted_ellipse_in_cm(poly_list), "cm^3")
            dp = fit_ellipse(poly)
            angle = dp["angle"]
            axes = dp["axes"]
            center_coordinates = dp["center_coordinates"]
            r1 = dp["r1"]
            r2 = dp["r2"]
            # img = cv.imread(filename=file)
            # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            x = cv2.ellipse(img=img, angle=angle,
                            center=center_coordinates,
                            axes=axes, color=(255, 0, 0), thickness=-1,
                            startAngle=0, endAngle=360)
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
