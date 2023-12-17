import cv2 as cv
import numpy as np
image_path=r"D:\graval detection project\resyris\resyris\test_data_realworld\left_rgb\000.png"
mask_path = r"D:\graval detection project\resyris\resyris\test_data_realworld\gt\000.png"

img=cv.imread(image_path)
#img=cv.cvtColor(img,cv.COLOR_BGR2RGB)
masked_img=cv.imread(mask_path,cv.IMREAD_GRAYSCALE)
_, binary_mask = cv.threshold(masked_img, 1, 255, cv.THRESH_BINARY)
print(img.shape)
# Find contours in the binary mask
contours, _ = cv.findContours(binary_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
dict={}
# Iterate through the contours (one for each object)
for i, contour in enumerate(contours):
    # Extract points (coordinates) from the contour
    points = contour.squeeze().tolist()
    for x,y in points:
        dict[f"object{i+1}"]=[x/img.shape[0],y/img.shape[1]]
    # Do something with the points, e.g., print or process them
    print(f"Object {i + 1} Points: {points}")
print(dict)
# Display the original images with contours (for visualization)
cv.drawContours(img, contours, -1, (0, 255, 0), 2)  # Draw contours on the original images
cv.imwrite('Original Image with Contours.png', img)
#cv.waitKey(0)
#cv.destroyAllWindows()
"""from ultralytics import YOLO
import cv2 as cv
import math
import os
import numpy as np

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
    main_dir = r"D:\graval detection project\resyris\resyris\test_data_realworld\dataset\images\test"
    paths = [dir for dir in os.listdir(main_dir)]
    for i, path in enumerate(paths):
        results = predict(weights_path=r"D:\graval detection project\runs\segment\train2\weights\best.pt",
                          images_path=os.path.join(main_dir, path), saving_flag=False)
        for j in range(len(results)):
            masks = results[j].masks
            print(f"number of detect stones in image {i + 1}: {len(masks)}")
            for k in range(len(masks)):
                mask = masks[k]
                obj = masks.data[k].cpu()
                poly_list = masks.xy[j]
                obj_numpy = obj.numpy()
                poly = np.array(poly_list, dtype=np.float32)
                print(f"area of the segment {k + 1} in image {i + 1}: ",
                      calculate_area_of_fitted_ellipse_in_cm(poly_list), "cm^2")
                print(f"volume of the segment {k + 1} in image {i + 1}: ",
                      calculate_volume_using_fitted_ellipse_in_cm(poly_list), "cm^3")
                dp = fit_ellipse(poly)
            angle = dp["angle"]
            axes = dp["axes"]
            center_coordinates = dp["center_coordinates"]
            r1 = dp["r1"]
            r2 = dp["r2"]
        img = cv.imread(filename=os.path.join(main_dir, path))
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        cv.ellipse(img=img, angle=angle,
                   center=center_coordinates,
                   axes=axes, color=(255, 0, 0), thickness=2,
                   startAngle=0, endAngle=360)

        cv.imshow("test", img)
        cv.waitKey(0)
"""