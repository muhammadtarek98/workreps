from yolov8_segmentation import predict
from test_results import calculate_area_of_ellipse_in_pixels
import numpy as np
import os
import arrow
import cv2

weights_path = r"D:\graval detection project\mareim runs\segment\train_UW_1080\weights\best.pt"
image_dir = r"D:\graval detection project\datasets\under water dataset\images\test"

def plugin_text_on_image(img,text,img_name,h,w):
    org=(0,w//2)
    color = (255, 0, 0)
    thickness = 2
    fontScale=1
    font = cv2.FONT_HERSHEY_SIMPLEX
    image = cv2.putText(img, text=text, org=org, fontFace=font,fontScale=fontScale, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    cv2.imwrite(fr"D:\graval detection project\Scripts\areas\{img_name}",image)
s = arrow.utcnow()
for image in os.listdir(image_dir):
    area_predicted_segments = 0
    results = predict(weights_path=weights_path, saving_flag=False, images_path=os.path.join(image_dir, image))
    img = cv2.imread(filename=os.path.join(image_dir, image), flags=0)
    h, w = img.shape
    area_of_image = h * w
    print(f"{os.path.join(image_dir, image)}: ")
    print(f"the height and width :{(h, w)} in pixels")
    for i, result in enumerate(results):
        masks = result.masks
        for j in range(len(masks)):
            object_mask = masks.data[j].cpu().numpy()
            polygon_list = masks.xy[j]
            polygon_array = np.array(polygon_list, dtype=np.float32)
            area_fitted_ellipse = calculate_area_of_ellipse_in_pixels(polygon_list)
            area_predicted_segments += area_fitted_ellipse
    text=f"""
    area of the image : {area_of_image} p^2\n
    total area of the predicted masks after fit to ellipse : {area_predicted_segments} p^2"\n
    percentage of the stones area {(area_predicted_segments / area_of_image) * 100}%"\n
    """
    #plugin_text_on_image(img,text,image,h,w)
    print(f"area of the image : {area_of_image} p^2")
    print(f"total area of the predicted masks after fit to ellipse : {area_predicted_segments} p^2")
    print(f"percentage of the stones area {(area_predicted_segments / area_of_image) * 100}%")

print(f"running {arrow.utcnow() - s} seconds for predict {len(os.listdir(image_dir))} images")
