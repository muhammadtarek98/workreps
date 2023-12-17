import os
import numpy as np
import json
from PIL import Image
import cv2
main_dir= r"D:\graval detection project\datasets\unperpared data\images under water\images"
masks_dir=r"D:\graval detection project\datasets\unperpared data\images under water\masks"
for file in os.listdir(main_dir):
    if file.endswith(".json"):
        with open(os.path.join(main_dir,file)) as f:
            data=json.load(f)
        image_height=data["imageHeight"]
        image_width=data["imageWidth"]
        mask = np.zeros((image_height, image_width), dtype=np.uint8)
        image_name=str(data["imagePath"])
        image_name=image_name.replace("../images/","")
        #print(image_name)
        for shape in data["shapes"]:
            label = shape["label"]
            points = shape["points"]
            points = np.array(points, dtype=np.int32)
            cv2.fillPoly(mask, [points], color=(0, 255, 0))
        cv2.imwrite(filename=
                            fr"D:\graval detection project\datasets\under_water_masks_dataset\train\colorful_masks\{image_name}"
                            , img=mask)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

