import os
import numpy as np
import cv2 as cv


def normalize_points(points, h, w):
    l = []
    for point in points:
        for x, y in point:
            x_norm = x / w
            y_norm = y / h
            l.append(x_norm)
            l.append(y_norm)
    return l

"""
images_path = r'D:\graval detection project\resyris\resyris\test_data_realworld\dataset\valid\images'
masks_path = r'D:\graval detection project\resyris\resyris\test_data_realworld\dataset\valid\labels'
for image_path, mask_path in zip(os.listdir(images_path), os.listdir(masks_path)):
    lines=[]
    if image_path == mask_path:
        txt_name = image_path.replace(".png", " ")
        img = cv.imread(os.path.join(images_path, image_path))
        masked_img = cv.imread(os.path.join(masks_path, mask_path), cv.IMREAD_GRAYSCALE)
        num_objects, labeled_mask = cv.connectedComponents(masked_img)
        h, w, _ = img.shape
        # Find contours in the binary mask
        for object_id in range(1, num_objects):
            # Extract the mask for the current object
            object_mask = (labeled_mask == object_id).astype(np.uint8)

            # Calculate the area of the object's mask
            object_area = float(np.sum(object_mask))

            # Find contours to generate object's mask
            contours, _ = cv.findContours(object_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            # Compute the convex hull of the object (assuming convex shapes)
            convex_hull = cv.convexHull(contours[0])
            points = normalize_points(convex_hull, h=h, w=w)
            lines.append(points)
            # Iterate through the contours (one for each object)
            # for i, contour in enumerate(contours):
            # Extract points (coordinates) from the contour
            # points=[]
            #   points = contour.squeeze().tolist()
            #  print(points)

            # for x,y in points:
            #    dict[f"object{i+1}"]=[x/img.shape[0],y/img.shape[1]]
            # Do something with the points, e.g., print or process them
        with open(
                    fr'D:\graval detection project\resyris\resyris\test_data_realworld\dataset\valid\labels in txt\{txt_name}.txt',
                    'w') as file2:
                    for line in lines:
                        class_id='0 '
                        line=str(line)
                        line = line.replace(']','')
                        line = line.replace('[','')
                        line = line.replace(',','')
                        line = class_id+str(line)+'\n'
                        file2.write(line)
"""
"""print(dict)
# Display the original images with contours (for visualization)
cv.drawContours(img, contours, -1, (0, 255, 0), 2)  # Draw contours on the original images
cv.imwrite(r'D:\graval detection project\resyris\resyris\test_data_realworld\dataset\Original Image with Contours.png',
           img)
# cv.waitKey(0)
# cv.destroyAllWindows()
"""
