import os
import cv2
import numpy as np
import os
from skimage.measure import label, regionprops


def get_images(main_dir):
    l = []
    for dir in os.listdir(main_dir):
        if dir.endswith('.png'):
            l.append(os.path.join(main_dir, dir))
    return l


# Load the images and their corresponding masked data
image_paths = get_images(
    r"D:\graval detection project\resyris\resyris\test_data_realworld\left_rgb")
mask_paths = get_images(
    r"D:\graval detection project\resyris\resyris\test_data_realworld\gt")
"""
print(image_paths[0])
print(mask_paths[0])
"""
# Create YOLOv8 instance segmentation annotation file
annotation_file = open("train_yolov8_instance_segmentation_annotations.txt", "w")

# Define the classes (you can customize this based on your dataset)
class_mapping = {
    1: "stone"
}

# Initialize a class ID counter
class_id_counter = 1

# Iterate through images and masks
for image_path, mask_path in zip(image_paths, mask_paths):
    # Load the images and its mask
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Label connected components in the mask
    num_objects, labeled_mask = cv2.connectedComponents(mask)

    # Iterate through objects
    for object_id in range(1, num_objects):
        # Extract the mask for the current object
        object_mask = (labeled_mask == object_id).astype(np.uint8)

        # Calculate object properties (center, width, height)
        object_contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        object_contour = max(object_contours, key=cv2.contourArea)
        object_x, object_y, object_w, object_h = cv2.boundingRect(object_contour)
        object_center_x = (object_x + object_w / 2) / image.shape[1]
        object_center_y = (object_y + object_h / 2) / image.shape[0]
        object_width = object_w / image.shape[1]
        object_height = object_h / image.shape[0]

        # Determine the class ID based on your class mapping
        # For example, you can assign class IDs based on object properties
        class_id = class_id_counter

        # Write YOLOv8 annotation for the object
        annotation_line = f"{image_path},{object_center_x:.6f},{object_center_y:.6f},{object_width:.6f},{object_height:.6f},{class_id},{class_mapping[1]}"
        annotation_file.write(annotation_line + "\n")

        # Increment the class ID counter
        class_id_counter += 1

# Close the annotation file
annotation_file.close()
