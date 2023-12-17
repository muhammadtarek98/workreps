import cv2
import numpy as np
import json
import os
from PIL import Image
# Create a dictionary to store the COCO-style annotations
coco_annotations = {
    "images": [],
    "annotations": [],
    "categories": [{"id": 1, "name": "object"}]  # Define the category (you can customize it)
}

# Load the images and their corresponding masked data
def get_images(main_dir):
    l = []
    for dir in os.listdir(main_dir):
        if dir.endswith('.png'):
            l.append(os.path.join(main_dir, dir))
    return l

def normalize_points(points,h,w):
    l=[]
    for point in points:
        for x,y in point:
            x_norm=x/w
            y_norm=y/h
            l.append(x_norm)
            l.append(y_norm)
    return l
# Load the images and their corresponding masked data
image_paths = get_images(
    r"D:\graval detection project\resyris\resyris\test_data_realworld\left_rgb")
mask_paths = get_images(
    r"D:\graval detection project\resyris\resyris\test_data_realworld\gt")

# Iterate through images and masks
annotation_id = 1

# Iterate through images and masks
for image_path, mask_path in zip(image_paths, mask_paths):
    # Load the images and its mask
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Label connected components in the mask
    num_objects, labeled_mask = cv2.connectedComponents(mask)

    # Create a COCO-style images entry
    image_info = {
        "id": len(coco_annotations["images"]) + 1,
        "file_name": image_path,
        "height": image.shape[0],
        "width": image.shape[1]
    }
    coco_annotations["images"].append(image_info)

    # Iterate through objects
    for object_id in range(1, num_objects):
        # Extract the mask for the current object
        object_mask = (labeled_mask == object_id).astype(np.uint8)

        # Calculate the area of the object's mask
        object_area = float(np.sum(object_mask))

        # Find contours to generate object's mask
        contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Compute the convex hull of the object (assuming convex shapes)
        convex_hull = cv2.convexHull(contours[0])

        # Create a COCO-style annotation entry
        """
        segmentation has the points of the contour 
        
        """
        annotation = {
            "id": annotation_id,
            "image_id": image_info["id"],
            "category_id": 1,  # Category ID (object)
            "segmentation": normalize_points(convex_hull,h=image_info["height"],w=image_info["width"]),
            "area": object_area/(image_info['height']*image_info['width']),  # Area of the object mask
            "bbox": cv2.boundingRect(convex_hull),  # Bounding box (x, y, width, height)
            "iscrowd": 0
        }

        coco_annotations["annotations"].append(annotation)
        annotation_id += 1

# Save the annotations in COCO format to a JSON file
with open("coco_annotations.json", "w") as json_file:
    json.dump(coco_annotations, json_file)

print("Annotations converted to COCO format and saved as coco_annotations.json")
print(normalize_points(convex_hull,w=image.shape[0],h=image.shape[1]))