from PIL import Image
import numpy as np


left_image = Image.open(r"D:\graval detection project\resyris\resyris\test_data_realworld\left_rgb\000.png").convert('L')
right_image = Image.open(r"D:\graval detection project\resyris\resyris\test_data_realworld\right_rgb\000.png").convert('L')
ground_truth_mask = Image.open(r"D:\graval detection project\resyris\resyris\test_data_realworld\gt\000.png")
image_array = np.array(left_image)
mask_array = np.array(ground_truth_mask)
print(mask_array.shape)
image_height, image_width = image_array.shape[:2]


mask_height, mask_width = mask_array.shape[:2]

scale_factor_mm_per_pixel = 0.1
image_area = image_height * image_width
mask_area = np.sum(mask_array > 0)
background_area = image_area - mask_area

print(f"Image Height: {image_height} pixels")
print(f"Image Width: {image_width} pixels")
print(f"Image Area: {image_area} square pixels")
print(f"Mask Height: {mask_height} pixels")
print(f"Mask Width: {mask_width} pixels")
print(f"Mask Area: {mask_area} square pixels")
print(f"Background Area: {background_area} square pixels")
print(f"number of stones:{len(mask_array)}")
mask_area_mm2 = mask_area * (scale_factor_mm_per_pixel ** 2)
print(f"area in mm^2={mask_area_mm2} mm^2")


# Convert the images and mask to NumPy arrays
left_image_array = np.array(left_image)
right_image_array = np.array(right_image)
mask_array = np.array(ground_truth_mask)
print(left_image_array.shape)
print(right_image_array.shape)
print(mask_array.shape)
# Convert the left images to grayscale
left_image_gray_array = np.array(left_image)

# Apply the mask to each images
left_masked = left_image_gray_array * (mask_array > 0)
right_masked = right_image_array * (mask_array > 0)

# Calculate the area of the masked regions in each images
left_masked_area = np.sum(left_masked > 0)
right_masked_area = np.sum(right_masked > 0)

# Print the results
print(f"Total Masked Area in Left Image: {left_masked_area} square pixels")
print(f"Total Masked Area in Right Image: {right_masked_area} square pixels")