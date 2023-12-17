import cv2
import os

# Set the directory containing your masks
mask_dir = r"D:\graval detection project\datasets\under_water_masks_dataset\test\masks"

# Set the output directory where you want to save the colored masks
output_dir = r"D:\graval detection project\datasets\under_water_masks_dataset\test\colorful_masks"

# Set the color you want to assign to non-white pixels
new_color = (0, 255, 0)  # You can change this to the RGB/ BGR values you prefer

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Iterate through each mask in the directory
for mask_file in os.listdir(mask_dir):
        # Read the mask
        mask_path = os.path.join(mask_dir, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Set white pixels to the new color
        colored_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        colored_mask[mask == 255] = new_color

        # Save the colored mask to the output directory
        output_path = os.path.join(output_dir, mask_file)
        cv2.imwrite(output_path, colored_mask)

print("Conversion completed.")
