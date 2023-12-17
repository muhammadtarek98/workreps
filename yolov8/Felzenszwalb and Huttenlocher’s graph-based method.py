from matplotlib import pyplot as plt
from skimage import segmentation,io,color
from skimage import graph
image = io.imread(r'D:\graval detection project\datasets\under water dataset\images\train\image6.jpg')
#if image.shape[-1] == 4:
    # Convert RGBA to RGB by removing the alpha channel
 #   image = image[:, :, :3]
# Convert the image to grayscale
image_gray = color.rgb2gray(image)

# Apply the Felzenszwalb and Huttenlocher segmentation method
segments = segmentation.felzenszwalb(image_gray, scale=0.2, min_size=400)

# Create a graph-based segmentation
g = graph.rag_mean_color(image, segments)

# Merge small segments (optional)
segments = graph.cut_threshold(segments, g, 50)
print(segments.shape)
# Display the segmented image
plt.imshow(image_gray)
plt.show()
plt.imshow(segments,cmap='hot')
plt.show()
"""
image = Image.open(r'D:\graval detection project\test real stones using ZED\Explorer_HD1080_SN23755918_12-44-31.png').convert('L')
segmented_image = segmentation.felzenszwalb(image, 0.2, 400, 50)
g = graph.rag_mean_color(image, segmented_image)
fig = plt.figure(figsize=(12, 12))
a = fig.add_subplot(1, 2, 1)
plt.imshow(image)
a = fig.add_subplot(1, 2, 2)
plt.imshow(segmented_image.astype(np.uint8))
plt.show()
"""