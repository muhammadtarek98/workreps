from transformers import pipeline,AutoImageProcessor, AutoModelForDepthEstimation
import torch
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import cv2
checkpoint = "vinvino02/glpn-nyu"
depth_estimator = pipeline(task="depth-estimation", model=checkpoint)
image = Image.open(r"D:\graval detection project\datasets\under water dataset\images\test\218_left_2023_10_22_10_45_06.jpg")
predictions = depth_estimator(image)
print(predictions["depth"])
image_processor = AutoImageProcessor.from_pretrained(checkpoint)
model = AutoModelForDepthEstimation.from_pretrained(checkpoint)
pixel_values = image_processor(image, return_tensors="pt").pixel_values
with torch.no_grad():
    outputs = model(pixel_values)
    predicted_depth = outputs.predicted_depth
# interpolate to original size
prediction = torch.nn.functional.interpolate(
    predicted_depth.unsqueeze(1),
    size=image.size[::-1],
    mode="bicubic",
    align_corners=False,
).squeeze()
output = prediction.numpy()
print(output)
formatted = (output * 255 / np.max(output)).astype("uint8")
#depth = Image.fromarray(formatted)
print(formatted)

cv2.imshow("original image",np.array(formatted))

#plt.imshow(depth,cmap="CMRmap_r")
#plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()



