import cv2
import numpy as np
import Configs


def image_overlay(image, segmented_image):
    alpha = 1.0  # Transparency for the original image.
    beta = 0.7  # Transparency for the segmentation map.
    gamma = 0.0  # Scalar added to each sum.
    segmented_image = segmented_image.astype(np.uint8)

    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
    #print(segmented_image.shape)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #print(image.shape)

    image = cv2.addWeighted(image, alpha, segmented_image, beta, gamma, image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return np.clip(image, a_min=0.0, a_max=1.0)


def num_to_rgb(num_arr, color_map=Configs.id2color):
    single_layer = np.squeeze(num_arr)
    output = np.zeros(num_arr.shape[:2] + (3,))

    for k in color_map.keys():
        output[single_layer == k] = color_map[k]
    print(output)
    # return a floating point array in range [0.0, 1.0]
    return np.float32(output)
