from yolo_cam.eigen_cam import EigenCAM
import ultralytics
import cv2
import numpy as np
from matplotlib import pyplot as plt
from ultralytics import YOLO
from yolo_cam.utils.image import show_cam_on_image, scale_cam_image

dp = []


def get_better_GRAD_CAM(rgb_img, previous_map=None, current_map=None, ):
    target_layers = [model.model.model[-8]]
    cam = EigenCAM(model, target_layers, task='od')
    grayscale_cam = cam(rgb_img)[0, :, :]
    cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    if len(dp) == 0:
        dp.append(cam_image)
    else:
        get_better_GRAD_CAM(rgb_img=dp[-1], current_map=cam_image)
        dp.append(cam_image)
    if cam_image.all() == dp[-1]:
        return


model = YOLO(r"D:\graval detection project\runs\segment\train_UW_1080\weights\best.pt")
img = cv2.imread(
    r"D:\graval detection project\datasets\unperpared data\images under water\DEV_000F3102E45A_22_October_2023_10_41_36_jpg\4_left_2023_10_22_10_41_36.jpg")
img = cv2.resize(img, (1080, 1080))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
rgb_img = img.copy()
img = np.float32(img) / 255
# get_better_GRAD_CAM(img)

for i in range(8):
    target_layers = [model.model.model[-9]]
    cam = EigenCAM(model, target_layers, task='od')
    grayscale_cam = cam(rgb_img)[0, :, :]
    cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    # cv2.imshow("asd",cam_image)
    cv2.imwrite(fr"D:\graval detection project\heat_map\197_left_2023_10_22_10_45_02_imgs\{i + 1}.png", cam_image)
    print(f"{i+1}.png is saved in it's choosen dir")
    # cv2.waitKey(0)
# cv2.destroyAllWindows()
# plt.imshow(cam_image)
# plt.show()
