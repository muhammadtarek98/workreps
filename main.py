import cv2 as cv
import numpy as np
import glob

checkboardsize = (15, 16)
squaresize = 30
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
object_points = []
img_points = []
imgs = glob.glob(r"D:\camera calibration\check board images\cam0\data\*.png")
idx = 1
image = None
for image_file in imgs:
    image = cv.imread(image_file)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, checkboardsize, None)
    if ret:
        # corner2=cv.cornerSubPix(gray, corners, (16, 17), (-1, -1), criteria)
        object_points.append(
            np.zeros((checkboardsize[0] * checkboardsize[1], 3), np.float32)
        )
        object_points[-1][:, :2] = np.mgrid[
            0: checkboardsize[0], 0: checkboardsize[1]
        ].T.reshape(-1, 2)
        img_points.append(corners)
        # cv.drawChessboardCorners(image, checkboardsize, corner2, ret)

    """dir = rf"D:\camera calibration\ results\cam1\{idx}.png".format(idx=idx)
    cv.imwrite(dir, image)
    idx += 1"""
h, w = image.shape[:2]
ret, cameraMatrix, dist, rotation_vector, translation_vector = cv.calibrateCamera(
    object_points, img_points, (h, w), None, None
)
"""
print("camera matrix")
print(cameraMatrix)
print("distortion")
print(dist)
print("rotation matrix")
print(rotation_vector)
print("translation matrix")
print(translation_vector)
"""

mean_error = 0
for i in range(len(object_points)):
    imgpoints2, _ = cv.projectPoints(object_points[i], rotation_vector[i], translation_vector[i], cameraMatrix, dist)
    error = cv.norm(img_points[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error

print("total error: {}".format(mean_error/len(object_points)))

"""
reprojection_errors = []
for i in range(len(object_points)):
    object_points_homogeneous = np.hstack(
        (object_points[i], np.ones((object_points[i].shape[0], 1))), dtype=np.float32
    )
    projected_points, _ = cv.projectPoints(
        object_points_homogeneous,
        np.float32(rotation_vector[i]),
        np.float32(translation_vector[i]),
        cameraMatrix,
        dist,
    )
    error = cv.norm(img_points[i], projected_points, cv.NORM_L2) / len(projected_points)
    reprojection_errors.append(error)
mean_reprojection_error = np.mean(reprojection_errors)
print("Mean Reprojection Error:", mean_reprojection_error)
"""