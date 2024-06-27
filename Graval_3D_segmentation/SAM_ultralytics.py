from ultralytics import SAM
import cv2
from ultralytics.data.annotator import auto_annotate

model=SAM("sam_l.pt")
model.info()
res=model("/home/cplus/projects/m.tarek_master/graval_detection_project/115351AA.mp4_/0_left.jpg")
print(res)

#cv2.imshow(mat=res,winname="test")
#cv2.waitKey(0)
#cv2.destroyAllWindows()