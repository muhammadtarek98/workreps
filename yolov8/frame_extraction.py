import cv2
import os
video=cv2.VideoCapture(r"D:\graval detection project\UV light\videos_3\video_20231031_121233.mp4")
current_frame=828
while True:
    ret,frame=video.read()
    if ret:
        directory=r"D:\graval detection project\UV light\frames_3"
        name=os.path.join(directory,str(current_frame)+'.png')
        print(name)
        cv2.imwrite(name,frame)
        current_frame+=1
    else:
        break
video.release()
cv2.destroyAllWindows()
print(current_frame)
