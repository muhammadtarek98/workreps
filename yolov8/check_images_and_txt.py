import os
img_dir=[i for i in os.listdir(r"D:\graval detection project\datasets\under water dataset\images\train")]
txt_dir=[i for i in os.listdir(r"D:\graval detection project\datasets\under water dataset\labels\train")]
for i,j in zip(img_dir,txt_dir):
    img_name,_=i.split('.')
    txt_name,_=j.split('.')
    if img_name!=txt_name:
        print(img_name)
    else:
        print("pass")

