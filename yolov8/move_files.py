import os
import shutil

l = [ ]
for i in os.listdir(r"D:\graval detection project\datasets\unperpared data\images under water\images"):
    l.append(i)
for i in os.listdir(r"D:\graval detection project\datasets\unperpared data\images under water\masks"):
    if i not in l:
        print(i)

# images=[i for i in os.listdir(r"D:\graval detection project\datasets\unperpared data\images under water\images")]
# labels=[i for i in os.listdir(r"D:\graval detection project\datasets\unperpared data\images under water\labels in txt")]
# txt_file_name=[]
# images_names=[]
# for i in labels:
#    f,_=i.split(".")
#    txt_file_name.append(f)
# for i in images:
#    f,_=i.split(".")
#    images_names.append(f)

# dp=dict()
# images_dir=r"D:\graval detection project\datasets\unperpared data\images under water\images_without_annotations"
# l=[]
# for i in os.listdir(images_dir):
#    dp[i]=[]
# json_dir=r"D:\graval detection project\datasets\unperpared data\images under water\finished_reannotated"
# for i in os.listdir(json_dir):
#    file,_=i.split(".")
# print(file)
#    for j in dp:
#        img,_=j.split(".")
#        if file == img:
#            print(f"copy file {i}")
#            shutil.move(os.path.join(images_dir,j),os.path.join(json_dir,j))

# print(l[88])
