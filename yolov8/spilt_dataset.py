import os
import shutil
def copy_images(set,dest):
    for dir in set:
        shutil.copy(dir,dest)
train_val=0.7
valid_val=0.2
test_val=0.1
main_dir=r"D:\graval detection project\resyris\resyris\test_data_realworld\left_rgb"
dirs=[os.path.join(main_dir,dir) for dir in os.listdir(main_dir) if dir.endswith('.png')]
dataset_size=len(dirs)

train_size=int(dataset_size*train_val)
valid_size=int(dataset_size*valid_val)
test_size=int(dataset_size*test_val)
train_set=dirs[:train_size]
val_last_idx=train_size+valid_size
valid_set=dirs[train_size:val_last_idx]
test_set=dirs[val_last_idx:]
copy_images(set=train_set,dest=r"D:\graval detection project\resyris\resyris\test_data_realworld\dataset\train\images")
copy_images(set=valid_set,dest=r"D:\graval detection project\resyris\resyris\test_data_realworld\dataset\val\images")
copy_images(set=test_set,dest=r"D:\graval detection project\resyris\resyris\test_data_realworld\dataset\test\images")

#copy_images(set=train_set,dest=r"D:\graval detection project\resyris\resyris\test_data_realworld\dataset\train\labels")
#copy_images(set=valid_set,dest=r"D:\graval detection project\resyris\resyris\test_data_realworld\dataset\val\labels")
#copy_images(set=test_set,dest=r"D:\graval detection project\resyris\resyris\test_data_realworld\dataset\test\labels")
"""
print(len(train_set)==train_size," ",len(train_set))
print(len(valid_set)==valid_size," ",len(valid_set))
print(len(test_set)==test_size," ",len(test_set))
print(len(test_set)+len(valid_set)+len(train_set)==len(dirs))"""
