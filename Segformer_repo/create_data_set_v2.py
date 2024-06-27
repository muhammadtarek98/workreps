import os
import shutil
import random
import numpy as np


def seed():
    return 0.1


images_root = "/home/cplus/projects/m.tarek_master/gravel_2D/graval_detection_project/datasets/under_water_masks_dataset_v2/images_pooling/"
masks_root = "/home/cplus/projects/m.tarek_master/gravel_2D/graval_detection_project/datasets/under_water_masks_dataset_v2/masks_pooling/"


def copy_images(set, image_dest, masks_dest):
    for batch in set:
        image, mask = batch[0], batch[1]
        print(f"checking {batch} ")
        i = image.replace(images_root, "")
        i = i.replace(".jpg", "")
        m = mask.replace(masks_root, "")
        m = m.replace(".png", "")
        if i == m:
            # move images
            #print(image)
            shutil.copy(image, image_dest)
            #move masks
            #print(mask)
            shutil.copy(mask, masks_dest)
        else:
            print(f"recheck {batch}")


images_root = "/home/cplus/projects/m.tarek_master/gravel_2D/graval_detection_project/datasets/under_water_masks_dataset_v2/images_pooling/"
masks_root = "/home/cplus/projects/m.tarek_master/gravel_2D/graval_detection_project/datasets/under_water_masks_dataset_v2/masks_pooling/"
images_dir = sorted([os.path.join(images_root, i) for i in os.listdir(images_root) if i.endswith(".jpg")])
masks_dir = sorted([os.path.join(masks_root, i) for i in os.listdir(masks_root) if i.endswith(".png")])
#print(images_dir)
#print(masks_dir)
print(len(images_dir) == len(masks_dir))
dataset = []
for i in images_dir:
    temp = i.replace(images_root, "")
    temp = temp.replace(".jpg", "")
    for j in masks_dir:
        temp_2 = j.replace(masks_root, "")
        temp_2 = temp_2.replace(".png", "")
        if temp == temp_2:
            dataset.append((i, j))

#print(dataset)
random.shuffle(dataset, seed)
train_set_size = int(0.6 * len(dataset))
print(train_set_size)
validation_set_size = int(0.2 * len(dataset))
print(validation_set_size)
val_last_idx = train_set_size + validation_set_size
test_set_size = int(0.1 * len(dataset))
test_last_idx = train_set_size + validation_set_size + test_set_size
print(test_set_size)
train_set = dataset[:train_set_size]
print(len(train_set))
validation_set = dataset[train_set_size:val_last_idx]
print(len(validation_set))
test_set = dataset[test_last_idx:]
train_dir = "/home/cplus/projects/m.tarek_master/gravel_2D/graval_detection_project/datasets/under_water_masks_dataset_v2/training/"
validation_dir = "/home/cplus/projects/m.tarek_master/gravel_2D/graval_detection_project/datasets/under_water_masks_dataset_v2/validation/"
test_dir = "/home/cplus/projects/m.tarek_master/gravel_2D/graval_detection_project/datasets/under_water_masks_dataset_v2/testing/"

copy_images(set=train_set,
            image_dest=f"{trainset_dir}images/",
            masks_dest=f"{trainset_dir}masks/")
copy_images(set=validation_set,
            image_dest=f"{validation_dir}images/",
            masks_dest=f"{validation_dir}masks/")
print(testset)

copy_images(set=test_set,
            image_dest=f"{test_dir}images/",
            masks_dest=f"{test_dir}masks/")
