import os
from yolov8_segmentation import predict
import json

#weight_path=r"D:\graval detection project\mareim runs\segment\train_UW_1080\weights\best.pt"
images_path = r"D:\graval detection project\datasets\unperpared data\images under water\images"
jsons_path = r"D:\graval detection project\datasets\unperpared data\images under water\annotations"


"""
rest = []
images = [
    i for i in os.listdir(images_path)
]
images.sort()
jsons = [i for i in os.listdir(jsons_path)]
jsons.sort()
x = []
for i in images:
    name = i.split(".")
    x.append([name[0], name[1]])
y = []
for i in jsons:
    name = i.split('.')
    y.append(name[0])
r=0
not_annotated=[]
#print(len(x)-len(y))
for i in x:
    if i[0] in y:
        continue
    else:
        not_annotated.append(i[0]+"."+i[1])
print(len(not_annotated))
annotated_image=[]
for i in not_annotated:
    annotation_data = {
        "version": "5.3.1",
        "flags": {},
        "shapes": [],
        "imageData": None,  # Base64-encoded image data (optional)
        "imageHeight": 1100,
        "imageWidth": 1376,
    }
    results=predict(weights_path=weight_path,images_path=os.path.join(images_path,i),confidence_threshold=0.7,saving_flag=False)
    for j, result in enumerate(results):
        masks = result.masks
        try:
            for k in range(len(masks)):
                polygon_list = masks.xy[k].tolist()
                #print(polygon_list)
                annotation_data["shapes"].append({
                "label": "stone",
                "points": polygon_list,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}})

            annotation_data["imagePath"]=os.path.join(images_path,i)
            annotated_image.append(i)
            t=i.replace("D:\\graval detection project\\datasets\\unperpared data\\images under water\\images\\","")
            c,e=t.split(".jpg" or ".png")
            print(c)
            file=c+".json"
            if file not in os.listdir(jsons_path):
                with open(os.path.join(jsons_path,file), "w") as json_file:
                    json.dump(annotation_data, json_file, indent=2)
        except:
            print(f"error in file {i}")
            not_annotated.append(i)


#for i in x[len(y)+1:]:
#    rest.append(i)
#full_images_dir=[os.path.join(images_path,i[0]+"."+i[1]) for i in rest]
#img_name=i[0]+"."+i[1]
#img_dir=os.path.join(images_path,img_name)
for i in full_images_dir:
    annotation_data = {
        "version": "5.3.1",
        "flags": {},
        "shapes": [],
        "imageData": None,  # Base64-encoded image data (optional)
        "imageHeight": 1100,
        "imageWidth": 1376,
    }
    results=predict(weights_path=weight_path,images_path=i,confidence_threshold=0.7,saving_flag=False)
    for j, result in enumerate(results):
        masks = result.masks
        try:
            for k in range(len(masks)):
                polygon_list = masks.xy[k].tolist()
                #print(polygon_list)
                annotation_data["shapes"].append({
                "label": "stone",
                "points": polygon_list,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}})

            annotation_data["imagePath"]=i
            annotated_image.append(i)
            t=i.replace("D:\\graval detection project\\datasets\\unperpared data\\images under water\\images\\","")
            c,e=t.split(".jpg" or ".png")
            print(c)
            file=c+".json"
            if file not in os.listdir(jsons_path):
                with open(os.path.join(jsons_path,file), "w") as json_file:
                    json.dump(annotation_data, json_file, indent=2)
        except:
            print(f"error in file {i}")
            not_annotated.append(i)
not_annotated.clear()
for i in annotated_image:
    if i not in images:
        not_annotated.append(i)
for i in not_annotated:
        annotation_data = {
            "version": "5.3.1",
            "flags": {},
            "shapes": [],
            "imageData": None,  # Base64-encoded image data (optional)
            "imageHeight": 1100,
            "imageWidth": 1376,
        }
        results = predict(weights_path=weight_path, images_path=i, confidence_threshold=0.7, saving_flag=False)
        for j, result in enumerate(results):
            masks = result.masks
            try:
                for k in range(len(masks)):
                    polygon_list = masks.xy[k].tolist()
                    # print(polygon_list)
                    annotation_data["shapes"].append({
                        "label": "stone",
                        "points": polygon_list,
                        "group_id": None,
                        "shape_type": "polygon",
                        "flags": {}})

                annotation_data["imagePath"] = i
                annotated_image.append(i)
                t = i.replace("D:\\graval detection project\\datasets\\unperpared data\\images under water\\images\\",
                              "")
                c, e = t.split(".jpg" or ".png")
                print(c)
                file = c + ".json"
                if file not in os.listdir(jsons_path):
                    with open(os.path.join(jsons_path, file), "w") as json_file:
                        json.dump(annotation_data, json_file, indent=2)
            except:
                print(f"error in file {i}")
                not_annotated.append(i)"""