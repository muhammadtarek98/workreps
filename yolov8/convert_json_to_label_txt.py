import json
import os


def normalize_points(points, h, w):
    l = []
    for point in points:
        x, y = point
        x_norm = x / w
        y_norm = y / h
        l.append(x_norm)
        l.append(y_norm)
    return l
main_dir = r"D:\graval detection project\datasets\unperpared data\images under water\ready_to_go\finished_annotations"
output_dir=r"D:\graval detection project\datasets\unperpared data\images under water\ready_to_go\label in txt"
files = [i for i in os.listdir(
    main_dir)]
# print(len(files))

for file in files:
    temp = file.replace('.json', "")

    json_file_dir = os.path.join(main_dir, file)
    keys = ['version', 'flags', 'shapes', 'imagePath', 'imageData', 'imageHeight', 'imageWidth']
    label = 0
    with open(json_file_dir, 'r') as json_file:
        data = json.load(json_file)
    img_width = data[keys[-1]]
    img_height = data[keys[-2]]
    print(len(data['shapes']))
    s = ""
    for obj in data['shapes']:
        points = obj['points']
        result = normalize_points(points, h=img_height, w=img_width)
        s =s+"0 "+str(result) + "\n"
    if f"{temp}.txt" in os.listdir(output_dir):
        continue
    else:
        file_dir=os.path.join(output_dir,f"{temp}.txt")
        print(file_dir)
        with open(file_dir,'a') as the_file:
                the_file.write(s)
