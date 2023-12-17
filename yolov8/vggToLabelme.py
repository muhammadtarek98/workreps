import json
import math
import cv2
import shutil

def ellipse_points(cx, cy, rx, ry, phi, num_points):
    # Initialize lists to store x and y coordinates
    shapes = []

    # Calculate points on the oriented ellipse
    for i in range(num_points):
        theta = 2 * math.pi * i / num_points
        x = cx + rx * math.cos(theta) * math.cos(phi) - ry * math.sin(theta) * math.sin(phi)
        y = cy + rx * math.cos(theta) * math.sin(phi) + ry * math.sin(theta) * math.cos(phi)
        shapes.append([x, y])

    return shapes

def create_holes_points_dict(holes_points):
    holes = []
    for hole in holes_points:
        holes.append({
                "label": "hole",
                "points": hole,
                "group_id": None,
                "description": "",
                "shape_type": "polygon",
                "flags": {}
        })

    return holes

def write_json_file(filename, holes):
    # Data to be written
    dictionary = {
        "version": "5.3.1",
        "flags": {},
        "shapes": holes,
        "imagePath": filename,
        "imageData": cv2.imread("../images/"+filename),
        "imageHeight": 1080,
        "imageWidth": 1920
    }   
    # Serializing json
    json_object = json.dumps(dictionary, indent=4)
    
    file = "../images" + filename[:-4] + ".json"
    with open(file, "w") as outfile:
        outfile.write(json_object)
    outfile.close()

def convertlabelme(jsonfile):
    file = open(jsonfile)
    data = json.load(file)

    for img in data:
        holes_points = []
        for hole in data[img]['regions']:
            attributes = hole['shape_attributes']
            points = ellipse_points(attributes['cx'], attributes['cy'], attributes['rx'], attributes['ry'], attributes['theta'], num_points=360)
            holes_points.append(points)

        holes_points_dict = create_holes_points_dict(holes_points) 
        write_json_file(data[img]['filename'], holes_points_dict)

    file.close()

def convertToYoloDataset(jsonfile):
    file = open(jsonfile)
    data = json.load(file)

    images_num = len(data)
    train_num = int(images_num * 0.85)

    i = 0
    folder_dist = "train/"
    for img in data:
        img_name = data[img]['filename']
        image = cv2.imread("../images/" + img_name)
        img_h, img_w, _ = image.shape
        # cv2.imwrite("../images/YOLODataset/images/" + folder_dist + img_name, image)
        shutil.copy("../images/" + img_name, "../images/YOLODataset/images/" + folder_dist + img_name)

        filename = "../images/YOLODataset/labels/" + folder_dist + img_name[:-4] + ".txt" 
        with open(filename, 'w') as f:
            for hole in data[img]['regions']:
                attributes = hole['shape_attributes']
                points = ellipse_points(attributes['cx'], attributes['cy'], attributes['rx'], attributes['ry'], attributes['theta'], num_points=1000)
                f.write(str(0) + " ")
                for point in points:
                    x = round(point[0] / img_w, 6)
                    y = round(point[1] / img_h, 6)
                    f.write(str(x) + " " + str(y) + " ")
                f.write("\n")
        i+=1
        if(i==train_num):
            folder_dist = "val/"
    print(jsonfile + " done!")
    file.close()

if __name__ == "__main__":
    jsonfiles = [
        "simulated.json",
        "via_export_json (1).json",
        "deeptech_abdulrahman.json",
        "F0-to-20_json.json",
        "F21_to_25_json.json",
        "F26_to_40_json.json",
        "F41_to_60_json.json",
        "F63_to_231_json.json",
        "annotations_json.json"   
    ]
    # convertlabelme(jsonfile)
    for f in jsonfiles:
        convertToYoloDataset(f)