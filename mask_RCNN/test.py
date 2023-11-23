from detection import utils
import torchvision
import torch
from dataset import CustomDataSet,get_transform
transforms = get_transform(train=True)
# torchvision.transforms.Resize(size=(256, 256), antialias=True),  # Resize to a fixed size
# torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
#print(type(transforms))
dataset = CustomDataSet(csv_file_dir=r"D:\mask_RCNN\meta_data.csv",
                        images_dir=r"D:\graval detection project\datasets\unperpared data\images under water\DEV_000F3102E45A_22_October_2023_10_44_29_jpg\images",
                        masks_dir=r"D:\graval detection project\datasets\unperpared data\images under water\DEV_000F3102E45A_22_October_2023_10_44_29_jpg\masks",
                        data_dir=r"D:\graval detection project\datasets\unperpared data\images under water\DEV_000F3102E45A_22_October_2023_10_44_29_jpg",
                        json_dirs=r"D:\graval detection project\datasets\unperpared data\images under water\DEV_000F3102E45A_22_October_2023_10_44_29_jpg\annotations",
                        transforms=transforms)
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
#dataset = PennFudanDataset('data/PennFudanPed', get_transform(train=True))
data_loader = torch.utils.data.DataLoader(dataset, batch_size=1,shuffle=True)

# For Training
images, targets = next(iter(data_loader))
images = list(image for image in images)
targets = [{k: v for k, v in t.items()} for t in targets]
output = model(images, targets)  # Returns losses and detections
print(output)

# For inference
model.eval()
x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
predictions = model(x)  # Returns predictions
print(predictions[0])