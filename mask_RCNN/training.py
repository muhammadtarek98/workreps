from masked_rcnn import get_model
from dataset import CustomDataSet,get_transform
import torch.utils.data
from detection.engine import train_one_epoch,evaluate

model=get_model(2)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.0005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=5,
                                               gamma=0.1)

num_epochs = 3
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
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_loader = torch.utils.data.DataLoader(dataset, batch_size=1,shuffle=True)
for epoch in range(num_epochs):
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=100)
    lr_scheduler.step()
    res=evaluate(model=model,device=device)
