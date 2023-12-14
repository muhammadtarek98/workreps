from detection import utils
import torchvision
import torch
from dataset_v2 import CustomDataset,get_transform
from masked_rcnn import get_model_instance_segmentation
from detection.engine import train_one_epoch, evaluate

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root_dir = r"D:\graval detection project\datasets\unperpared data\images under water\DEV_000F3102E45A_22_October_2023_10_44_29_jpg"
    transforms = get_transform(train=True)
    dataset = CustomDataset(
        root=root_dir,
        transforms=transforms)
    model=get_model_instance_segmentation(num_classes=2).to(device=device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=1e-2, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=5,
                                                   gamma=0.1)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=2,
        collate_fn=utils.collate_fn
    )
    num_epochs = 10
    for epoch in range(num_epochs):
        run = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=100)
        lr_scheduler.step()
        evaluate(model, data_loader, device=device)
    """
    # For Training
    images, targets = next(iter(data_loader))
    images = list(image for image in images)
    targets = [{k: v for k, v in t.items()} for t in targets]
    output = model(images, targets)  # Returns losses and detections
    print(output)

    # For inference
    #model.eval()
    #x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    #predictions = model(x)  # Returns predictions
    #print(predictions[0])
    """
