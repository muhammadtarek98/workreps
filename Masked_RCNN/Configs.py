from torchvision.transforms import v2 as T
import torch
from Dataset import CustomDataset
from detection import utils


def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float32, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)


train_dir = "/home/cplus/projects/m.tarek_master/gravel_2D/graval_detection_project/datasets/under_water_masks_dataset/train"
valid_dir = "/home/cplus/projects/m.tarek_master/gravel_2D/graval_detection_project/datasets/under_water_masks_dataset/val"
test_dir = "/home/cplus/projects/m.tarek_master/gravel_2D/graval_detection_project/datasets/under_water_masks_dataset/test"
train_transforms = get_transform(train=True)
valid_transforms = get_transform(train=False)
test_transforms = get_transform(train=False)
lr = 1e-4
num_epochs = 10
num_classes = 2
batch_size = 2
training_dataset = CustomDataset(
    root=train_dir,
    transforms=train_transforms)
valid_dataset = CustomDataset(
    root=valid_dir,
    transforms=valid_transforms)
test_dataset = CustomDataset(root=test_dir,
                             transforms=test_transforms)
device = torch.device("cpu")

training_data_loader = torch.utils.data.DataLoader(
    training_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    collate_fn=utils.collate_fn
)
valid_data_loader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    collate_fn=utils.collate_fn
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def optimizer_schedular_configs(model):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.99), weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=5,
                                                   gamma=0.1)
    return optimizer, lr_scheduler
