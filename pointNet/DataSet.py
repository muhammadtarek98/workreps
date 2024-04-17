import torch
import torch_geometric
import open3d
import torchvision
import os


class CustomDataSet(torch.utils.data.Dataset):
    def __init__(self, root_dir: str):
        super(CustomDataSet, self).__init__()
        dir=os.path.join(root_dir,)

    def __len__(self):
        return len(self.plys)

    def __getitem__(self, idx):
        pass
