import torch
import torchinfo
from Configs import *



def conv_forward_block(x, conv, bn):
    x = conv(x)
    x = torch.nn.functional.relu(x)
    x = bn(x)
    return x


def fc_forward_block(x, fc, bn):
    x = fc(x)
    x = torch.nn.functional.relu(x)
    x = bn(x)
    return x


class Tnet(torch.nn.Module):
    def __init__(self, dim: int, num_points: int):
        super(Tnet, self).__init__()
        self.dim = dim
        self.num_points = num_points
        self.conv1 = torch.nn.Conv1d(in_channels=self.dim, out_channels=64, kernel_size=1)
        self.conv2 = torch.nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1)
        self.conv3 = torch.nn.Conv1d(in_channels=128, out_channels=1024, kernel_size=1)

        self.fc1 = torch.nn.Linear(in_features=1024, out_features=512)
        self.fc2 = torch.nn.Linear(in_features=512, out_features=256)
        self.fc3 = torch.nn.Linear(in_features=256, out_features=dim ** 2)

        self.bn2 = torch.nn.BatchNorm1d(num_features=128)
        self.bn1 = torch.nn.BatchNorm1d(num_features=64)

        self.bn3 = torch.nn.BatchNorm1d(num_features=1024)
        self.max_pool = torch.nn.MaxPool1d(kernel_size=self.num_points)
        self.bn4 = torch.nn.BatchNorm1d(num_features=512)
        self.bn5 = torch.nn.BatchNorm1d(num_features=256)

    def forward(self, x: torch.Tensor):
        x = conv_forward_block(x, conv=self.conv1, bn=self.bn1)
        x = conv_forward_block(x, conv=self.conv2, bn=self.bn2)
        x = conv_forward_block(x, conv=self.conv3, bn=self.bn3)
        x = self.max_pool(x).view(batch_size, -1)
        x = fc_forward_block(x, fc=self.fc1, bn=self.bn4)
        x = fc_forward_block(x, fc=self.fc2, bn=self.bn5)
        x = self.fc3(x)

        identity = torch.eye(n=self.dim, requires_grad=True).repeat(batch_size, 1, 1)
        if torch.cuda.is_available():
            identity.cuda()

        x = x.view(-1, self.dim, self.dim) + identity

        return x

"""
model = Tnet(dim=3, num_points=2500)
torchinfo.summary(model=model)"""
