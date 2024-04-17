import torchinfo

from TNets import Tnet, conv_forward_block
import torch

from Configs import *
class BackBone(torch.nn.Module):
    def __init__(self, num_points: int, dim: int, num_global_feature: int, segmentation_flag: bool = True):
        super(BackBone, self).__init__()
        self.num_points = num_points
        self.num_global_feature = num_global_feature
        self.dim = dim
        self.segmentation_flag = segmentation_flag
        """
        input transformation
        """
        self.input_transformation = Tnet(dim=self.dim, num_points=self.num_points)
        """
        the first feature extraction part ===> MLP {fc1:(in=3,out=64),fc2:(in=64,out=64)}
        """
        self.conv1 = torch.nn.Conv1d(in_channels=3, out_channels=64, kernel_size=1)
        self.bn1 = torch.nn.BatchNorm1d(num_features=64)

        self.conv2 = torch.nn.Conv1d(64, 64, kernel_size=1)
        self.bn2 = torch.nn.BatchNorm1d(num_features=64)

        """
        feature transformation
        """
        self.feature_transformation = Tnet(dim=64, num_points=self.num_points)

        """
        the second feature extractor part ====> MLP {fc1:(in=64,out=64),fc2:(in=64,out=128),fc2:(in=128,out=1024)}
        """
        self.conv3 = torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1)
        self.bn3 = torch.nn.BatchNorm1d(num_features=64)

        self.conv4 = torch.nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1)
        self.bn4 = torch.nn.BatchNorm1d(num_features=128)

        self.conv5 = torch.nn.Conv1d(in_channels=128, out_channels=self.num_global_feature, kernel_size=1)
        self.bn5 = torch.nn.BatchNorm1d(num_features=self.num_global_feature)
        self.max_pool = torch.nn.MaxPool1d(kernel_size=num_points, return_indices=True)

    def forward(self, x: torch.Tensor):
        t1 = self.input_transformation(x)
        print(t1.shape)
        print(x.shape)
        x = torch.bmm(x.transpose(dim0=2, dim1=1), t1).transpose(dim0=2, dim1=1)
        print(x.shape)
        x = conv_forward_block(x, conv=self.conv1, bn=self.bn1)
        print(x.shape)
        x = conv_forward_block(x, conv=self.conv2, bn=self.bn2)

        t2 = self.feature_transformation(x)
        x = torch.bmm(x.transpose(dim0=2, dim1=1), t2).transpose(dim0=2, dim1=1)
        local_features = x.clone()

        x = conv_forward_block(x, conv=self.conv3, bn=self.bn3)
        x = conv_forward_block(x, conv=self.conv4, bn=self.bn4)
        x = conv_forward_block(x, conv=self.conv5, bn=self.bn5)

        global_features, indices = self.max_pool(x)
        global_features = global_features.reshape(batch_size, -1)
        indices = indices.reshape(batch_size, -1)

        if self.segmentation_flag:
            global_and_local_feature = torch.cat(tensors=(
                local_features, global_features.unsqueeze(-1).repeat(1, 1, self.num_points)
            ), dim=1)
            return global_and_local_feature, indices, t2
        else:
            return global_features, indices, t2

"""
feature_extractor = BackBone(segmentation_flag=False, num_points=2500, num_global_feature=1024, dim=3)
torchinfo.summary(model=feature_extractor)"""
