import torch
import torchinfo
from TNets import conv_forward_block
from Feature_Extractor import BackBone


class PointNetSegmentation(torch.nn.Module):
    def __init__(self, num_points: int, dim: int, num_global_feature: int, labels: int):
        super(PointNetSegmentation, self).__init__()
        self.num_points = num_points
        self.dim = dim
        self.num_global_feature = num_global_feature
        self.total_num_features = self.num_global_feature + 64
        self.labels = labels

        self.back_bone = BackBone(num_points=self.num_points, dim=self.dim, num_global_feature=self.num_global_feature,
                                  segmentation_flag=True)
        self.conv1 = torch.nn.Conv1d(in_channels=self.total_num_features, out_channels=512, kernel_size=1)
        self.conv2 = torch.nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)
        self.conv3 = torch.nn.Conv1d(in_channels=256, out_channels=128, kernel_size=1)
        self.conv4 = torch.nn.Conv1d(in_channels=128, out_channels=labels, kernel_size=1)

        self.bn1 = torch.nn.BatchNorm1d(512)
        self.bn2 = torch.nn.BatchNorm1d(256)
        self.bn3 = torch.nn.BatchNorm1d(128)

    def forward(self, x: torch.Tensor):
        global_and_local_feature, indices, t2 = self.back_bone(x)

        x = conv_forward_block(global_and_local_feature, conv=self.conv1, bn=self.bn1)
        x = conv_forward_block(x, conv=self.conv2, bn=self.bn2)
        x = conv_forward_block(x, conv=self.conv3, bn=self.bn3)
        x = self.conv4(x)
        x = x.transpose(2, 1)
        return x, indices, t2


"""segmentation_model = PointNetSegmentation(num_points=2500, num_global_feature=1024, dim=3, labels=2)
torchinfo.summary(model=segmentation_model)"""
