# model/pointnet.py
import torch.nn as nn
import torch.nn.functional as F

class PointNetFeat(nn.Module):
    def __init__(self, normal_channel=True):
        super(PointNetFeat, self).__init__()
        channel = 6 if normal_channel else 3
        self.conv1 = nn.Conv1d(channel, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)
        self.mp1 = nn.MaxPool1d(1024)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.mp1(x)
        return x

class PointNetCls(nn.Module):
    def __init__(self, num_classes=40, normal_channel=True):
        super(PointNetCls, self).__init__()
        self.normal_channel = normal_channel
        self.feat = PointNetFeat(normal_channel=self.normal_channel)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.feat(x)
        x = x.view(-1, 1024)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        x = F.log_softmax(x, -1)
        return x
