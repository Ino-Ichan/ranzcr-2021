import torch
import torch.nn as nn
from torchvision import transforms, models

import timm

class Net(nn.Module):
    def __init__(self, name="resnest101e"):
        super(Net, self).__init__()
        self.model = timm.create_model(name, pretrained=True, num_classes=11)

    def forward(self, x):
        x = self.model(x).squeeze(-1)
        return x


class RANZCRResNet200D(nn.Module):
    def __init__(self, model_name='resnet200d', out_dim=11, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=False)
        n_features = self.model.fc.in_features
        self.model.global_pool = nn.Identity()
        self.model.fc = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(n_features, out_dim)

    def forward(self, x):
        bs = x.size(0)
        features = self.model(x)
        pooled_features = self.pooling(features).view(bs, -1)
        output = self.fc(pooled_features)
        return output

class NetCVC(nn.Module):
    def __init__(self, name="resnest101e"):
        super(NetCVC, self).__init__()
        self.model = timm.create_model(name, pretrained=True, num_classes=3)

    def forward(self, x):
        x = self.model(x).squeeze(-1)
        return x