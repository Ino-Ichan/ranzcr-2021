import torch
import torch.nn as nn
from torchvision import transforms, models

import timm

class Net(nn.Module):
    def __init__(self, name="resnest101e"):
        super(Net, self).__init__()
        self.model = timm.create_model(name, pretrained=True, num_classes=5)

    def forward(self, x):
        x = self.model(x).squeeze(-1)
        return x


class Effnet(nn.Module):
    def __init__(self, name="tf_efficientnet_b6_ns"):
        super(Effnet, self).__init__()
        self.model = timm.create_model(name, pretrained=True, num_classes=5)

    def forward(self, x):
        x = self.model(x).squeeze(-1)
        return x


# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/regnet.py
class RegNet(nn.Module):
    def __init__(self, name="regnety_120"):
        super(RegNet, self).__init__()
        self.model = timm.create_model(name, pretrained=True, num_classes=5)

    def forward(self, x):
        x = self.model(x).squeeze(-1)
        return x