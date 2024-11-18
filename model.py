import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class BCE(nn.Module):
    def __init__(self):
        super(CE, self).__init__()
        self.resnet = resnet18(pretrained='ResNet18_Weights.IMAGENET1K_V1')
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.fc1 = nn.Linear(1000, 400)
        self.fc1_drop = nn.Dropout()
        self.fc2 = nn.Linear(400, 84)
        self.fc2_drop = nn.Dropout()
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc1_drop(F.relu(self.fc1(x)))
        x = self.fc2_drop(F.relu(self.fc2(x)))
        x = torch.sigmoid(self.fc3(x))
        return x

    def init_weights(model):  # weight initialization
        w = torch.empty(3, 5)
        nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))
