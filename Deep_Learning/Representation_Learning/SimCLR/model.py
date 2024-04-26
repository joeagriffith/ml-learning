import torch
import torch.nn as nn
from torchvision.models import resnet18

class Model(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.encoder = resnet18()
        self.encoder.conv1 = nn.Conv2d(in_features, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.encoder.maxpool = nn.Identity()
        self.encoder.fc = nn.Identity()

        self.project = nn.Sequential(
            nn.Linear(512, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 256, bias=False),
            nn.BatchNorm1d(256),
        )

    def forward(self, x):
        return self.encoder(x)

