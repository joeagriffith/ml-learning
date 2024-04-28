import torch
import torch.nn as nn
from torchvision.models import resnet18, alexnet

class BYOL(nn.Module):
    def __init__(self, in_features, backbone='alexnet'):
        super().__init__()
        self.in_features = in_features
        self.backbone = backbone

        if backbone == 'resnet18':
            self.encoder = resnet18()
            self.encoder.conv1 = nn.Conv2d(in_features, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            self.encoder.maxpool = nn.Identity()
            self.encoder.fc = nn.Flatten()
            self.num_features = 512
        elif backbone == 'alexnet':
            self.encoder = alexnet()
            self.encoder.features[0] = nn.Conv2d(in_features, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.encoder.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.encoder.classifier = nn.Flatten()
            self.num_features = 256
        

        self.project = nn.Sequential(
            nn.Linear(self.num_features, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 256, bias=False),
        )

        self.predict = nn.Sequential(
            nn.Linear(256, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256, bias=False),
        )

    def forward(self, x):
        return self.encoder(x)
    
    def copy(self):
        model = BYOL(self.in_features, backbone=self.backbone).to(next(self.parameters()).device)
        model.load_state_dict(self.state_dict())
        return model
