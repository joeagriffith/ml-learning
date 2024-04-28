import torch
import torch.nn as nn
from torchvision.models import resnet18, alexnet


class LAugPC(nn.Module):
    def __init__(self, in_features, num_actions, backbone='resnet18'):
        super().__init__()
        self.in_features = in_features
        self.num_actions = num_actions
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

        self.action_encoder = nn.Sequential(
            nn.Linear(num_actions, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        self.transition = nn.Sequential(
            nn.Linear(self.num_features + 128, 4096, bias=True),
            nn.ReLU(),
            nn.Linear(4096, 2048, bias=True),
            nn.ReLU(),
            nn.Linear(2048, self.num_features, bias=True),
        )
        
    
    def forward(self, x):
        z = self.encoder(x)
        return z
    
    def predict(self, x, a=None):
        if a is None:
            a = torch.zeros(x.shape[0], self.num_actions).to(x.device)
        
        z = self.encoder(x)
        a = self.action_encoder(a)
        z_pred = self.transition(torch.cat([z, a], dim=1))
        return z_pred
    
    def copy(self):
        model = LAugPC(self.in_features, self.num_actions, self.backbone).to(next(self.parameters()).device)
        model.load_state_dict(self.state_dict())
        return model