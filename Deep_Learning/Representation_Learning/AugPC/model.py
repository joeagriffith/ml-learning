import torch
import torch.nn as nn

from torchvision.models import resnet18

class Model(nn.Module):
    def __init__(self, in_features, num_actions):
        super().__init__()
        self.in_features = in_features
        self.num_actions = num_actions

        self.encoder = resnet18()
        self.encoder.conv1 = nn.Conv2d(in_features, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.encoder.maxpool = nn.Identity()
        # self.encoder.fc = nn.Identity()
    
        self.action_encoder = nn.Sequential(
            nn.Linear(num_actions, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.project = nn.Sequential(
            nn.Linear(1000 + 128, 1024, bias=False),
            nn.ReLU(),
            nn.Linear(1024, 1000, bias=False),
        )

        self.generate = nn.Sequential(
            nn.Unflatten(1, (1000, 1, 1)),
            nn.ConvTranspose2d(1000, 512, 3, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 3, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 1, 2, 1),
        )

    def forward(self, x):
        z = self.encoder(x)
        return z
    
    def predict(self, x, a=None):
        if a is None:
            a = torch.zeros(x.shape[0], self.num_actions, device=x.device)
        
        z = self.encoder(x)
        a = self.action_encoder(a)
        z_pred = self.project(torch.cat([z, a], dim=1))
        pred = self.generate(z_pred)
        return pred