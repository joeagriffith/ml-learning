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

        self.predict = nn.Sequential(
            nn.Linear(1000 + 128, 1024, bias=False),
            nn.ReLU(),
            nn.Linear(1024, 1000, bias=False),
        )
    
    def forward(self, x):
        z = self.encoder(x)
        return z
    
    def predict_z(self, z, a):
        a = self.action_encoder(a)
        z_pred = self.predict(torch.cat([z, a], dim=1))
        return z_pred
    
    def copy(self):
        model = Model(self.in_features, self.num_actions).to(next(self.parameters()).device)
        model.load_state_dict(self.state_dict())
        return model