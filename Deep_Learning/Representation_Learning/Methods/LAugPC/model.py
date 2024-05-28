import torch
import torch.nn as nn
from torchvision.models import resnet18, alexnet
from rvit import RegisteredVisionTransformer


class LAugPC(nn.Module):
    def __init__(self, in_features, num_actions, backbone='resnet18'):
        super().__init__()
        self.in_features = in_features
        self.num_actions = num_actions
        self.backbone = backbone

        # MNIST ONLY
        if backbone == 'vit':
            self.encoder = RegisteredVisionTransformer(
                image_size=28,
                patch_size=7,
                num_layers=6,
                num_heads=4,
                hidden_dim=256,
                num_registers=4,
                mlp_dim=1024,
            )
            self.encoder.conv_proj = nn.Conv2d(1, 256, kernel_size=7, stride=7)
            self.encoder.heads = nn.Identity()
            self.num_features = 256
        elif backbone == 'resnet18':
            self.encoder = resnet18()
            self.encoder.conv1 = nn.Conv2d(in_features, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            self.encoder.maxpool = nn.Identity()
            self.encoder.fc = nn.Linear(512, 256)
            self.num_features = 256
        elif backbone == 'alexnet':
            self.encoder = alexnet()
            self.encoder.features[0] = nn.Conv2d(in_features, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.encoder.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.encoder.classifier = nn.Flatten()
            self.num_features = 256
        elif backbone == 'mnist_cnn':
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.BatchNorm2d(32),
                nn.ReLU(),

                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.BatchNorm2d(64),
                nn.ReLU(),

                nn.Conv2d(64, 128, kernel_size=3, stride=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),

                nn.Conv2d(128, 256, kernel_size=3, stride=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),

                nn.Conv2d(256, 256, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
            )
            self.num_features = 256

        self.action_encoder = nn.Sequential(
            nn.Linear(num_actions, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        # # NO BATCHNORM
        # self.transition = nn.Sequential(
        #     nn.Linear(self.num_features + 128, 1024, bias=False),
        #     nn.ReLU(),
        #     nn.Linear(1024, 512, bias=False),
        #     nn.ReLU(),
        #     nn.Linear(512, self.num_features, bias=False)
        # )

        self.project = nn.Sequential(
            nn.Linear(self.num_features, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 256, bias=False),
        )

        self.predictor = nn.Sequential(
            nn.Linear(256+128, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256, bias=False),
        )
        
    def forward(self, x):
        z = self.encoder(x)
        return z
    
    def predict(self, x, a=None):
        if a is None:
            a = torch.zeros(x.shape[0], self.num_actions).to(x.device)
        
        # z = self.encoder(x)
        # a = self.action_encoder(a)
        # z_pred = self.transition(torch.cat([z, a], dim=1))

        z = self.encoder(x)
        z = self.project(z)
        a = self.action_encoder(a)
        z_pred = self.predictor(torch.cat([z, a], dim=1))
        return z_pred
    
    def get_targets(self, x):
        return self.project(self.encoder(x))
    
    def copy(self):
        model = LAugPC(self.in_features, self.num_actions, self.backbone).to(next(self.parameters()).device)
        model.load_state_dict(self.state_dict())
        return model