import torch
import torch.nn as nn

from torchvision.models import resnet18, alexnet

class AugPC(nn.Module):
    def __init__(self, in_features, num_actions, backbone='resnet18'):
        super().__init__()
        self.in_features = in_features
        self.num_actions = num_actions

        if backbone == 'resnet18':
            self.encoder = resnet18()
            self.encoder.conv1 = nn.Conv2d(in_features, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            self.encoder.maxpool = nn.Identity()
            self.encoder.fc = nn.Flatten() # Actually performs better without this line
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
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.transition = nn.Sequential(
            nn.Linear(self.num_features + 128, 4096, bias=False),
            nn.ReLU(),
            nn.Linear(4096, 2048, bias=False),
            nn.ReLU(),
            nn.Linear(2048, self.num_features, bias=False)
        )

        #for Mnist (-1, 1, 28, 28)
        # self.generate = nn.Sequential(
        #     nn.Unflatten(1, (self.num_features, 1, 1)),
        #     nn.ConvTranspose2d(self.num_features, 512, 3, 1),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(512, 256, 3, 3),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(256, 128, 3, 3),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(128, 1, 2, 1),
        # )

        # for CIFAR-10 (-1, 3, 32, 32)
        self.generate = nn.Sequential(
            nn.Unflatten(1, (256, 1, 1)),
            nn.ConvTranspose2d(256, 512, 3, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 3, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, 4, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 3, 3, 1),
        )

        # # for STL-10 (-1, 3, 96, 96)
        # self.generate = nn.Sequential(
        #     nn.Unflatten(1, (256, 1, 1)),
        #     nn.ConvTranspose2d(256, 512, 3, 1),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(512, 256, 4, 3),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(256, 128, 4, 3),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(128, 128, 3, 3),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(128, 64, 4, 1),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 3, 3, 1, 1),
        # )


    def forward(self, x):
        z = self.encoder(x)
        return z
    
    def predict(self, x, a=None):
        if a is None:
            a = torch.zeros(x.shape[0], self.num_actions, device=x.device)
        
        z = self.encoder(x)
        a = self.action_encoder(a)
        z_pred = self.transition(torch.cat([z, a], dim=1))
        pred = self.generate(z_pred)
        return pred