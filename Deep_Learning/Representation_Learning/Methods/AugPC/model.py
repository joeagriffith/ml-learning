import torch
import torch.nn as nn

from torchvision.models import resnet18, alexnet
from rvit import RegisteredVisionTransformer
from Deep_Learning.Representation_Learning.Utils.nets import mnist_cnn_encoder, mnist_cnn_decoder

class AugPC(nn.Module):
    def __init__(self, in_features, num_actions, backbone='mnist_cnn'):
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
            # self.encoder.fc = nn.Flatten()
            self.encoder.fc = nn.Linear(512, 256)
            self.num_features = 256

        elif backbone == 'alexnet':
            self.encoder = alexnet()
            self.encoder.features[0] = nn.Conv2d(in_features, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.encoder.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.encoder.classifier = nn.Flatten()
            self.num_features = 256

        elif backbone == 'mnist_cnn':
            self.num_features = 256
            self.encoder = mnist_cnn_encoder(self.num_features)
    
        self.action_encoder = nn.Sequential(
            nn.Linear(num_actions, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        # NO BATCHNORM
        self.transition = nn.Sequential(
            nn.Linear(self.num_features + 128, 1024, bias=False),
            nn.ReLU(),
            nn.Linear(1024, 512, bias=False),
            nn.ReLU(),
            nn.Linear(512, self.num_features, bias=False)
        )

        #for Mnist (-1, 1, 28, 28)
        self.decoder = mnist_cnn_decoder(self.num_features)

    def forward(self, x, stop_at=None):
        if self.backbone == 'mnist_cnn':
            z = self.encoder(x, stop_at)
        else:
            z = self.encoder(x)
        return z
    
    def predict(self, x, a=None, stop_at=None):
        if a is None:
            a = torch.zeros(x.shape[0], self.num_actions, device=x.device)
        
        z = self.encoder(x)
        a = self.action_encoder(a)
        z_pred = self.transition(torch.cat([z, a], dim=1))
        pred = self.decoder(z_pred, stop_at)
        return pred
    
    def copy(self):
        model = AugPC(self.in_features, self.num_actions, self.backbone).to(next(self.parameters()).device)
        model.load_state_dict(self.state_dict())
        return model