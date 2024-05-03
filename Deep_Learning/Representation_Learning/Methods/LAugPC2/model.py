import torch
import torch.nn as nn

from torchvision.models import resnet18, alexnet

class LAugPC2(nn.Module):
    def __init__(self, in_features, num_actions, backbone='resnet18'):
        super().__init__()
        self.in_features = in_features
        self.num_actions = num_actions
        self.backbone = backbone

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

        # self.generate = nn.Sequential(
        #     nn.Unflatten(1, (self.num_features, 1, 1)),
        #     nn.ConvTranspose2d(256, 256, 2),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(256, 256, 3),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(256, 384, 3),
        #     # nn.ReLU(),
        #     # nn.ConvTranspose2d(256, 192, 3),
        # )
        self.gen_nets = nn.ModuleList([
            nn.Sequential(
                nn.Unflatten(1, (self.num_features, 1, 1)),
                nn.ConvTranspose2d(256, 256, 2),
                nn.ReLU(),
                nn.Conv2d(256, 256, 3, padding=1),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(256, 256, 3, stride=3),
                nn.ReLU(),
                nn.Conv2d(256, 192, 3, padding=1)
            ),
            nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.ConvTranspose2d(192, 192, 2),
                nn.ReLU(),
                nn.Conv2d(192,64, 3, padding=1)
            ),
            nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.ConvTranspose2d(64, 64, 3),
                nn.ReLU(),
                nn.Conv2d(64,1, 3, padding=1)
            )
        ])


    
    def generate(self, z):
        preds = []
        for net in self.gen_nets:
            z = net(z)
            preds.append(z)
        return preds

    def get_targets(self, x):
        assert self.backbone == 'alexnet', 'get_targets only implemented for alexnet'
        xs = [x]
        for module in self.encoder.features:
            x = module(x)
            if isinstance(module, nn.MaxPool2d):
                xs.append(x)
        xs.append(self.encoder.avgpool(x))
        return xs

    def forward(self, x):
        z = self.encoder(x)
        return z
    
    def predict(self, x, a=None):
        if a is None:
            a = torch.zeros(x.shape[0], self.num_actions, device=x.device)
        
        z = self.encoder(x)
        a = self.action_encoder(a)
        z_pred = self.transition(torch.cat([z, a], dim=1))
        preds = self.generate(z_pred)
        return preds
    
    def copy(self):
        model = LAugPC2(self.in_features, self.num_actions, self.backbone)
        model.load_state_dict(self.state_dict())
        return model