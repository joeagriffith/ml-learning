import torch
import torch.nn as nn

class EncBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, pool=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if pool else nn.Identity()
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class DecBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, upsample=False):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.upsample = nn.Upsample(scale_factor=2) if upsample else nn.Identity()
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.upsample(x)
        x = self.relu(x)
        return x


class HAugPC(nn.Module):
    def __init__(self, in_features, num_actions):
        super().__init__()
        self.in_features = in_features
        self.num_actions = num_actions
        self.backbone = 'mnist_cnn'

        self.enc_blocks = nn.ModuleList([
            EncBlock(1, 32, 3, 1, 1, pool=True),
            EncBlock(32, 64, 3, 1, 1, pool=True),
            EncBlock(64, 128, 3, 1, 0),
            EncBlock(128, 256, 3, 1, 0),
            EncBlock(256, 256, 3, 1, 0),
        ])
        self.num_features = 256
    
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

        self.dec_blocks = nn.ModuleList([
            DecBlock(self.num_features, 256, 3, 1, 0),
            DecBlock(256, 128, 3, 1, 0),
            DecBlock(128, 64, 3, 1, 0),
            DecBlock(64, 32, 3, 1, 1, upsample=True),
            DecBlock(32, 1, 3, 1, 1, upsample=True),
        ])

    def forward(self, x, stop_at=None):
        for i, block in enumerate(self.enc_blocks):
            if i == stop_at:
                break
            x = block(x)
        if stop_at is not None:
            return x
        else:
            return x.flatten(1)
        
    def generate(self, z, stop_at=None):
        z = z.view(-1, 256, 1, 1)
        for i, block in enumerate(self.dec_blocks):
            if stop_at is not None and i == len(self.dec_blocks) - stop_at:
                break
            z = block(z)
        return z
    
    def predict(self, x, a=None, stop_at=None):
        if a is None:
            a = torch.zeros(x.shape[0], self.num_actions, device=x.device)
        
        z = self(x)
        a = self.action_encoder(a)
        z_pred = self.transition(torch.cat([z, a], dim=1))
        pred = self.generate(z_pred, stop_at)
        return pred