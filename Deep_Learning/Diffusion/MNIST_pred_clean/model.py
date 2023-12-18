
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from tqdm.auto import trange
from utils import showExamples

class DoubleConv(nn.Module):
    def __init__(self, in_channels=128, size=32):
        super(DoubleConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
        self.dense = nn.Linear(192, 128)
        self.conv2 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
        self.ln = nn.LayerNorm([128, size, size])

    def forward(self, img, t):

        x_param = F.relu(self.conv1(img))
        t = F.relu(self.dense(t)).view(-1, 128, 1, 1)
        x_param = x_param * t

        out = self.conv2(img) + x_param
        out = F.relu(self.ln(out))
        return out

class UNet(nn.Module):
    def __init__(self, sizes=[32, 16, 8, 4]):
        super().__init__()

        self.time_embed = nn.Sequential(
            nn.Linear(1, 192),
            nn.LayerNorm([192]),
            nn.ReLU(),
        )

        self.device= torch.device('cpu')

        self.encoder = nn.ModuleList()
        for i, size in enumerate(sizes):
            in_c = 1 if i == 0 else 128
            self.encoder.append(DoubleConv(in_c, size))

        self.mlp = nn.Sequential(
            nn.Linear(2240, 128),
            nn.LayerNorm([128]),
            nn.ReLU(inplace=True),

            nn.Linear(128, 32 * 4 * 4),
            nn.LayerNorm([32 * 4 * 4]),
            nn.ReLU(inplace=True)
        )

        self.decoder = nn.ModuleList()
        for i, size in reversed(list(enumerate(sizes))):
            in_c = 32 + 128 if i == len(sizes)-1 else 128 + 128
            self.decoder.append(DoubleConv(in_c, size))

        self.conv_out = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, padding=0)
    
    def to(self, device):
        self.device = device
        return super().to(device)

    def forward(self, x, t):
        t = self.time_embed(t.float())

        # Downward path of the UNet
        skip_connections = []
        for i, module in enumerate(self.encoder):
            x = module(x, t)
            skip_connections.append(x)
            if i < len(self.encoder) - 1:
                x = F.max_pool2d(x, 2)
        
        # Bottleneck
        x = x.view(-1, 128 * 4 * 4)
        x = torch.cat([x, t], dim=1)
        x = self.mlp(x)
        x = x.view(-1, 32, 4, 4)

        skip_connections = skip_connections[::-1] # Reverse list

        # Upward path of the UNet
        for i, module in enumerate(self.decoder):
            skip = skip_connections[i]
            # x = F.interpolate(x, scale_factor=2, mode='nearest')
            x = torch.cat([x, skip], dim=1)
            x = module(x, t)

            if i < len(self.decoder) - 1:
                x = F.interpolate(x, scale_factor=2, mode='nearest')
        
        x = self.conv_out(x)
    
        return x

    #  generates one example image, and returns x_t for all timesteps
    def sample_steps(self, timesteps=16):
        xs = []
        x = torch.randn((1, 1, 32, 32)).to(self.device)

        with torch.no_grad():
            for t in trange(timesteps):
                x = self.forward(x, torch.full([1, 1], t, dtype=torch.float32, device=self.device))
        
        showExamples(xs)
    
    # generates 16 examples, and returns the final x_t for each
    def sample(self, timesteps=16):
        x = torch.randn((16, 1, 32, 32)).to(self.device)

        with torch.no_grad():
            for t in trange(timesteps):
                x = self.forward(x, torch.full([16, 1], t, dtype=torch.float32, device=self.device))

        showExamples(x)
        