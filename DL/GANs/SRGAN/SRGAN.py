import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels, 
        out_channels, 
        discriminator=False, 
        use_actv=True, 
        use_bn=True,
        **kwargs,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs, bias=not use_bn)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.actv = nn.LeakyReLU(0.2, inplace=True) if discriminator else nn.PReLU(num_parameters=out_channels)
        self.use_actv = use_actv
        
    def forward(self, x):
        return self.actv(self.bn(self.conv(x))) if self.use_actv else self.bn(self.conv(x))
        
        
class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, scale_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * scale_factor ** 2, 3, 1, 1)
        self.ps = nn.PixelShuffle(scale_factor)  # in_c * sf**2, H, W --> in_c, H*sf, W*sf
        self.actv = nn.PReLU(num_parameters=in_channels)
       
    def forward(self, x):
        return self.actv(self.ps(self.conv(x)))
    
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.block1 = ConvBlock(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.block2 = ConvBlock(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            use_actv=False,
        )
        
    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        return out + x
    

class Generator(nn.Module):
    def __init__(self, in_channels=3, num_channels = 64, num_blocks=16):
        super().__init__()
        self.initial = ConvBlock(in_channels, num_channels, kernel_size=9, stride=1, padding=4, use_bn=False)
        self.residuals = nn.Sequential(*[ResidualBlock(num_channels) for _ in range(num_blocks)])
        self.convblock = ConvBlock(num_channels, num_channels, kernel_size=3, stride=1, padding=1, use_actv=False)
        self.upsamples = nn.Sequential(
            UpsampleBlock(num_channels, scale_factor=2),
            UpsampleBlock(num_channels, scale_factor=2),
        )
        self.final = nn.Conv2d(num_channels, in_channels, kernel_size=9, stride=1, padding=4)
    
    def forward(self, x):
        initial = self.initial(x)
        x = self.residuals(initial)
        x = self.convblock(x) + initial
        x = self.upsamples(x)
        return torch.tanh(self.final(x))
        
                                   

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 64, 128, 128, 256, 256, 512, 512]):
        super().__init__()
        blocks = []
        for idx, feature in enumerate(features):
            blocks.append(
                ConvBlock(
                    in_channels,
                    feature,
                    kernel_size=3,
                    stride = 1 + idx%2,
                    padding=1,
                    discriminator=True,
                    use_actv=True,
                    use_bn=False if idx==0 else True,
                )
            )
            in_channels = feature
            
        self.net = nn.Sequential(*blocks)
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6,6)),
            nn.Flatten(),
            nn.Linear(512*6*6, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
        )
        
    def forward(self, x):
        x = self.net(x)
        return self.classifier(x)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
