import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import math

from Deep_Learning.Diffusion.StanfordCars.functional import get_index_from_list


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super(SinusoidalPositionalEmbedding, self).__init__()
        self.dim = dim
    
    def forward(self, time):
        device = time.device
        embeddings = math.log(10000) / (self.dim // 2 - 1)
        embeddings = torch.exp(torch.arange(self.dim//2, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super(DoubleConv, self).__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, t):
        time_emb = self.relu(self.time_mlp(t))
        h = self.bnorm(self.relu(self.conv1(x)))
        time_emb = time_emb[(..., ) + (None, ) * 2]
        h = h + time_emb
        h = self.bnorm(self.relu(self.conv2(h)))
        return h


class DiffusionUNet(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 features, 
                 time_emb_dim, 
                 betas, 
                 sqrt_alphas_cumprod,
                 sqrt_one_minus_alphas_cumprod, 
                 sqrt_recip_alphas, 
                 posterior_variance,
                 T,
                 device='cpu'
                ):
        super(DiffusionUNet, self).__init__()

        self.betas = betas
        self.sqrt_alphas_cumprod = sqrt_alphas_cumprod
        self.sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod
        self.sqrt_recip_alphas = sqrt_recip_alphas
        self.posterior_variance = posterior_variance
        self.T = T
        self.device = device

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(inplace=True),
        )

        self.encoder = nn.ModuleList()
        for feature in features:
            self.encoder.append(DoubleConv(in_channels, feature, time_emb_dim))
            in_channels = feature
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.decoder = nn.ModuleList()
        for feature in reversed(features):
            self.decoder.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.decoder.append(DoubleConv(feature*2, feature, time_emb_dim))
        
        self.bottleneck = DoubleConv(features[-1], features[-1]*2, time_emb_dim)

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def to(self, device):
        self.device = device
        for module in self.encoder + self.decoder + [self.bottleneck, self.final_conv, self.time_mlp]:
            module.to(device)
        return self

    def forward(self, x, timestep):
        t = self.time_mlp(timestep)

        skip_connections = []

        for module in self.encoder:
            x = module(x, t)
            skip_connections.append(x)
            x = self.pool(x)
        skip_connections = skip_connections[::-1] # Reverse list
        
        x = self.bottleneck(x, t)

        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx+1](concat_skip, t)
        
        return self.final_conv(x)

    def sample_timestep(self, x, t):
        with torch.no_grad():
            betas_t = get_index_from_list(self.betas, t, x.shape)
            sqrt_one_minus_alphas_cumprod_t = get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
            sqrt_recip_alphas_t = get_index_from_list(self.sqrt_recip_alphas, t, x.shape)

            model_mean = sqrt_recip_alphas_t * (x - betas_t * self(x, t) / sqrt_one_minus_alphas_cumprod_t)
            posterior_variance_t = get_index_from_list(self.posterior_variance, t, x.shape)

            if t == 0:
                return model_mean
            else:
                noise = torch.rand_like(x)
                return model_mean + torch.sqrt(posterior_variance_t) * noise

    def sample_images(self, img_size=128, device='cpu'):
        images = [torch.randn((1, 3, img_size, img_size), device=device)]
        for i in range(0,self.T)[::-1]:
            t = torch.full((1,), i, device=device, dtype=torch.long)
            img = self.sample_timestep(images[-1], t)
            images.append(torch.clamp(img, -1.0, 1.0))
        return images
        
        
        




def test():
    x = torch.randn((3, 1, 128, 128))
    model = DiffusionUNet(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape

if __name__ == "__main__":
    test()