import torch
import torch.nn as nn

from torchvision.models import resnet18, alexnet

class Layer(nn.Module):
    def __init__(self, in_features, out_features, action_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.action_features = action_features

        self.encoder = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
        )

        # self.transition = nn.Sequential(
        #     nn.Linear(self.out_features + self.action_features, 4096, bias=False),
        #     nn.ReLU(),
        #     nn.Linear(4096, 2048, bias=False),
        #     nn.ReLU(),
        #     nn.Linear(2048, self.out_features, bias=False)
        # )

        self.generate = nn.Sequential(
            nn.Linear(self.out_features + self.action_features, 4096, bias=False),
            nn.ReLU(),
            nn.Linear(4096, 2048, bias=False),
            nn.ReLU(),
            nn.Linear(2048, self.in_features, bias=False)
        )

    def forward(self, x):
        z = self.encoder(x)
        return z
    
    def predict(self, x, a):
        z = self.encoder(x)
        # z_pred = self.transition(torch.cat([z, a], dim=1))
        pred = self.generate(torch.cat([z, a], dim=1))
        return pred, z
        
class HAugPC(nn.Module):
    def __init__(self, sizes, num_actions, action_features=128):
        super().__init__()
        self.num_actions = num_actions
        self.action_features = action_features
        self.num_features = sizes[-1]
    
        self.action_encoder = nn.Sequential(
            nn.Linear(num_actions, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_features),
            nn.ReLU(),
        )

        self.layers = nn.ModuleList([
            Layer(sizes[i], sizes[i+1], action_features) for i in range(len(sizes) - 1)
        ])

    def forward(self, x, return_all_latents=False):
        xs = [x]
        for layer in self.layers:
            xs.append(layer(xs[-1]))
        if return_all_latents:
            return xs
        else:
            return xs[-1]
    
    def predict(self, x, a=None):
        if a is None:
            a = torch.zeros(x.shape[0], self.action_features, device=x.device)
        a = self.action_encoder(a)

        preds = []
        for layer in self.layers:
            pred, x = layer.predict(x, a)
            preds.append(pred)
        
        return preds