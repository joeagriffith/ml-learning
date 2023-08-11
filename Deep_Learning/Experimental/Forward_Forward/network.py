import torch
import torch.nn as nn
import torch.nn.functional as F

class Feedforward(nn.Module):
    def __init__(self, sizes, activation=nn.ReLU(), bias=True, momentum=0.95):
        super().__init__()
        self.in_features = sizes[0]
        self.out_features = sizes[-1]
        self.hidden_features = sizes[1:-1]
        self.activation = activation
        self.momentum = momentum

        self.layers = nn.ModuleList()
        for i in range(len(sizes)-1):
            self.layers.append(nn.Linear(sizes[i], sizes[i+1], bias=bias))
        self.target_actvs = [0 for _ in self.layers]
    
    def update_target_actvs(self, x, layer_i):
        new_target = x.mean().item()
        self.target_actvs[layer_i] = self.target_actvs[layer_i]*self.momentum + new_target*(1-self.momentum)

    def forward(self, x, out_layer=None):
        outs = []
        for i, layer in enumerate(self.layers):
            out = self.activation(layer(F.normalize(x, dim=1)))
            outs.append(out)
            self.update_target_actvs(out, i)
            if i == out_layer:
                return out, x
            x = out
        return outs