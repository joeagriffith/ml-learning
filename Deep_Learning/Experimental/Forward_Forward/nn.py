import torch
import torch.nn as nn
import torch.nn.functional as F

class FFLayer(nn.Module):
    def __init__(self, in_features, out_features, activation_fn=torch.relu, threshold=2.0, bias=True, device=torch.device('cpu'), dtype=torch.float32):
        super(FFLayer, self).__init__()
        self.layer = nn.Linear(in_features, out_features, bias, device, dtype)
        self.in_features = in_features
        self.out_features = out_features
        self.activation_fn = activation_fn
        self.threshold = threshold
        self.bias = bias
        self.device = device
        self.dtype = dtype

    def forward(self, x):
        x = self.activation_fn(self.layer(x))
        return x


class FFNet(nn.Module):
    def __init__(self, sizes, activation_fn=torch.relu, bias=True, threshold=2.0, device=torch.device('cpu'), dtype=torch.float32):
        super(FFNet, self).__init__()
        self.device = device
        self.dtype = dtype
        self.layers = nn.ModuleList([FFLayer(sizes[i], sizes[i+1], activation_fn, bias=bias, threshold=threshold, device=device, dtype=dtype) for i in range(len(sizes)-1)])
    
    # Returns list of normalised output from each layer
    def forward(self, x):
        outs = []
        for layer in self.layers:
            x = F.normalize(layer(x))
            outs.append(x)
        return outs
    
