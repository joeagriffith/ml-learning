import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# loeweX @ https://github.com/loeweX/Forward-Forward/blob/main/src/ff_model.py#L161
class ReLU_full_grad(torch.autograd.Function):
    """ ReLU activation function that passes through the gradient irrespective of its input value."""

    @staticmethod
    def forward(ctx, input):
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()

class FFLayer(nn.Module):
    def __init__(self, in_features, out_features, activation_fn=ReLU_full_grad, threshold=2.0, bias=True, device=torch.device('cpu'), dtype=torch.float32):
        super(FFLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias, device, dtype)
        self.in_features = in_features
        self.out_features = out_features
        self.activation_fn = activation_fn
        self.threshold = threshold
        self.bias = bias
        self.device = device
        self.dtype = dtype

    def forward(self, x):
        x = self.activation_fn(self.linear(x))
        return x


class FFNet(nn.Module):
    def __init__(self, sizes, activation_fn=torch.relu, dropout=0.0, bias=True, threshold=2.0, device=torch.device('cpu'), dtype=torch.float32):
        super(FFNet, self).__init__()
        self.device = device
        self.dtype = dtype
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([FFLayer(sizes[i], sizes[i+1], activation_fn, bias=bias, threshold=threshold, device=device, dtype=dtype) for i in range(len(sizes)-1)])
        self._init_weights()
    
    # Initialisations derived from loeweX @ https://github.com/loeweX/Forward-Forward/blob/main/src/ff_model.py
    def _init_weights(self):
        for layer in self.layers:
            nn.init.normal_(layer.linear.weight, mean=0, std=1/math.sqrt(layer.out_features))
            nn.init.zeros_(layer.linear.bias)

    # Returns list of normalised output from each layer
    def forward(self, x):
        outs = []
        for layer in self.layers:
            x = self.dropout(F.normalize(layer(x)))
            outs.append(x)
        return outs
    
