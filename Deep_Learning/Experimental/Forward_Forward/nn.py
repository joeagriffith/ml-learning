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
    def __init__(self, in_features, out_features, activation_fn, threshold=2.0, momentum=0.9, bias=True, device=torch.device('cpu'), dtype=torch.float32):
        super(FFLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias, device, dtype)
        self.in_features = in_features
        self.out_features = out_features
        self.activation_fn = activation_fn
        self.threshold = threshold
        self.momentum = momentum
        self.running_mean = torch.zeros(out_features, device=device, dtype=dtype) + 0.5
        self.bias = bias
        self.device = device
        self.dtype = dtype

    # Inspired by loeweX @ https://github.com/loeweX/Forward-Forward/blob/92c0c1c0565063dae0121734075013cab20da5c5/src/ff_model.py#L60
    def calc_peer_norm_loss(self, x):
        # Update running mean
        with torch.no_grad():
            mean_activity = torch.mean(x, dim=0)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean_activity

        # Calculate loss
        peer_loss = (self.running_mean - torch.mean(self.running_mean)) ** 2

        return torch.mean(peer_loss)


    def forward(self, x):
        x = self.activation_fn.apply(self.linear(x))
        return x


class FFNet(nn.Module):
    def __init__(self, sizes, activation_fn=ReLU_full_grad(), threshold=2.0, bias=True, device=torch.device('cpu'), dtype=torch.float32):
        super(FFNet, self).__init__()
        self.device = device
        self.dtype = dtype
        self.layers = nn.ModuleList([FFLayer(sizes[i], sizes[i+1], activation_fn, threshold=threshold, bias=bias, device=device, dtype=dtype) for i in range(len(sizes)-1)])
        self.classifier = nn.Linear(sum(sizes[2:]), 10, bias=False, device=device, dtype=dtype)
        self._init_weights()
        self._init_classifier()
    
    # Initialisations derived from loeweX @ https://github.com/loeweX/Forward-Forward/blob/main/src/ff_model.py
    def _init_weights(self):
        for layer in self.layers:
            nn.init.normal_(layer.linear.weight, mean=0, std=1/math.sqrt(layer.out_features))
            nn.init.zeros_(layer.linear.bias)
    
    def _init_classifier(self):
        nn.init.zeros_(self.classifier.weight)

    def forward(self, x):
        with torch.no_grad():
            outs = []
            for layer in self.layers:
                x = F.normalize(layer(x))
                outs.append(x)
            outs = torch.cat(outs[1:], dim=1)
        out = self.classifier(outs)
        return out
    
