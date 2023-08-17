import torch
import torch.nn as nn
import torch.nn.functional as F


class FFLayer(nn.Linear):
    def __init__(self, in_features, out_features, activation_fn=torch.relu, threshold=2.0, bias=True, device=torch.device('cpu'), dtype=torch.float32):
        super(FFLayer, self).__init__(in_features, out_features, bias, device, dtype)
        self.device = device
        self.dtype = dtype
        self.activation_fn = activation_fn
        self.threshold = threshold

    def calc_loss(self, pos_actvs, neg_actvs, mode="maximise"):
        
        if mode == "maximise":
            pos_goodness = torch.sigmoid((pos_actvs.square() - self.threshold).sum(dim=1)).sum()
            neg_goodness = (1 - torch.sigmoid((neg_actvs.square() - self.threshold).sum(dim=1))).sum()
        
        elif mode == "minimise":
            pos_goodness = (1 - torch.sigmoid((pos_actvs.square() - self.threshold).sum(dim=1))).sum()
            neg_goodness = torch.sigmoid((neg_actvs.square() - self.threshold).sum(dim=1)).sum()
        
        return pos_goodness + neg_goodness
    
    def forward(self, x):
        """
        The input is normalised, 
        then passed through a simple forward pass of a linear layer with an activation function.

        Args:
            x: torch.Tensor

        Returns:
            torch.Tensor

        """
        x = self.activation_fn(super(FFLayer, self).forward(x))
        return F.normalize(x)
    

class FFNet(nn.Module):
    def __init__(self, sizes, activation_fn=torch.relu, dropout=0.0, bias=True, device=torch.device('cpu'), dtype=torch.float32):
        super(FFNet, self).__init__()
        self.device = device
        self.dtype = dtype
        self.layers = nn.ModuleList([FFLayer(sizes[i], sizes[i+1], activation_fn, bias=bias, device=device, dtype=dtype) for i in range(len(sizes)-1)])
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, return_all=False):
        if return_all:
            outs = []
            for layer in self.layers:
                x = layer(self.dropout(x))
                outs.append(x)
            return outs

        else:
            for layer in self.layers:
                x = layer(x)
            return x
    
