import torch
import torch.nn as nn
import torch.nn.functional as F

def goodness(activations, reduction="sum"):
    assert reduction in ["sum", "mean", None], "reduction must be either 'mean' or 'sum'"
    score = activations.square()
    if reduction == "sum":
        score = score.sum(dim=1)
    elif reduction == "mean":
        score = score.mean(dim=1)

    return score

def log_loss(pos_actvs, neg_actvs, threshold=2.0, mode="maximise"):
    
    pos_logits = goodness(pos_actvs, "mean") - threshold
    neg_logits = goodness(neg_actvs, "mean") - threshold
    diff_logits = (pos_logits.mean() - neg_logits.mean()).abs() # For logging purposes

    if mode == "maximise":
        logits = torch.cat([-pos_logits, neg_logits], dim=0)
    elif mode == "minimise":
        logits = torch.cat([pos_logits, -neg_logits], dim=0)
    
    probs = torch.nan_to_num(torch.exp(logits))
    loss = torch.log(1 + probs).mean()
    return loss, diff_logits

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
    
    def forward(self, x):
        """
        x: input tensor

        returns: list of normalised output activations from each layer
        """
        outs = []
        for layer in self.layers:
            x = F.normalize(layer(x))
            outs.append(x)
        return outs
    
