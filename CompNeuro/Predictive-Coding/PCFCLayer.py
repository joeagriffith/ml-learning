import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter

class PCFCLayer(nn.Module):
    def __init__(self, 
                 e_size,
                 r_size,
                 
                 nu, # r decay
                 mu, # step_size 
                 eta,# step_size correction

                 bu_actv=nn.Tanh(),
                 td_actv=nn.Tanh(),
                 
                 relu_errs=True,
                ):
        super().__init__()
        self.e_size = e_size
        self.r_size = r_size

        self.nu = nu
        self.mu = mu
        self.eta = eta

        self.relu_errs = relu_errs
        self.device = "cpu"

        self.bottomUp = nn.Sequential(
            nn.Linear(e_size, r_size),
            bu_actv,
        )
        
        self.topDown = nn.Sequential(
            nn.Linear(r_size, e_size),
            td_actv,
        )
    
    def init_vars(self, batch_size):
        e = torch.zeros((batch_size, self.e_size)).to(self.device)
        r = torch.zeros((batch_size, self.r_size)).to(self.device)
        return e,r
    
    def forward(self, x, e, r, td_err=None):
        e = x - self.topDown(r)
        if self.relu_errs:
            e = F.relu(e)
        r = self.nu*r + self.mu*self.bottomUp(e)
        if td_err is not None:
            r += self.eta*td_err
        return e, r



class PCFCLayerv2(nn.Module):
    __constants__ = ['in_features', 'out_features', 'nu', 'mu', 'eta']
    in_features: int
    out_features: int
    nu: float
    mu: float
    eta: float
    weight: Tensor
    td_actv: nn.Module

    def __init__(
            self, 
            e_size,
            r_size,
            r_bias=True,
            e_bias=True,
                 
            nu=1.0, # r decay
            mu=0.2, # step_size 
            eta=0.05,# step_size correction

            td_actv=F.tanh(),
                 
            relu_errs=True,
            device=None,
            dtype=None,
        ) -> None:
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super(PCFCLayerv2, self).__init__()
        self.in_features = e_size
        self.out_features = r_size
        self.nu = nu
        self.mu = mu
        self.eta = eta
        self.td_actv = td_actv

        self.relu_errs = relu_errs

        self.weight = Parameter(torch.empty((r_size, e_size), **self.factory_kwargs))   
        if r_bias:
            self.r_bias = Parameter(torch.empty(r_size, **self.factory_kwargs))
        else:
            self.register_parameter('r_bias', None)
        if e_bias:
            self.e_bias = Parameter(torch.empty(e_size, **self.factory_kwargs))
        else:
            self.register_parameter('e_bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.weight, a=math.sqrt(5))
        if self.r_bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.r_bias, -bound, bound)
        if self.e_bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight.T)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.e_bias, -bound, bound)
    
    def init_vars(self, batch_size):
        e = torch.zeros((batch_size, self.e_size), **self.factory_kwargs)
        r = torch.zeros((batch_size, self.r_size), **self.factory_kwargs)
        return e,r
    
    def forward(self, x, e, r, td_err=None):
        e = x - self.td_actv(F.Linear(r, self.weight.T, self.e_bias))
        if self.relu_errs:
            e = F.relu(e)
        r = self.nu*r + self.mu*F.Linear(e, self.weight, self.r_bias)
        if td_err is not None:
            r += self.eta*td_err
        return e, r