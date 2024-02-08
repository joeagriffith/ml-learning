import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
from typing import List

# Boltzmann Machine with activations of 0 and 1
class RestrictedBoltzmannMachine(nn.Module):
    __constants__ = ['vis_features', 'hid_features', 'temp_k', 'device']
    size: int

    def __init__(self, vis_features:int, hid_features:int, device='cpu'):
        super().__init__()

        self.sizes = [vis_features, hid_features]
        self.device = device

        self.vis_bias = nn.Parameter(torch.empty(vis_features, device=device)*0.5)
        bound = 1 / math.sqrt(hid_features)
        nn.init.uniform_(self.vis_bias, -bound, bound)

        self.hid_bias = nn.Parameter(torch.empty(hid_features, device=device))
        bound = 1 / math.sqrt(vis_features)
        nn.init.uniform_(self.hid_bias, -bound, bound)

        self.weights = nn.Parameter(torch.empty(hid_features, vis_features, device=device) * 0.1)
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))


    def init_state(self, x:torch.Tensor=None, batch_size:int=None, mean_field:bool=False):
        assert not (x is None and batch_size is None), "Either x or batch_size must be provided."
        if batch_size is None:
            batch_size = x.shape[0]

        if x is not None:
            state = [x, torch.empty((batch_size, self.sizes[1]), device=self.device)]
            self.update_hid(state, mean_field=mean_field)
        else:
            state = [torch.ones((batch_size, size), device=self.device) * 0.5 for size in self.sizes]
            if not mean_field:
                state = [torch.bernoulli(s) for s in state]

        return state
    

    def update_vis(self, state:List[torch.Tensor], mean_field:bool=False):
        vis_z = F.linear(state[1], self.weights.T, self.vis_bias)
        vis_prob = torch.sigmoid(vis_z)# / temp)
        if mean_field:
            state[0] = 0.9 * state[0] + 0.1 * vis_prob
        else:
            state[0] = torch.bernoulli(vis_prob)
    

    def update_hid(self, state:List[torch.Tensor], mean_field:bool=False):
        hid_z = F.linear(state[0], self.weights, self.hid_bias)
        hid_prob = torch.sigmoid(hid_z)# / temp)
        if mean_field:
            state[1] = 0.9 * state[1] + 0.1 * hid_prob
        else:
            state[1] = torch.bernoulli(hid_prob)


    def energy(self, state:List[torch.Tensor]):
        energy = -(state[0] @ self.vis_bias).mean()
        energy += -(state[1] @ self.hid_bias).mean()
        energy += -((state[0] @ self.weights) * state[1]).sum(1).mean()
        return energy
    

    def free_energy(self, x):
        v_bias_term = (x @ self.vis_bias)
        wx_b = F.linear(x, self.weights, self.hid_bias)
        hid_term = F.softplus(wx_b).sum(1)
        out = (-hid_term - v_bias_term).mean()
        return out

    # def free_energy(self, state):
    #     v_bias_term = (state[0] @ self.vis_bias).mean()
    #     h_bias_term = (state[1] @ self.hid_bias).mean()
    #     vh_term = ((state[0] @ self.weights.T) * state[1]).sum(1).mean()
    #     return -v_bias_term - h_bias_term - vh_term

    def forward(self, state, steps:int):
        for _ in range(steps):
            state[0] = state[0].detach()
            state[1] = state[1].detach()
            self.update_vis(state)
            self.update_hid(state)