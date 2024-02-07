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

    def __init__(self, vis_features:int, hid_features:int, temp_k:float=0.001, device='cpu'):
        super().__init__()

        self.sizes = [vis_features, hid_features]
        self.temp_k = temp_k
        self.device = device

        self.vis_bias = nn.Parameter(torch.empty(vis_features, device=device)*0.5)
        bound = 1 / math.sqrt(hid_features)
        nn.init.uniform_(self.vis_bias, -bound, bound)

        self.hid_bias = nn.Parameter(torch.empty(hid_features, device=device))
        bound = 1 / math.sqrt(vis_features)
        nn.init.uniform_(self.hid_bias, -bound, bound)

        self.weights = nn.Parameter(torch.empty(vis_features, hid_features, device=device) * 0.1)
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))


    def init_state(self, x:torch.Tensor=None, batch_size:int=None, mean_field:bool=False):
        assert not (x is None and batch_size is None), "Either x or batch_size must be provided."
        if batch_size is None:
            batch_size = x.shape[0]

        state = [torch.ones((batch_size, size), device=self.device) * 0.5 for size in self.sizes]
        if not mean_field:
            state = [torch.bernoulli(s) for s in state]
        if x is not None:
            state[0] = x

        return state
    

    def energy(self, state:List[torch.Tensor]):
        energy = -(state[0] @ self.vis_bias).mean()
        energy += -(state[1] @ self.hid_bias).mean()
        energy += -((state[0] @ self.weights) * state[1]).sum(1).mean()
        return energy


    # Temperature starts at 10 and decays exponentially to 1
    def calc_temp(self, step_i:int, steps:int):
        return self.temp_k * (1 + 4 * math.exp(-5 * step_i / steps))


    def step(self, state:List[torch.Tensor], temp:float, pin_vis:bool=False, mean_field:bool=False):

        if not pin_vis:
            vis_z = F.linear(state[1], self.weights, self.vis_bias)
            vis_prob = torch.sigmoid(vis_z)# / temp)
            if mean_field:
                state[0] = 0.9 * state[0] + 0.1 * vis_prob
            else:
                state[0] = torch.bernoulli(vis_prob)

        hid_z = F.linear(state[0], self.weights.T, self.hid_bias)

        hid_prob = torch.sigmoid(hid_z)# / temp)
        if mean_field:
            state[1] = 0.9 * state[1] + 0.1 * hid_prob
        else:
            assert hid_prob.min() >= 0 and hid_prob.max() <= 1, f"Hidden probabilities out of range: {hid_prob.min()}, {hid_prob.max()}"
            state[1] = torch.bernoulli(hid_prob)
        

        return state
    

    # def free_energy(self, x):
    #     v_bias_term = (x @ self.vis_bias)
    #     wx_b = F.linear(x, self.weights.T, self.hid_bias)
    #     hid_term = wx_b.exp()
    #     hid_term = hid_term.add(1).log().sum(1)
    #     return (-hid_term - v_bias_term).mean()
    def free_energy(self, x):
        v_bias_term = (x @ self.vis_bias).mean()
        # h_bias_term = F.softplus(F.linear(x, self.weights.T, self.hid_bias)).sum(1)
        hid = F.linear(x, self.weights.T, self.hid_bias)
        h_bias_term = (hid @ self.hid_bias).mean()
        vh_term = ((x @ self.weights) * hid).sum(1).mean()
        return -v_bias_term - h_bias_term - vh_term

    

    def forward(self, x, steps:int):
        state = self.init_state(x, mean_field=False)
        state = self.step(state, self.temp_k, pin_vis=True, mean_field=False)

        for _ in range(steps):
            state[0] = state[0].detach()
            state[1] = state[1].detach()
            state = self.step(state, self.temp_k, pin_vis=False, mean_field=False)
        
        return state[0]


    def reconstruct(self, x:torch.Tensor, max_steps:int=100, mean_field=False):

        state = self.init_state(x, mean_field=mean_field)
        
        for step_i in range(max_steps):
            temp = self.calc_temp(step_i, max_steps)
            state = self.step(state, temp, pin_vis=False, mean_field=mean_field)
        
        return state[0]


    def generate(self, sample_size:int=1, steps:int=100, mean_field=False):

        state = self.init_state(batch_size=sample_size, mean_field=mean_field)
        for step_i in range(steps):
            temp = self.calc_temp(step_i, steps)
            state = self.step(state, temp, pin_vis=False, mean_field=mean_field)
        
        return state[0]