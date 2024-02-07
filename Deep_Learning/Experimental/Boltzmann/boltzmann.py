import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math

# Boltzmann Machine with activations of 0 and 1
class BoltzmannMachine(nn.Module):
    __constants__ = ['size']
    size: int

    def __init__(self, v_size:int, h_size:int, device:str='cpu'):
        super().__init__()
        self.v_size = v_size
        self.h_size = h_size
        self.device = device

        self.vis_bias = nn.Parameter(torch.zeros(v_size, device=device) * 0.01)
        self.hid_bias = nn.Parameter(torch.zeros(h_size, device=device) * 0.01)
        self.vis_hid = nn.Parameter(torch.randn(v_size, h_size, device=device) * 0.01)
        self.vis_vis_raw = nn.Parameter(torch.randn(v_size, v_size, device=device) * 0.01)
        self.hid_hid_raw = nn.Parameter(torch.randn(h_size, h_size, device=device) * 0.01)
    
    @property
    def vis_vis(self):
        return torch.triu(self.vis_vis_raw, diagonal=1) + torch.triu(self.vis_vis_raw, diagonal=1).T
    
    @property
    def hid_hid(self):
        return torch.triu(self.hid_hid_raw, diagonal=1) + torch.triu(self.hid_hid_raw, diagonal=1).T

    def init_state(self, x:torch.Tensor=None, batch_size:int=1, mean_field:bool=False):
        state = {}
        if x is None:
            state['vis'] = torch.ones((batch_size, self.v_size), device=self.device) * 0.5
            if not mean_field:
                state['vis'] = torch.bernoulli(state['vis'])
        else:
            state['vis'] = x

        state['hid'] = torch.ones((state['vis'].shape[0], self.h_size), device=self.device) * 0.5
        if not mean_field:
            state['hid'] = torch.bernoulli(state['hid'])
        
        return state
    
    def energy(self, state:dict):
        return (
                - ( state['vis'] @ self.vis_bias) \
                - ( state['hid'] @ self.hid_bias) \
                - ((state['vis'] @ self.vis_hid) * state['hid']).sum(1) \
                - ((state['hid'] @ torch.triu(self.hid_hid, diagonal=1)) * state['hid']).sum(1) \
                - ((state['vis'] @ torch.triu(self.vis_vis, diagonal=1)) * state['vis']).sum(1) \
        ).mean()

    # Temperature starts at 10 and decays exponentially to 1
    def calc_temp(self, step_i:int, steps:int):
        return 0.01 * (1 + 4 * math.exp(-5 * step_i / steps))

    def step(self, state:dict, temp:float, pin_vis:bool=False, mean_field:bool=False):
        hid_z = (state['hid'] @ self.hid_hid) + (state['vis'] @ self.vis_hid) + self.hid_bias
        hid_prob = torch.sigmoid(hid_z / temp)
        if mean_field:
            state['hid'] = 0.9 * state['hid'] + 0.1 * hid_prob
        else:
            state['hid'] = torch.bernoulli(hid_prob)
        
        if not pin_vis:
            vis_z = (state['vis'] @ self.vis_vis) + (state['hid'] @ self.vis_hid.T) + self.vis_bias
            vis_prob = torch.sigmoid(vis_z / temp)
            if mean_field:
                state['vis'] = 0.9 * state['vis'] + 0.1 * vis_prob
            if not mean_field:
                state['vis'] = torch.bernoulli(vis_prob)

        return state
    
    def update_grads(self, state:dict, maximise:bool=True):
        for p in self.parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)

        flip = 1 if maximise else -1    
        
        self.vis_bias.grad += flip * (state['vis'].mean(0))
        self.hid_bias.grad += flip * (state['hid'].mean(0))

        self.vis_hid.grad += flip * (state['vis'].T @ state['hid']) / state['vis'].shape[0]
        self.vis_vis_raw.grad += flip * (state['vis'].T @ state['vis']) / state['vis'].shape[0]
        self.hid_hid_raw.grad += flip * (state['hid'].T @ state['hid']) / state['hid'].shape[0]


    def reconstruct(self, x:torch.Tensor, max_steps:int=100):

        state = self.init_state(x, mean_field=True)
        
        for step_i in range(max_steps):
            temp = self.calc_temp(step_i, max_steps)
            state = self.step(state, temp, pin_vis=False, mean_field=True)
        
        return state[0]
    
    def learn(self, x:torch.Tensor, steps:int=100):

        state = self.init_state(x, mean_field=True)
        for step_i in range(steps):
            temp = self.calc_temp(step_i, steps)
            state = self.step(state, temp, pin_vis=True, mean_field=True)

        self.update_grads(state, True)
        energy = self.energy(state)

        return energy.item()


    def unlearn(self, state:dict, steps:int=100):

        for step_i in range(steps):
            temp = self.calc_temp(step_i, steps)
            state = self.step(state, temp, False)
        
        self.update_grads(state, False)
        energy = self.energy(state)

        return state, energy.item()
        

        
