import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# A Fully Connected Boltzmann Machine with activations of 0 and 1
class FCBoltzmannModel(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int

    def __init__(self, input_size, output_size, hiddens=[]):
        super(FCBoltzmannModel, self).__init__()

        in_features = input_size
        out_features = output_size
        
        layers = []
        in_size = input_size
        for h in hiddens:
            layers.append(nn.Linear(in_size, h))
            in_size = h
        layers.append(nn.Linear(in_size, output_size))        
        self.layers = nn.ModuleList(layers)

    def init_state(self, x=None, batch_size=1, mode="rand"):
        modes = ["rand", "zeros"]
        assert mode in modes, f"mode: {mode}, must be one of: {modes}"

        if x is None:
            if mode == "rand":
                x = torch.ones((batch_size, self.in_features))/2.0
                x = torch.bernoulli(x)
            elif mode == "zeros":
                x = torch.zeros((batch_size, self.in_features))
        
        state = [x]
        for layer in self.layers:
            if mode == "rand":
                state.append(torch.bernoulli(torch.ones((batch_size, layer.out_features))/2.0))
            elif mode == "zeros":
                state.append(torch.zeros((batch_size, layer.out_features)))
        
        return state
      
                
    def _energy_gap(self, state, layer_i:int, node_i:int):
        result = 0

        if layer_i < len(state)-1:
            result += state[layer_i] @ self.layers[layer_i].weight.data[:,node_i]
        
        if layer_i > 0:
            result += self.layers[layer_i-1].weight.data[node_i,:] @ state[layer_i-1] + self.layers[layer_i-1].bias.data[node_i]


    def _update_node(self, state, layer_i:int, node_i:int, temperature=1.0):
        energy_gap = self._energy_gap(state, layer_i, node_i)
        p_1 = 1.0 / (1 + (-energy_gap/temperature).exp())
        activation = 1.0 if p_1 > random.random() else 0.0
        state[layer_i][node_i] = activation
        return state


    def forward(self, x=None, steps=20, temp_range=(1.0, 5.0), replacement=False, unlock_vis=False):

        state = self.init_state(x)
        
        # Build temperatures. linear decrease from max temp_range to min in temp range, over {steps} steps
        grad = (temp_range[1] - temp_range[0]) / steps
        intercept = temp_range[0]
        temperature = lambda x : grad*(steps-x-1) + intercept
        temperatures = [temperature(i) for i in range(steps)]

        # calc random each node
        # layer_i = random.randint(0, len(self.layers))
        # node_i = random.randint(0, self.layers[layer_i].out_features)

        # OR precalc order and shuffle. faster and ensures every node is updated
        for step_i in range(steps):
            ids = [] # list of (layer_i, node_i) for identifying unique units.

            # Randomly select units, without replacement
            if not replacement:
                # Ensures we clamp visible units in positive phase, and update during negative phase
                if x is None or unlock_vis:
                    node_idxs = torch.arange(self.in_features, dtype=torch.int64).unsqueeze(1)
                    layer_idxs = torch.tensor([1]).unsqueeze(1).repeat(node_idxs.shape)
                    ids.append(torch.cat([layer_idxs, node_idxs], dim=1))

                for i, layer in enumerate(self.layers):
                    node_idxs = torch.arange(layer.out_features, dtype=torch.int64).unsqueeze(1)
                    layer_idxs = torch.tensor([i+1]).unsqueeze(1).repeat(node_idxs.shape)
                    ids.append(torch.cat([layer_idxs, node_idxs], dim=1))
                ids = torch.cat(ids, dim=0)

            # Randomly select units, with replacement
            # TODO: handle visible units selection. Not when training
            else:
                num_units = sum([layer.out_features for layer in self.layers]) + self.in_features
                for _ in range(num_units):
                    layer_i = random.randint(0, len(state))
                    node_i = random.randint(0, len(state[layer_i]))
                    ids.append([layer_i, node_i])
                ids = torch.tensor(ids)
            
            # shuffle to random
            ids = ids[torch.randperm(len(ids))]

            # update units in 'ids' order
            for layer_i, node_i in ids:
                state = self._update_node(state, layer_i.item(), node_i.item(), temperatures[step_i])
        
        return state


    def _correlations(self, state):
        out = []
        for i in range(len(state)-1):
            out.append((state[i+1].t() @ state[i]) / len(state[i]))
        return out
    
    def update(self, state, lr=0.1, negative=False):
        correlations = self._correlations(state)
        if negative:
            correlations = [-1.0 * corr for corr in correlations]
        
        for i, corr in correlations:
            self.layers[i].weight.data += lr*corr




