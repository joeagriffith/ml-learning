import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math

# A Fully Connected Boltzmann Machine with activations of 0 and 1
class FCBoltzmannModel(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int

    def __init__(self, input_size, output_size, hiddens=[]):
        super(FCBoltzmannModel, self).__init__()

        self.in_features = input_size
        self.out_features = output_size
        
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
        else:
            batch_size = x.shape[0]
        
        
        state = [x]
        for layer in self.layers:
            if mode == "rand":
                state.append(torch.bernoulli(torch.ones((batch_size, layer.out_features))/2.0))
            elif mode == "zeros":
                state.append(torch.zeros((batch_size, layer.out_features)))
        
        return state
      
                
    def _energy_gap(self, state, layer_i:int, node_i:int):
        result = torch.zeros(state[0].shape[0]) # batch_size

        if layer_i > 0:
            # print(f'state[layer_i-1].shape: {state[layer_i-1].shape}, self.layers[layer_i-1].weight.data[node_i,:].shape: {self.layers[layer_i-1].weight.data[node_i,:].shape}')
            result += state[layer_i-1] @ self.layers[layer_i-1].weight.data[node_i,:] + self.layers[layer_i-1].bias.data[node_i]

        if layer_i < len(state)-1:
            # print(f'state[layer_i+1].shape: {state[layer_i+1].shape}, self.layers[layer_i].weight.data[:,node_i].shape: {self.layers[layer_i].weight.data[:,node_i].shape}')
            result += state[layer_i+1] @ self.layers[layer_i].weight.data[:,node_i]

        return result
        


    def _update_node(self, state, layer_i:int, node_i:int, temperature=1.0, debug=False):
        energy_gap = self._energy_gap(state, layer_i, node_i)
        p_1 = 1.0 / (1 + (-energy_gap/temperature).exp())
        activation = (p_1 > torch.rand(p_1.shape)).float()
        updated = not torch.equal(state[layer_i][:,node_i], activation)
        state[layer_i][:,node_i] = activation
        if debug:
            print(f'layer_i: {layer_i}, node_i: {node_i}, energy_gap: {energy_gap}, p_1: {p_1}, activation: {activation}, updated: {updated}')
        return updated


    def forward(self, x=None, max_steps=100, temp_coeffs=(2.0, 5.0), replacement=False, unlock_vis=False):

        state = self.init_state(x)
        
        # Build temperatures. linear decrease from max temp_range to min in temp range, over {steps} steps
        temperature = lambda x : temp_coeffs[0]/math.exp(x/temp_coeffs[1])
        temperatures = [temperature(i) for i in range(max_steps)]

        # calc random each node
        # layer_i = random.randint(0, len(self.layers))
        # node_i = random.randint(0, self.layers[layer_i].out_features)

        # OR precalc order and shuffle. faster and ensures every node is updated
        # for step_i in range(steps):
        thermal_eq = False
        step_i = 0
        while not thermal_eq:
            thermal_eq = True
            ids = [] # list of (layer_i, node_i) for identifying unique units.

            # Randomly select units, without replacement
            if not replacement:
                # Ensures we clamp visible units in positive phase, and update during negative phase
                if x is None or unlock_vis:
                    node_idxs = torch.arange(self.in_features, dtype=torch.int64).unsqueeze(1)
                    layer_idxs = torch.tensor([0]).unsqueeze(1).repeat(node_idxs.shape)
                    ids.append(torch.cat([layer_idxs, node_idxs], dim=1))

                for i, layer in enumerate(self.layers):
                    node_idxs = torch.arange(layer.out_features, dtype=torch.int64).unsqueeze(1)
                    layer_idxs = torch.tensor([i+1]).unsqueeze(1).repeat(node_idxs.shape)
                    ids.append(torch.cat([layer_idxs, node_idxs], dim=1))
                ids = torch.cat(ids, dim=0)

            # Randomly select units, with replacement
            # TODO: handle visible units selection. Not when training
            else:
                raise(NotImplementedError)
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
                updated = self._update_node(state, layer_i.item(), node_i.item(), temperatures[step_i])
                if updated:
                    thermal_eq = False

            if step_i >= max_steps-1:
                print(f"Max steps reached: {max_steps}")
                break

            step_i += 1 
        return state


    def _correlations(self, state):
        out = []
        for i in range(len(state)-1):
            correlation = (state[i+1].t() @ state[i]) / len(state[i])
            out.append(correlation)
        return out
    
    def update(self, state, lr=0.1, negative=False):
        correlations = self._correlations(state)
        if negative:
            lr *= -1
        
        for i, corr in enumerate(correlations):
            self.layers[i].weight.data += lr*corr




