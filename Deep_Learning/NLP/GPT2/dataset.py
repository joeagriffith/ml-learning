import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import tiktoken

import os

class TinyShakespeare(Dataset):
    def __init__(self, block_size, device):
        path = '../Datasets/mini_shakespeare.txt'

        if not os.path.isfile(path):
            print("Dataset not found")

        with open(path, 'r') as file:
            data = file.read()

        self.block_size = block_size
        self.enc = tiktoken.get_encoding('gpt2')
        data = self.enc.encode(data)
        self.data = torch.tensor(data, dtype=torch.long, device=device)
        self.device = device
    
    def __len__(self):
        return len(self.data) - self.block_size - 1
    
    def to(self, device):
        self.data = self.data.to(device)
        self.device = device
    
    def __getitem__(self, idx):
        return self.data[idx:idx+self.block_size], self.data[idx+1:idx+self.block_size+1]
    
    def get_as_text(self, idx):
        return self.enc.decode(self.data[idx:idx+self.block_size].tolist()) 
    
    def get_random_batch(self, batch_size):
        start = torch.randint(0, len(self.data) - (batch_size*self.block_size) - 1, (1,)).item()
        return self.get_specific_batch(start, batch_size)
    
    def get_specific_batch(self, start, batch_size):
        inputs = self.data[start:start+(batch_size*self.block_size)].view(batch_size, self.block_size)
        targets = self.data[start+1:start+(batch_size*self.block_size)+1].view(batch_size, self.block_size)
        return inputs, targets
    
    def split(self, val_split=0.1):
        train = TinyShakespeare(self.block_size, self.device)
        val = TinyShakespeare(self.block_size, self.device)

        split_idx = int(len(self.data) * (1 - val_split))
        train.data = self.data[:split_idx]
        val.data = self.data[split_idx:]

        return train, val