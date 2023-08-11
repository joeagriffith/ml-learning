import torch
from torch.utils.data import DataLoader

def eval(
    model,
    data_loader,
    criterion,

):

def train_ff(
        model,
        train_dataset,
        val_dataset,
        eval_criterion,
        epochs,
        mode='minimise' # minise or maximise pre-normalised activations
):
    assert type(epochs) == list, f"epochs must be a list (int for each layer), got: {type(epochs)}"
    modes = ['minimise', 'maximise']
    assert mode in modes, f"mode: {mode} is invalid, must be one of: {modes}"






