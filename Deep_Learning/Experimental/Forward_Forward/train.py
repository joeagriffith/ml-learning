import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_unsup_backward(
    model,
    optimiser,
    pos_dataset,
    neg_dataset,
    val_pos_dataset,
    val_neg_dataset,
    epochs=20,
    batch_size=256,
    mode='minimise',
    neg_coeff=1.0,
    track_actvs=False,
    actvs=None, # [pos_actv, neg_actv, val_pos_actv, val_neg_actv]
    track_norms=False,
    norms=None, # [pos_norm, neg_norm, val_pos_norm, val_neg_norm]
    track_weights=False,
    weights=None,
    biases=None,
    steps=None,
    device="cpu",
):
    maxmin = 1 if mode == 'maximise' else -1

    if type(epochs) == int:
        epochs = [epochs for layer in model.layers]
    assert mode in ['minimise', 'maximise'], "Mode must be either 'minimise' or 'maximise'"

    # Initialise trackers
    if track_actvs and actvs[0] is None:
        actvs = [[torch.empty((epochs[layer_i], model.layers[layer_i].out_features)) for layer_i in range(len(model.layers))] for _ in range(4)] # actvs[pos/neg/val_pos/val_neg][layer_i][epoch]
    if track_norms and norms[0] is None:
        norms = [[torch.empty((epochs[layer_i],)) for layer_i in range(len(model.layers))] for _ in range(4)] # norms[pos/neg/val_pos/val_neg][layer_i][epoch]
    if track_weights and weights is None:
        weights = [torch.empty((epochs[layer_i], model.layers[layer_i].out_features, model.layers[layer_i].in_features)) for layer_i in range(len(model.layers))] # weights[layer_i][epoch]
        biases = [torch.empty((epochs[layer_i], model.layers[layer_i].out_features)) for layer_i in range(len(model.layers))] # biases[layer_i][epoch]
    if steps is None:
        steps = [0 for layer in model.layers]

    # Store initial weights
    if track_weights and steps[0] == 0:
        for layer_i in range(len(model.layers)):
            weights[layer_i][0] = model.layers[layer_i].weight.data.clone()
            if model.layers[layer_i].bias is not None:
                biases[layer_i][0] = model.layers[layer_i].bias.data.clone()

    # Learning Loop. For each layer, for each epoch, for each batch, perform a positive pass and a negative pass.
    for layer_i, layer in enumerate(model.layers):
        for ep in range(epochs[layer_i]):

            # Set up dataloaders and tqdm
            pos_dataloader = DataLoader(pos_dataset, batch_size=batch_size, shuffle=True)
            neg_dataset.apply_transform()
            neg_dataloader = iter(DataLoader(neg_dataset, batch_size=batch_size, shuffle=True))
            val_pos_dataloader = DataLoader(val_pos_dataset, batch_size=batch_size, shuffle=False)
            val_neg_dataset.apply_transform()
            val_neg_dataloader = iter(DataLoader(val_neg_dataset, batch_size=batch_size, shuffle=True))
            loop = tqdm(enumerate(pos_dataloader), total=len(pos_dataloader), leave=False)
            if ep > 0:
                if track_norms:
                    loop.set_description(f"Epoch {ep+1}/{epochs[layer_i]} - Layer {layer_i}/{len(epochs)-1} - Mean Norms (pos/neg): {norms[0][layer_i][ep-1].item():.3f} / {norms[1][layer_i][ep-1].item():.3f}")
                else:
                    loop.set_description(f"Epoch {ep+1}/{epochs[layer_i]} - Layer {layer_i}/{len(epochs)-1}")

            # Initialise batch trackers
            if track_actvs:
                batches_pos_actv_total = torch.zeros((layer.out_features)).to(device)
                batches_neg_actv_total = torch.zeros((layer.out_features)).to(device)
            if track_norms:
                batches_pos_norm_total = torch.zeros((1)).to(device)
                batches_neg_norm_total = torch.zeros((1)).to(device)

            for batch_i, (x, y) in loop:
                model.train()

                # Positive Pass
                x = x.flatten(start_dim=1)
                with torch.no_grad():
                    for i in range(layer_i):
                        x = model.layers[i](x)
                pos_actvs = layer(x)
                # Track activations and norms
                if track_actvs:
                    batches_pos_actv_total += pos_actvs.detach().sum(dim=0)
                if track_norms:
                    batches_pos_norm_total += pos_actvs.detach().norm(dim=1).sum()
            
                # Negative Pass
                x, y = next(neg_dataloader)
                x = x.flatten(start_dim=1)
                with torch.no_grad():
                    for i in range(layer_i):
                        x = model.layers[i](x)
                neg_actvs = layer(x)
                # Track activations and norms
                if track_actvs:
                    batches_neg_actv_total += neg_actvs.detach().sum(dim=0)
                if track_norms:
                    batches_neg_norm_total += neg_actvs.detach().norm(dim=1).sum()

                model.zero_grad()
                loss = layer.calc_loss(pos_actvs, neg_actvs, mode)
                loss.backward()
                optimiser.step()

            # Track epochs
            if track_actvs:
                actvs[0][layer_i][ep] = batches_pos_actv_total / len(pos_dataset)
                actvs[1][layer_i][ep] = batches_neg_actv_total / len(pos_dataset)
            if track_norms:
                norms[0][layer_i][ep] = batches_pos_norm_total / len(pos_dataset)
                norms[1][layer_i][ep] = batches_neg_norm_total / len(pos_dataset)
            if track_weights:
                weights[layer_i][ep] = layer.weight.data.clone()
                if layer.bias is not None:
                    biases[layer_i][ep] = layer.bias.data.clone()
            steps[layer_i] += len(pos_dataset)
            
            # Validation Pass
            model.eval()
            if track_actvs:
                batches_val_pos_actv_total = torch.zeros((layer.out_features)).to(device)
                batches_val_neg_actv_total = torch.zeros((layer.out_features)).to(device)
            if track_norms:
                batches_val_pos_norm_total = torch.zeros((1)).to(device)
                batches_val_neg_norm_total = torch.zeros((1)).to(device)
            for batch_i, (x, y) in enumerate(val_pos_dataloader):
                x = x.flatten(start_dim=1)
                with torch.no_grad():
                    for i in range(layer_i+1):
                        x = model.layers[i](x)
                # Track activations and norms
                if track_actvs:
                    batches_val_pos_actv_total += x.sum(dim=0)
                if track_norms:
                    batches_val_pos_norm_total += x.norm(dim=1).sum()
            
                # Negative Pass
                x, y = next(val_neg_dataloader)
                x = x.flatten(start_dim=1)
                with torch.no_grad():
                    for i in range(layer_i+1):
                        x = model.layers[i](x)
                # Track activations and norms
                if track_actvs:
                    batches_val_neg_actv_total += x.sum(dim=0)
                if track_norms:
                    batches_val_neg_norm_total += x.norm(dim=1).sum()

            # Track mean activations and norms over epochs
            if track_actvs:
                actvs[2][layer_i][ep] = batches_val_pos_actv_total / len(val_pos_dataset)
                actvs[3][layer_i][ep] = batches_val_neg_actv_total / len(val_pos_dataset)
            if track_norms:
                norms[2][layer_i][ep] = batches_val_pos_norm_total / len(val_pos_dataset)
                norms[3][layer_i][ep] = batches_val_neg_norm_total / len(val_pos_dataset)

    return actvs, norms, weights, biases, steps