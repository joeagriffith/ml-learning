import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from Deep_Learning.Experimental.Forward_Forward.utils import goodness_loss, prob_loss, log_loss


def unsupervised(
    model,
    lr,
    pos_dataset,
    neg_dataset,
    val_pos_dataset,
    val_neg_dataset,
    epochs=20,
    batch_size=256,
    mode='minimise',
    loss='log',
    tracker=None,
):

    if type(epochs) == int:
        epochs = [epochs for layer in model.layers]
    assert mode in ['minimise', 'maximise'], "Mode must be either 'minimise' or 'maximise'"
    assert loss in ['goodness', 'prob', 'log'], "Loss must be either 'goodness', 'prob' or 'log'"

    if loss == 'goodness':
        loss_fn = goodness_loss
    elif loss == 'prob':
        loss_fn = prob_loss
    elif loss == 'log':
        loss_fn = log_loss

    tracker = {
        "layer_losses": [[] for layer in model.layers],
        "layer_val_losses": [[] for layer in model.layers],
        "layer_diffs": [[] for layer in model.layers],
        "layer_val_diffs": [[] for layer in model.layers],
    }

    for layer_i in range(len(model.layers)):

        optimiser = torch.optim.AdamW(model.layers[layer_i].parameters(), lr=lr, weight_decay=0.1)

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
                loop.set_description(
                    f"Epoch {ep+1}/{epochs[layer_i]} - "
                    f"Layer {layer_i}/{len(epochs)-1} - "
                    f"Loss: {tracker['layer_losses'][layer_i][-1]:.4f} - "
                    f"Val Loss: {tracker['layer_val_losses'][layer_i][-1]:.4f} - "
                    f"Diff Logits: {tracker['layer_diffs'][layer_i][-1]:.4f} - "
                    f"Val Diff Logits: {tracker['layer_val_diffs'][layer_i][-1]:.4f}"
                )

            # Train Pass
            epoch_loss = 0 
            epoch_diffs = 0
            model.train()
            for batch_i, (x, y) in loop:
                
                # Positive Pass
                with torch.no_grad():
                    x = x.flatten(start_dim=1)
                    for i in range(layer_i):
                        x = F.normalize(model.layers[i](x))
                pos_actvs = model.layers[layer_i](x.detach())            

                # Negative Pass
                with torch.no_grad():
                    x, y = next(neg_dataloader)
                    x = x.flatten(start_dim=1)
                    for i in range(layer_i):
                        x = F.normalize(model.layers[i](x))
                neg_actvs = model.layers[layer_i](x.detach())

                optimiser.zero_grad()
                loss, diffs = loss_fn(pos_actvs, neg_actvs, model.layers[layer_i].threshold, mode=mode)
                loss.backward()
                optimiser.step()

                epoch_loss += loss.item()
                epoch_diffs += diffs.item()

            tracker['layer_losses'][layer_i].append(epoch_loss / len(pos_dataloader))
            tracker['layer_diffs'][layer_i].append(epoch_diffs / len(pos_dataloader))
            
            # Validation Pass
            epoch_val_loss = 0
            epoch_val_diffs = 0
            model.eval()
            for batch_i, (x, y) in enumerate(val_pos_dataloader):

                # Positive Pass
                with torch.no_grad():
                    x = x.flatten(start_dim=1)
                    for i in range(layer_i):
                        x = F.normalize(model.layers[i](x))
                pos_actvs = model.layers[layer_i](x)
            
                # Negative Pass
                with torch.no_grad():
                    x, y = next(val_neg_dataloader)
                    x = x.flatten(start_dim=1)
                    for i in range(layer_i):
                        x = F.normalize(model.layers[i](x))
                neg_actvs = model.layers[layer_i](x)
                
                val_loss, diffs = loss_fn(pos_actvs, neg_actvs, model.layers[layer_i].threshold, mode)
                epoch_val_loss += val_loss.item()
                epoch_val_diffs += diffs.item()
            
            tracker['layer_val_losses'][layer_i].append(epoch_val_loss / len(val_pos_dataloader))
            tracker['layer_val_diffs'][layer_i].append(epoch_val_diffs / len(val_pos_dataloader))

    return tracker


def unsupervised_tracked(
    model,
    lr,
    pos_dataset,
    neg_dataset,
    val_pos_dataset,
    val_neg_dataset,
    epochs=20,
    batch_size=256,
    mode='minimise',
    loss='log',
    tracker=None,
):

    if type(epochs) == int:
        epochs = [epochs for layer in model.layers]
    assert mode in ['minimise', 'maximise'], "Mode must be either 'minimise' or 'maximise'"
    assert loss in ['goodness', 'prob', 'log'], "Loss must be either 'goodness', 'prob' or 'log'"

    if loss == 'goodness':
        loss_fn = goodness_loss
    elif loss == 'prob':
        loss_fn = prob_loss
    elif loss == 'log':
        loss_fn = log_loss

    if tracker is None:
        tracker = {
            "layer_losses": [[] for layer in model.layers],
            "layer_val_losses": [[] for layer in model.layers],
            "layer_diffs": [[] for layer in model.layers],
            "layer_val_diffs": [[] for layer in model.layers],
            "actvs": [[[] for layer in model.layers] for _ in range(4)],
            "norms": [[[] for layer in model.layers] for _ in range(4)],
            "weights": [[] for layer in model.layers],
            "biases": [[] for layer in model.layers],
            "steps": [0 for layer in model.layers],
        }

    # Learning Loop. For each layer, for each epoch, for each batch, perform a positive pass and a negative pass.
    for layer_i in range(len(model.layers)):

        optimiser = torch.optim.AdamW(model.layers[layer_i].parameters(), lr=lr, weight_decay=0.1)

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
                loop.set_description(
                    f"Epoch {ep+1}/{epochs[layer_i]} - "
                    f"Layer {layer_i}/{len(epochs)-1} - "
                    f"Loss: {tracker['layer_losses'][layer_i][-1]:.4f} - "
                    f"Val Loss: {tracker['layer_val_losses'][layer_i][-1]:.4f} - "
                    f"Diff Logits: {tracker['layer_diffs'][layer_i][-1]:.4f} - "
                    f"Val Diff Logits: {tracker['layer_val_diffs'][layer_i][-1]:.4f}"
                )

            # Initialise batch trackers
            epoch_loss = 0 
            epoch_diffs = 0
            batches_pos_actv_total = torch.zeros((model.layers[layer_i].out_features))
            batches_neg_actv_total = torch.zeros((model.layers[layer_i].out_features))
            batches_pos_norm_total = 0
            batches_neg_norm_total = 0
            model.train()
            for batch_i, (x, y) in loop:

                # Positive Pass
                with torch.no_grad():
                    x = x.flatten(start_dim=1)
                    for i in range(layer_i):
                        x = F.normalize(model.layers[i](x))
                pos_actvs = model.layers[layer_i](x.detach())
                # Track activations and norms
                batches_pos_actv_total += pos_actvs.detach().sum(dim=0).cpu()
                batches_pos_norm_total += pos_actvs.detach().norm(dim=1).sum().item()
            
                # Negative Pass
                with torch.no_grad():
                    x, y = next(neg_dataloader)
                    x = x.flatten(start_dim=1)
                    for i in range(layer_i):
                        x = F.normalize(model.layers[i](x))
                neg_actvs = model.layers[layer_i](x.detach())
                # Track activations and norms
                batches_neg_actv_total += neg_actvs.detach().sum(dim=0).cpu()
                batches_neg_norm_total += neg_actvs.detach().norm(dim=1).sum().item()

                optimiser.zero_grad()
                loss, diffs = loss_fn(pos_actvs, neg_actvs, model.layers[layer_i].threshold, mode)
                loss.backward()
                optimiser.step()
                epoch_loss += loss.item()
                epoch_diffs += diffs.item()

            # Track epochs
            tracker['layer_losses'][layer_i].append(epoch_loss / len(pos_dataloader))
            tracker['layer_diffs'][layer_i].append(epoch_diffs / len(pos_dataloader))
            tracker['actvs'][0][layer_i].append(batches_pos_actv_total / len(pos_dataset))
            tracker['actvs'][1][layer_i].append(batches_neg_actv_total / len(pos_dataset))
            tracker['norms'][0][layer_i].append(batches_pos_norm_total / len(pos_dataset))
            tracker['norms'][1][layer_i].append(batches_neg_norm_total / len(pos_dataset))
            tracker['weights'][layer_i].append(model.layers[layer_i].layer.weight.data.cpu().clone())
            if model.layers[layer_i].bias is not None:
                tracker['biases'][layer_i].append(model.layers[layer_i].layer.bias.data.cpu().clone())
            tracker['steps'][layer_i] += len(pos_dataset)
            
            # Validation Pass
            epoch_val_loss = 0
            epoch_val_diffs = 0
            batches_val_pos_actv_total = torch.zeros((model.layers[layer_i].out_features))
            batches_val_neg_actv_total = torch.zeros((model.layers[layer_i].out_features))
            batches_val_pos_norm_total = 0
            batches_val_neg_norm_total = 0
            model.eval()
            for batch_i, (x, y) in enumerate(val_pos_dataloader):
                with torch.no_grad():
                    x = x.flatten(start_dim=1)
                    for i in range(layer_i):
                        x = F.normalize(model.layers[i](x))
                pos_actvs = model.layers[layer_i](x.detach())
                # Track activations and norms
                batches_val_pos_actv_total += pos_actvs.detach().sum(dim=0).cpu()
                batches_val_pos_norm_total += pos_actvs.detach().norm(dim=1).sum().item()
            
                # Negative Pass
                with torch.no_grad():
                    x, y = next(val_neg_dataloader)
                    x = x.flatten(start_dim=1)
                    for i in range(layer_i):
                        x = F.normalize(model.layers[i](x))
                neg_actvs = model.layers[layer_i](x.detach())
                # Track activations and norms
                batches_val_neg_actv_total += neg_actvs.detach().sum(dim=0).cpu()
                batches_val_neg_norm_total += neg_actvs.detach().norm(dim=1).sum().item()

                val_loss, diffs = loss_fn(pos_actvs, neg_actvs, model.layers[layer_i].threshold, mode)
                epoch_val_loss += val_loss.item()
                epoch_val_diffs += diffs.item()

            # Track mean activations and norms over epochs
            tracker['layer_val_losses'][layer_i].append(epoch_val_loss / len(val_pos_dataloader))
            tracker['layer_val_diffs'][layer_i].append(epoch_val_diffs / len(val_pos_dataloader))
            tracker['actvs'][2][layer_i].append(batches_val_pos_actv_total / len(val_pos_dataset))
            tracker['actvs'][3][layer_i].append(batches_val_neg_actv_total / len(val_pos_dataset))
            tracker['norms'][2][layer_i].append(batches_val_pos_norm_total / len(val_pos_dataset))
            tracker['norms'][3][layer_i].append(batches_val_neg_norm_total / len(val_pos_dataset))

    return tracker