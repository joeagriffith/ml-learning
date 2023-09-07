import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from Deep_Learning.Experimental.Forward_Forward.utils import goodness_loss, prob_loss, log_loss, bce_loss


def train_unsupervised(
    model,
    lr,
    weight_decay,
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
    assert loss in ['goodness', 'prob', 'log', 'bce'], "Loss must be either 'goodness', 'prob', 'log' or 'bce'"

    if loss == 'goodness':
        loss_fn = goodness_loss
    elif loss == 'prob':
        loss_fn = prob_loss
    elif loss == 'log':
        loss_fn = log_loss
    elif loss == 'bce':
        loss_fn = bce_loss

    tracker = {
        "layer_losses": [[] for layer in model.layers],
        "layer_val_losses": [[] for layer in model.layers],
        "layer_diffs": [[] for layer in model.layers],
        "layer_val_diffs": [[] for layer in model.layers],
        "layer_peer_losses": [[] for layer in model.layers],
    }

    for layer_i in range(len(model.layers)):

        optimiser = torch.optim.AdamW(model.layers[layer_i].parameters(), lr=lr, weight_decay=weight_decay)

        for ep in range(epochs[layer_i]):
            # Set up dataloaders and tqdm
            pos_dataset.apply_transform()
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
            epoch_peer_loss = 0
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
                peer_loss = model.layers[layer_i].calc_peer_norm_loss(pos_actvs)
                loss += 0.03 * peer_loss
                loss.backward()
                optimiser.step()

                epoch_loss += loss.item()
                epoch_diffs += diffs.item()
                epoch_peer_loss += peer_loss.item()

            tracker['layer_losses'][layer_i].append(epoch_loss / len(pos_dataloader))
            tracker['layer_diffs'][layer_i].append(epoch_diffs / len(pos_dataloader))
            tracker['layer_peer_losses'][layer_i].append(epoch_peer_loss / len(pos_dataloader))
            
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


def train_unsupervised_tracked(
    model,
    lr,
    weight_decay,
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
    assert loss in ['goodness', 'prob', 'log', 'bce'], "Loss must be either 'goodness', 'prob',  'log' or 'bce'"

    if loss == 'goodness':
        loss_fn = goodness_loss
    elif loss == 'prob':
        loss_fn = prob_loss
    elif loss == 'log':
        loss_fn = log_loss
    elif loss == 'bce':
        loss_fn = bce_loss


    if tracker is None:
        tracker = {
            "layer_losses": [[] for layer in model.layers],
            "layer_val_losses": [[] for layer in model.layers],
            "layer_diffs": [[] for layer in model.layers],
            "layer_val_diffs": [[] for layer in model.layers],
            "layer_peer_losses": [[] for layer in model.layers],
            "actvs": [[[] for layer in model.layers] for _ in range(4)],
            "norms": [[[] for layer in model.layers] for _ in range(4)],
            "weights": [[] for layer in model.layers],
            "biases": [[] for layer in model.layers],
            "steps": [0 for layer in model.layers],
        }

    # Learning Loop. For each layer, for each epoch, for each batch, perform a positive pass and a negative pass.
    for layer_i in range(len(model.layers)):

        # optimiser = torch.optim.AdamW(model.layers[layer_i].parameters(), lr=lr, weight_decay=weight_decay)
        optimiser = torch.optim.SGD(model.layers[layer_i].parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)

        for ep in range(epochs[layer_i]):
            # Set up dataloaders and tqdm
            pos_dataset.apply_transform()
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
            epoch_peer_loss = 0
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
                peer_loss = model.layers[layer_i].calc_peer_norm_loss(pos_actvs)
                loss += 0.03 * peer_loss

                loss.backward()
                optimiser.step()
                epoch_loss += loss.item()
                epoch_diffs += diffs.item()
                epoch_peer_loss += peer_loss.item()

            # Track epochs
            tracker['layer_losses'][layer_i].append(epoch_loss / len(pos_dataloader))
            tracker['layer_diffs'][layer_i].append(epoch_diffs / len(pos_dataloader))
            tracker['layer_peer_losses'][layer_i].append(epoch_peer_loss / len(pos_dataloader))
            tracker['actvs'][0][layer_i].append(batches_pos_actv_total / len(pos_dataset))
            tracker['actvs'][1][layer_i].append(batches_neg_actv_total / len(pos_dataset))
            tracker['norms'][0][layer_i].append(batches_pos_norm_total / len(pos_dataset))
            tracker['norms'][1][layer_i].append(batches_neg_norm_total / len(pos_dataset))
            tracker['weights'][layer_i].append(model.layers[layer_i].linear.weight.data.cpu().clone())
            if model.layers[layer_i].bias:
                tracker['biases'][layer_i].append(model.layers[layer_i].linear.bias.data.cpu().clone())
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
                    pos_actvs = model.layers[layer_i](x)
                    # Track activations and norms
                    batches_val_pos_actv_total += pos_actvs.sum(dim=0).cpu()
                    batches_val_pos_norm_total += pos_actvs.norm(dim=1).sum().item()
            
                # Negative Pass
                with torch.no_grad():
                    x, y = next(val_neg_dataloader)
                    x = x.flatten(start_dim=1)
                    for i in range(layer_i):
                        x = F.normalize(model.layers[i](x))
                    neg_actvs = model.layers[layer_i](x)
                    # Track activations and norms
                    batches_val_neg_actv_total += neg_actvs.sum(dim=0).cpu()
                    batches_val_neg_norm_total += neg_actvs.norm(dim=1).sum().item()

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


def train_classifier(
    model,
    train_dataset, 
    val_dataset, 
    epochs, 
    batch_size,
    optimiser, 
    criterion, 
    stats,
):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    if stats is None:
        stats = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'steps': [],
        }

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        if epoch > 0:
            loop.set_description(f"Epoch {epoch}")
            loop.set_postfix(train_loss=stats['train_loss'][-1], train_acc=stats['train_acc'][-1], val_loss=stats['val_loss'][-1], val_acc=stats['val_acc'][-1])
        for i, data in loop:
            x, y = data
            optimiser.zero_grad()
            y_pred = model(x.flatten(start_dim=1))
            y_pred = y_pred - torch.max(y_pred, dim=1, keepdim=True)[0] # normalisation for numerical stability
            loss = criterion(y_pred, y)
            loss.backward()
            optimiser.step()
            running_loss += loss.item()

            _, predicted = torch.max(y_pred, 1)
            correct += (predicted == y).sum().item()

        stats['train_loss'].append(running_loss / len(train_loader))
        stats['train_acc'].append(correct / len(train_dataset))
        if len(stats['steps']) > 0:
            stats['steps'].append(stats['steps'][-1] + len(train_dataset))
        else:
            stats['steps'].append(len(train_dataset))

        model.eval()
        running_loss = 0.0
        correct = 0
        for i, data in enumerate(val_loader):
            x, y = data
            y_pred = model(x.flatten(start_dim=1))
            y_pred = y_pred - torch.max(y_pred, dim=1, keepdim=True)[0] # normalisation for numerical stability
            loss = criterion(y_pred, y)
            running_loss += loss.item()

            _, predicted = torch.max(y_pred, 1)
            correct += (predicted == y).sum().item()
        stats['val_loss'].append(running_loss / len(val_loader))
        stats['val_acc'].append(correct / len(val_dataset))

    return stats