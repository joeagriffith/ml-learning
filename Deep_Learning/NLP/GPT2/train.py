import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def train(
    model,
    optimizer,
    batch_size,
    epochs,
    train_dataset,
    val_dataset=None,
    log_dir=None,
    save_dir=None,
):

    if log_dir is not None:
        writer = SummaryWriter(log_dir)

    stats = {
        'best_loss': float('inf'),
    }
    for epoch in range(epochs):

        num_batches = len(train_dataset) // (batch_size*train_dataset.block_size) + 1
        # num_batches = 1

        epoch_losses = torch.zeros(num_batches)
        loop = tqdm(range(num_batches), postfix=stats, leave=False)
        loop.set_description(f'Epoch [{epoch+1}/{epochs}]')
        for i in loop:
            inputs, targets = train_dataset.get_random_batch(batch_size)
            # inputs, targets = train_dataset.get_specific_batch(0, batch_size)

            logits = model(inputs)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses[i] = loss.item()
        
        stats['train_loss'] = epoch_losses.mean().item()
        writer.add_scalar('Loss/train', stats['train_loss'], epoch)
        last_loss = stats['train_loss']

        if val_dataset is not None:
            with torch.no_grad():
                num_batches = len(val_dataset) // (batch_size*val_dataset.block_size) + 1

                epoch_losses = torch.zeros(num_batches)
                for i in range(num_batches):
                    inputs, targets = val_dataset.get_random_batch(batch_size)

                    logits = model(inputs)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
                    epoch_losses[i] = loss.item()
                
                stats['val_loss'] = epoch_losses.mean().item()
                writer.add_scalar('Loss/val', stats['val_loss'], epoch)

                # Overwrites if val_dataset exists
                last_loss = stats['val_loss']
        
        if last_loss < stats['best_loss']:
            stats['best_loss'] = last_loss
            if save_dir is not None:
                torch.save(model.state_dict(), save_dir)
        
        



        
