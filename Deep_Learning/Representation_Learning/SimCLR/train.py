import torch
from Deep_Learning.Representation_Learning.SimCLR.functional import NTXent
from Deep_Learning.Representation_Learning.SimCLR.lars import LARS
from tqdm import tqdm


def train(
        model,
        train_dataset,
        val_dataset,
        num_epochs,
        batch_size,
        lr,
        wd,
        temperature,
        augmentation,
        writer=None,
        save_dir=None,
        save_every=1,
):
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    optimiser = LARS(model.parameters(), lr=lr, weight_decay=wd)
    scaler = torch.cuda.amp.GradScaler()

    last_train_loss = -1
    last_val_loss = -1
    best_val_loss = float('inf')
    postfix = {}
    for epoch in range(num_epochs):
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        loop.set_description(f'Epoch [{epoch}/{num_epochs}]')
        if epoch > 0:
            loop.set_postfix(postfix)

        epoch_train_losses = torch.zeros(len(train_loader), device=next(model.parameters()).device)
        for i, (images, _) in loop:
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    x1 = augmentation(images)
                    x2 = augmentation(images)
                    x = torch.stack([x1, x2], dim=1).reshape(images.shape[0]*2, images.shape[1], images.shape[2], images.shape[3]).contiguous()

                h = model(x)
                z = model.project(h)
                loss = NTXent(z, temperature)

            optimiser.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()


        epoch_train_losses[i] = loss.detach()
        
        with torch.no_grad():
            epoch_val_losses = torch.zeros(len(val_loader), device=next(model.parameters()).device)
            for i, (images, _) in enumerate(val_loader):
                with torch.cuda.amp.autocast():
                    x1 = augmentation(images)
                    x2 = augmentation(images)
                    x = torch.stack([x1, x2], dim=1).reshape(-1, images.shape[1], images.shape[2], images.shape[3]).contiguous()

                    h = model(x)
                    z = model.project(h)
                    loss = NTXent(z, temperature)

                epoch_val_losses[i] = loss.detach()
        
        last_train_loss = epoch_train_losses.mean().item()
        last_val_loss = epoch_val_losses.mean().item()
        postfix = {'train_loss': last_train_loss, 'val_loss': last_val_loss} 
        if writer is not None:
            writer.add_scalar('Encoder/train_loss', last_train_loss, epoch)
            writer.add_scalar('Encoder/val_loss', last_val_loss, epoch)
        if last_val_loss < best_val_loss and save_dir is not None and epoch % save_every == 0:
            best_val_loss = last_val_loss
            torch.save(model.state_dict(), save_dir)