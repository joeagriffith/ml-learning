import torch
# from Deep_Learning.Representation_Learning.SimCLR.functional import NTXent
from Deep_Learning.Representation_Learning.SimCLR.lars import LARS
import torch.nn.functional as F
from tqdm import tqdm



def train(
        online_model,
        train_dataset,
        val_dataset,
        num_epochs,
        batch_size,
        lr,
        wd,
        augmentation,
        beta=0.996,
        writer=None,
        save_dir=None,
        save_every=1,
):
    target_model = online_model.copy()
    betas = torch.linspace(beta, 1.0, num_epochs)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    optimiser = torch.optim.AdamW(online_model.parameters(), lr=lr, weight_decay=wd)
    # optimiser = LARS(online_model.parameters(), lr=lr, weight_decay=wd)
    # optimiser = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
    scaler = torch.cuda.amp.GradScaler()

    train_options = {
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'lr': lr,
        'wd': wd,
        'augmentation': str(augmentation),
        'beta': beta,
    }
    if writer is not None:
        writer.add_text('Encoder/options', str(train_options))
        writer.add_text('Encoder/model', str(online_model).replace('\n', '<br/>').replace(' ', '&nbsp;'))
        writer.add_text('Encoder/optimiser', str(optimiser).replace('\n', '<br/>').replace(' ', '&nbsp;'))

    last_train_loss = -1
    last_val_loss = -1
    best_val_loss = float('inf')
    postfix = {}
    for epoch in range(num_epochs):
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        loop.set_description(f'Epoch [{epoch}/{num_epochs}]')
        if epoch > 0:
            loop.set_postfix(postfix)

        epoch_train_losses = torch.zeros(len(train_loader), device=next(online_model.parameters()).device)
        for i, (images, _) in loop:
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    v = augmentation(images)
                    v_dash = augmentation(images)
                    y_dash = target_model(v_dash)
                    z_dash = target_model.project(y_dash)

                y = online_model(v)
                z = online_model.project(y)
                pred = online_model.predict(z)

                loss = F.mse_loss(F.normalize(pred, dim=-1), F.normalize(z_dash, dim=-1))

                # Symmetrize
                with torch.no_grad():
                    y_dash = target_model(v)
                    z_dash = target_model.project(y_dash)

                y = online_model(v_dash)
                z = online_model.project(y)
                pred = online_model.predict(z)

                loss += F.mse_loss(F.normalize(pred, dim=-1), F.normalize(z_dash, dim=-1))

            optimiser.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()

            with torch.no_grad():
                for o_param, t_param in zip(online_model.parameters(), target_model.parameters()):
                    t_param.data = betas[epoch] * t_param.data + (1 - betas[epoch]) * o_param.data

            epoch_train_losses[i] = loss.detach()
        
        with torch.no_grad():
            epoch_val_losses = torch.zeros(len(val_loader), device=next(online_model.parameters()).device)
            for i, (images, _) in enumerate(val_loader):
                with torch.cuda.amp.autocast():
                    v = augmentation(images)
                    v_dash = augmentation(images)

                    y_dash = target_model(v_dash)
                    z_dash = target_model.project(y_dash)
                    y = online_model(v)
                    z = online_model.project(y)
                    pred = online_model.predict(z)
                    loss = F.mse_loss(F.normalize(pred, dim=-1), F.normalize(z_dash, dim=-1))

                    y_dash = target_model(v)
                    z_dash = target_model.project(y_dash)
                    y = online_model(v_dash)
                    z = online_model.project(y)
                    pred = online_model.predict(z)
                    loss += F.mse_loss(F.normalize(pred, dim=-1), F.normalize(z_dash, dim=-1))

                epoch_val_losses[i] = loss.detach()
        
        last_train_loss = epoch_train_losses.mean().item() 
        last_val_loss = epoch_val_losses.mean().item()
        postfix = {'train_loss': last_train_loss, 'val_loss': last_val_loss}
        if writer is not None:
            writer.add_scalar('Encoder/train_loss', last_train_loss, epoch)
            writer.add_scalar('Encoder/val_loss', last_val_loss, epoch)
        if last_val_loss < best_val_loss and save_dir is not None and epoch % save_every == 0:
            best_val_loss = last_val_loss
            torch.save(online_model.state_dict(), save_dir)

            

            



