import torch
import torch.nn.functional as F
import torchvision.transforms.v2.functional as F_v2
from tqdm import tqdm


def train(
        model,
        train_dataset,
        val_dataset,
        num_epochs,
        batch_size,
        lr,
        wd,
        writer=None,
        save_dir=None,
        save_every=1,
        aug_scaler='none',
):
    # Exclude bias and batch norm parameters from weight decay
    decay_parameters = [param for name, param in model.named_parameters() if 'weight' in name]
    decay_parameters = [{'params': decay_parameters}]
    non_decay_parameters = [param for name, param in model.named_parameters() if 'weight' not in name]
    non_decay_parameters = [{'params': non_decay_parameters, 'weight_decay': 0.0}]
    optimiser = torch.optim.AdamW(decay_parameters + non_decay_parameters, lr=lr, weight_decay=wd)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
    scaler = torch.cuda.amp.GradScaler()

    train_options = {
        num_epochs: num_epochs,
        batch_size: batch_size,
        lr: lr,
        wd: wd,
        aug_scaler: aug_scaler
    }
    if writer is not None:
        writer.add_text('Encoder/options', str(train_options))
        writer.add_text('Encoder/model', str(model).replace('\n', '<br/>').replace(' ', '&nbsp;'))
        writer.add_text('Encoder/optimiser', str(optimiser).replace('\n', '<br/>').replace(' ', '&nbsp;'))

    assert aug_scaler in ['linear', 'exp', 'none'], 'aug_scaler must be one of ["linear", "exp"]'
    if aug_scaler == 'linear':
        aug_ps = torch.linspace(0, 0.25, num_epochs)
    elif aug_scaler == 'exp':
        aug_ps = 0.25 * (1.0 - torch.exp(torch.linspace(0, -5, num_epochs)))
    elif aug_scaler == 'none':
        aug_ps = 0.25 * torch.ones(num_epochs)

    last_train_loss = torch.zeros(1, device=next(model.parameters()).device)
    last_val_loss = torch.zeros(1, device=next(model.parameters()).device)
    best_val_loss = torch.zeros(1, device=next(model.parameters()).device) + 1e6
    postfix = {}
    device=next(model.parameters()).device
    for epoch in range(num_epochs):
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        loop.set_description(f'Epoch [{epoch}/{num_epochs}]')
        if epoch > 0:
            loop.set_postfix(postfix)

        epoch_train_losses = torch.zeros(len(train_loader), device=next(model.parameters()).device)
        for i, (images, _) in loop:
            images = images.to(device)

            act_p = torch.rand(5)
            angle = torch.rand(1).item() * 360 - 180 if act_p[0] < aug_ps[epoch] else 0
            translate_x = torch.randint(-8, 9, (1,)).item() if act_p[1] < aug_ps[epoch] else 0
            translate_y = torch.randint(-8, 9, (1,)).item() if act_p[2] < aug_ps[epoch] else 0
            scale = torch.rand(1).item() * 0.5 + 0.75 if act_p[3] < aug_ps[epoch] else 1.0
            shear = torch.rand(1).item() * 50 - 25 if act_p[4] < aug_ps[epoch] else 0
            images_aug = F_v2.affine(images, angle=angle, translate=(translate_x, translate_y), scale=scale, shear=shear)
            action = torch.tensor([angle/180, translate_x/8, translate_y/8, (scale-1.0)/0.25, shear/25], dtype=torch.float32, device=images.device).unsqueeze(0).repeat(images.shape[0], 1)

            with torch.cuda.amp.autocast():
                images_pred = model.predict(images, action)
                loss = F.mse_loss(images_pred, images_aug)

            optimiser.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()

            epoch_train_losses[i] = loss.detach()
        
        with torch.no_grad():
            epoch_val_losses = torch.zeros(len(val_loader), device=next(model.parameters()).device)
            for i, (images, _) in enumerate(val_loader):
                images = images.to(device)
                act_p = torch.rand(5)
                angle = torch.rand(1).item() * 360 - 180 if act_p[0] > 0.75 else 0
                translate_x = torch.randint(-8, 9, (1,)).item() if act_p[1] > 0.75 else 0
                translate_y = torch.randint(-8, 9, (1,)).item() if act_p[2] > 0.75 else 0
                scale = torch.rand(1).item() * 0.5 + 0.75 if act_p[3] > 0.75 else 1.0
                shear = torch.rand(1).item() * 50 - 25 if act_p[4] > 0.75 else 0
                images_aug = F_v2.affine(images, angle=angle, translate=(translate_x, translate_y), scale=scale, shear=shear)
                action = torch.tensor([angle/180, translate_x/8, translate_y/8, (scale-1.0)/0.25, shear/25], dtype=torch.float32, device=images.device).unsqueeze(0).repeat(images.shape[0], 1)

                with torch.cuda.amp.autocast():
                    images_pred = model.predict(images, action)
                    loss = F.mse_loss(images_pred, images_aug)

                epoch_val_losses[i] = loss.detach()
        
        last_train_loss = epoch_train_losses.mean().item()
        last_val_loss = epoch_val_losses.mean().item()
        postfix = {'train_loss': last_train_loss, 'val_loss': last_val_loss}
        if last_val_loss < best_val_loss and save_dir is not None and epoch % save_every == 0:
            best_val_loss = last_val_loss
            torch.save(model.state_dict(), save_dir)

        if writer is not None:
            writer.add_scalar('Encoder/train_loss', last_train_loss, epoch)
            writer.add_scalar('Encoder/val_loss', last_val_loss, epoch)