import torch
import torch.nn.functional as F
import torchvision.transforms.v2.functional as F_v2
from tqdm import tqdm


def train(
        online_model,
        train_dataset,
        val_dataset,
        num_epochs,
        batch_size,
        lr,
        wd,
        beta=0.996,
        writer=None,
        save_dir=None,
        save_every=1,
):
    # Exclude bias and batch norm parameters from weight decay
    decay_parameters = [param for name, param in online_model.named_parameters() if 'weight' in name]
    decay_parameters = [{'params': decay_parameters}]
    non_decay_parameters = [param for name, param in online_model.named_parameters() if 'weight' not in name]
    non_decay_parameters = [{'params': non_decay_parameters, 'weight_decay': 0.0}]
    optimiser = torch.optim.AdamW(decay_parameters + non_decay_parameters, lr=lr, weight_decay=wd)

    target_model = online_model.copy()
    betas = torch.linspace(beta, 1.0, num_epochs)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
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

        epoch_train_losses = torch.zeros(len(train_loader), device=next(online_model.parameters()).device)
        for i, (images, _) in loop:
    
            angle = torch.rand(1).item() * 360 - 180 if torch.rand(1).item() > 0.75 else 0
            translate_x = torch.randint(-8, 9, (1,)).item() if torch.rand(1).item() > 0.75 else 0
            translate_y = torch.randint(-8, 9, (1,)).item() if torch.rand(1).item() > 0.75 else 0
            scale = torch.rand(1).item() * 0.5 + 0.75 if torch.rand(1).item() > 0.75 else 1.0
            shear = torch.rand(1).item() * 50 - 25 if torch.rand(1).item() > 0.75 else 0
            images_aug = F_v2.affine(images, angle=angle, translate=(translate_x, translate_y), scale=scale, shear=shear)
            action = torch.tensor([angle/180, translate_x/8, translate_y/8, (scale-1.0)/0.25, shear/25], dtype=torch.float32, device=images.device).unsqueeze(0).repeat(images.shape[0], 1)

            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    z_target = target_model(images_aug)
                z = online_model(images)
                z_pred = online_model.predict(z, action)
                loss = F.mse_loss(z_pred, z_target)

            optimiser.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()

            for o_param, t_param in zip(online_model.parameters(), target_model.parameters()):
                t_param.data = betas[epoch] * t_param.data + (1 - betas[epoch]) * o_param.data

            epoch_train_losses[i] = loss.detach()
        
        with torch.no_grad():
            epoch_val_losses = torch.zeros(len(val_loader), device=next(online_model.parameters()).device)
            for i, (images, _) in enumerate(val_loader):
                act_p = torch.rand(5)
                angle = torch.rand(1).item() * 360 - 180 if act_p[0] > 0.75 else 0
                translate_x = torch.randint(-8, 9, (1,)).item() if act_p[1] > 0.75 else 0
                translate_y = torch.randint(-8, 9, (1,)).item() if act_p[2] > 0.75 else 0
                scale = torch.rand(1).item() * 0.5 + 0.75 if act_p[3] > 0.75 else 1.0
                shear = torch.rand(1).item() * 50 - 25 if act_p[4] > 0.75 else 0
                images_aug = F_v2.affine(images, angle=angle, translate=(translate_x, translate_y), scale=scale, shear=shear)
                action = torch.tensor([angle/180, translate_x/8, translate_y/8, (scale-1.0)/0.25, shear/25], dtype=torch.float32, device=images.device).unsqueeze(0).repeat(images.shape[0], 1)

                with torch.cuda.amp.autocast():
                    z_target = target_model(images_aug).detach()
                    z = online_model(images)
                    z_pred = online_model.predict(z, action)
                    loss = F.mse_loss(z_pred, z_target)

                epoch_val_losses[i].append(loss.detach())
        
        last_train_loss = epoch_train_losses.mean().item()
        last_val_loss = epoch_val_losses.mean().item()
        postfix = {'train_loss': last_train_loss, 'val_loss': last_val_loss}
        if writer is not None:
            writer.add_scalar('Encoder/train_loss', last_train_loss, epoch)
            writer.add_scalar('Encoder/val_loss', last_val_loss, epoch)
        
        if last_val_loss < best_val_loss and save_dir is not None and epoch % save_every == 0:
            best_val_loss = last_val_loss
            torch.save(online_model.state_dict(), save_dir)
