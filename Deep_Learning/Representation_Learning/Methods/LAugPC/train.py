import math
import torch
import torch.nn.functional as F
import torchvision.transforms.v2.functional as F_v2
from tqdm import tqdm
from Deep_Learning.Representation_Learning.Methods.LAugPC.model import LAugPC
from Deep_Learning.Representation_Learning.Utils.functional import smooth_l1_loss
from Deep_Learning.Representation_Learning.Examples.MNIST.mnist_linear_1k import single_step_classification_eval, get_ss_mnist_loaders

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from Utils.dataset import PreloadedDataset

def train(
        online_model,
        optimiser,
        train_dataset,
        val_dataset,
        num_epochs,
        batch_size,
        beta=None,
        tau_base=0.996,
        aug_scaler='none',
        learn_on_ss=False,
        normalise=False,
        writer=None,
        save_dir=None,
        save_every=1,
):
    # Initialise target model and its EMA schedule
    target_model = online_model.copy()
    taus = 1 - (1 - tau_base) * ((torch.arange(0, num_epochs, 1) * math.pi / num_epochs).cos() + 1) / 2

    # Initialise augmentation probabilty schedule
    assert aug_scaler in ['linear', 'exp', 'none'], 'aug_scaler must be one of ["linear", "exp"]'
    if aug_scaler == 'linear':
        aug_ps = torch.linspace(0, 0.25, num_epochs)
    elif aug_scaler == 'exp':
        aug_ps = 0.25 * (1.0 - torch.exp(torch.linspace(0, -5, num_epochs)))
    elif aug_scaler == 'none':
        aug_ps = 0.25 * torch.ones(num_epochs)

    device = next(online_model.parameters()).device
    ss_train_loader, ss_val_loader = get_ss_mnist_loaders(batch_size, device)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    scaler = torch.cuda.amp.GradScaler()

    train_options = {
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'beta': beta,
        'tau_base': tau_base,
        'aug_scaler': aug_scaler,
        'learn_on_ss': learn_on_ss,
        'normalise': normalise,
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
        train_dataset.apply_transform(batch_size=batch_size)
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        loop.set_description(f'Epoch [{epoch}/{num_epochs}]')
        if epoch > 0:
            loop.set_postfix(postfix)

        # Training Pass
        epoch_train_losses = torch.zeros(len(train_loader), device=device)
        for i, (images, _) in loop:
    
            # Create Target Images and Action vectors
            angle = torch.rand(1).item() * 360 - 180 if torch.rand(1).item() < aug_ps[epoch] else 0
            translate_x = torch.randint(-8, 9, (1,)).item() if torch.rand(1).item() < aug_ps[epoch] else 0
            translate_y = torch.randint(-8, 9, (1,)).item() if torch.rand(1).item() < aug_ps[epoch] else 0
            scale = torch.rand(1).item() * 0.5 + 0.75 if torch.rand(1).item() < aug_ps[epoch] else 1.0
            shear = torch.rand(1).item() * 50 - 25 if torch.rand(1).item() < aug_ps[epoch] else 0
            images_aug = F_v2.affine(images, angle=angle, translate=(translate_x, translate_y), scale=scale, shear=shear)
            action = torch.tensor([angle/180, translate_x/8, translate_y/8, (scale-1.0)/0.25, shear/25], dtype=torch.float32, device=images.device).unsqueeze(0).repeat(images.shape[0], 1)

            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    z_target = target_model(images_aug)
                z_pred = online_model.predict(images, action)

                # Normalize
                if normalise:
                    z_pred = F.normalize(z_pred, dim=-1)
                    z_target = F.normalize(z_target, dim=-1)

                if beta is None:
                    loss = F.mse_loss(z_pred, z_target)
                else:
                    loss = smooth_l1_loss(z_pred, z_target, beta)

            # Update online model
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            optimiser.zero_grad(set_to_none=True)

            # Update target model
            for o_param, t_param in zip(online_model.parameters(), target_model.parameters()):
                t_param.data = taus[epoch] * t_param.data + (1 - taus[epoch]) * o_param.data

            epoch_train_losses[i] = loss.detach()
        
        # Validation Pass
        with torch.no_grad():
            epoch_val_losses = torch.zeros(len(val_loader), device=next(online_model.parameters()).device)
            for i, (images, _) in enumerate(val_loader):

                # Create Target Image and Action vector
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
                    z_pred = online_model.predict(images, action)

                    # Normalize
                    z_pred = F.normalize(z_pred, dim=-1)
                    z_target = F.normalize(z_target, dim=-1)

                    if beta is None:
                        loss = F.mse_loss(z_pred, z_target)
                    else:
                        loss = smooth_l1_loss(z_pred, z_target, beta)

                epoch_val_losses[i] = loss.detach()

        # single step linear classification eval
        ss_val_acc, ss_val_loss = single_step_classification_eval(online_model, ss_train_loader, ss_val_loader, scaler, learn_on_ss)
        if learn_on_ss:
            scaler.step(optimiser)
            scaler.update()
            optimiser.zero_grad(set_to_none=True)

        last_train_loss = epoch_train_losses.mean().item()
        last_val_loss = epoch_val_losses.mean().item()
        postfix = {'train_loss': last_train_loss, 'val_loss': last_val_loss}
        if writer is not None:
            writer.add_scalar('Encoder/train_loss', last_train_loss, epoch)
            writer.add_scalar('Encoder/val_loss', last_val_loss, epoch)
            writer.add_scalar('Encoder/1step_val_acc', ss_val_acc, epoch)
            writer.add_scalar('Encoder/1step_val_loss', ss_val_loss, epoch)
        
        if ss_val_loss < best_val_loss and save_dir is not None and epoch % save_every == 0:
            best_val_loss = ss_val_loss
            torch.save(online_model.state_dict(), save_dir)