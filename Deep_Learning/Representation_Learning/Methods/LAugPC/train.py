import torch
import torch.nn.functional as F
import torchvision.transforms.v2.functional as F_v2
from tqdm import tqdm
from Deep_Learning.Representation_Learning.Methods.LAugPC.model import LAugPC
from Deep_Learning.Representation_Learning.Utils.functional import smooth_l1_loss

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from Utils.dataset import PreloadedDataset

def train(
        online_model,
        train_dataset,
        val_dataset,
        num_epochs,
        batch_size,
        lr,
        wd,
        beta=None,
        tau_0=0.996,
        tau_e=0.999,
        tau_T=100,
        writer=None,
        save_dir=None,
        save_every=1,
        aug_scaler='none',
        learn_on_ss=False,
):
    # # Prepare data for single step classification eval
    # Load data
    device = next(online_model.parameters()).device
    t_dataset = datasets.MNIST(root='../Datasets/', train=False, transform=transforms.ToTensor(), download=True)
    dataset = datasets.MNIST(root='../Datasets/', train=True, transform=transforms.ToTensor(), download=True)
    train1k = PreloadedDataset.from_dataset(dataset, transforms.ToTensor(), device)
    test = PreloadedDataset.from_dataset(t_dataset, transforms.ToTensor(), device)
    # Reduce to 1000 samples, 100 from each class.
    indices = []
    for i in range(10):
        idx = train1k.targets == i
        indices.append(torch.where(idx)[0][:100])
    indices = torch.cat(indices)
    train1k.images = train1k.images[indices]
    train1k.transformed_images = train1k.transformed_images[indices]
    train1k.targets = train1k.targets[indices]
    # Build data loaders
    ss_train_loader = DataLoader(train1k, batch_size=100, shuffle=True)
    ss_val_loader = DataLoader(test, batch_size=batch_size, shuffle=False)


    # Exclude bias and batch norm parameters from weight decay
    decay_parameters = [param for name, param in online_model.named_parameters() if 'weight' in name]
    decay_parameters = [{'params': decay_parameters}]
    non_decay_parameters = [param for name, param in online_model.named_parameters() if 'weight' not in name]
    non_decay_parameters = [{'params': non_decay_parameters, 'weight_decay': 0.0}]
    optimiser = torch.optim.AdamW(decay_parameters + non_decay_parameters, lr=lr, weight_decay=wd)
    # optimiser = torch.optim.AdamW(online_model.parameters(), lr=lr, weight_decay=wd)

    target_model = online_model.copy()

    taus = torch.linspace(tau_0, tau_e, tau_T)
    taus_end = torch.ones(num_epochs - tau_T) * tau_e
    taus = torch.cat([taus, taus_end])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    scaler = torch.cuda.amp.GradScaler()

    train_options = {
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'lr': lr,
        'wd': wd,
        'beta': beta,
        'aug_scaler': aug_scaler,
    }
    if writer is not None:
        writer.add_text('Encoder/options', str(train_options))
        writer.add_text('Encoder/model', str(online_model).replace('\n', '<br/>').replace(' ', '&nbsp;'))
        writer.add_text('Encoder/optimiser', str(optimiser).replace('\n', '<br/>').replace(' ', '&nbsp;'))

    assert aug_scaler in ['linear', 'exp', 'none'], 'aug_scaler must be one of ["linear", "exp"]'
    if aug_scaler == 'linear':
        aug_ps = torch.linspace(0, 0.25, num_epochs)
    elif aug_scaler == 'exp':
        aug_ps = 0.25 * (1.0 - torch.exp(torch.linspace(0, -5, num_epochs)))
    elif aug_scaler == 'none':
        aug_ps = 0.25 * torch.ones(num_epochs)

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
    
            angle = torch.rand(1).item() * 360 - 180 if torch.rand(1).item() < aug_ps[epoch] else 0
            translate_x = torch.randint(-8, 9, (1,)).item() if torch.rand(1).item() < aug_ps[epoch] else 0
            translate_y = torch.randint(-8, 9, (1,)).item() if torch.rand(1).item() < aug_ps[epoch] else 0
            scale = torch.rand(1).item() * 0.5 + 0.75 if torch.rand(1).item() < aug_ps[epoch] else 1.0
            shear = torch.rand(1).item() * 50 - 25 if torch.rand(1).item() < aug_ps[epoch] else 0
            action = torch.tensor([angle/180, translate_x/8, translate_y/8, (scale-1.0)/0.25, shear/25], dtype=torch.float32, device=images.device).unsqueeze(0).repeat(images.shape[0], 1)

            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    images_aug = F_v2.affine(images, angle=angle, translate=(translate_x, translate_y), scale=scale, shear=shear)
                    z_target = target_model(images_aug)
                    # z_target = online_model(images_aug)
                z_pred = online_model.predict(images, action)

                # Normalize
                z_pred = F.normalize(z_pred, dim=-1)
                z_target = F.normalize(z_target, dim=-1)

                if beta is None:
                    loss = F.mse_loss(z_pred, z_target)
                else:
                    loss = smooth_l1_loss(z_pred, z_target, beta)

            optimiser.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()

            for o_param, t_param in zip(online_model.parameters(), target_model.parameters()):
                t_param.data = taus[epoch] * t_param.data + (1 - taus[epoch]) * o_param.data

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
                action = torch.tensor([angle/180, translate_x/8, translate_y/8, (scale-1.0)/0.25, shear/25], dtype=torch.float32, device=images.device).unsqueeze(0).repeat(images.shape[0], 1)

                with torch.cuda.amp.autocast():
                    images_aug = F_v2.affine(images, angle=angle, translate=(translate_x, translate_y), scale=scale, shear=shear)
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
        if learn_on_ss:
            optimiser.zero_grad(set_to_none=True)
        ss_val_acc = single_step_classification_eval(online_model, ss_train_loader, ss_val_loader, scaler, learn_on_ss)
        if learn_on_ss:
            scaler.step(optimiser)
            scaler.update()

        
        last_train_loss = epoch_train_losses.mean().item()
        last_val_loss = epoch_val_losses.mean().item()
        postfix = {'train_loss': last_train_loss, 'val_loss': last_val_loss}
        if writer is not None:
            writer.add_scalar('Encoder/train_loss', last_train_loss, epoch)
            writer.add_scalar('Encoder/val_loss', last_val_loss, epoch)
            writer.add_scalar('Encoder/1step_val_acc', ss_val_acc, epoch)
        
        if last_val_loss < best_val_loss and save_dir is not None and epoch % save_every == 0:
            best_val_loss = last_val_loss
            torch.save(online_model.state_dict(), save_dir)

def single_step_classification_eval(
        encoder,
        train_loader,
        val_loader,
        scaler,
        learn_encoder=False,
):
    encoder.eval()
    device = next(encoder.parameters()).device

    classifier = torch.nn.Linear(encoder.num_features, 10, bias=False).to(device)
    optimiser = torch.optim.AdamW(classifier.parameters(), lr=1e-1, weight_decay=1e-4)

    for i, (images, labels) in enumerate(train_loader):
        with torch.cuda.amp.autocast():
            if learn_encoder:
                z = encoder(images)        
            else:
                with torch.no_grad():
                    z = encoder(images)

            y_pred = classifier(z)
            loss = F.cross_entropy(y_pred, labels)

        optimiser.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()

    val_accs = torch.zeros(len(val_loader), device=device)
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    z = encoder(images)        
                y_pred = classifier(z)
            val_accs[i] = (y_pred.argmax(dim=1) == labels).float().mean()

    val_acc = val_accs.mean().item()

    return val_acc