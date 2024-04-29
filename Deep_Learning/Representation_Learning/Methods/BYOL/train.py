import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from Utils.dataset import PreloadedDataset
from tqdm import tqdm


from Deep_Learning.Representation_Learning.Utils.functional import smooth_l1_loss
from Deep_Learning.Representation_Learning.Examples.MNIST.mnist_linear_1k import single_step_classification_eval


def train(
        online_model,
        train_dataset,
        val_dataset,
        num_epochs,
        batch_size,
        lr,
        wd,
        augmentation,
        beta=None,
        tau_0=0.996,
        tau_e=0.999,
        tau_T=100,
        normalise=True,
        learn_on_ss=False,
        writer=None,
        save_dir=None,
        save_every=1,
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

    target_model = online_model.copy()

    taus = torch.linspace(tau_0, tau_e, tau_T)
    taus_end = torch.ones(num_epochs - tau_T) * tau_e
    taus = torch.cat([taus, taus_end])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    optimiser = torch.optim.AdamW(online_model.parameters(), lr=lr, weight_decay=wd)
    scaler = torch.cuda.amp.GradScaler()

    train_options = {
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'lr': lr,
        'wd': wd,
        'augmentation': str(augmentation),
        'tau_0': tau_0,
        'tau_e': tau_e,
        'tau_T': tau_T,
        'beta': beta,
        'normalise': normalise,
        'learn_on_ss': learn_on_ss,
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
                    v1 = augmentation(images)
                    v2 = augmentation(images)

                    target_y = target_model(v2)
                    target_z = target_model.project(target_y)

                online_y = online_model(v1)
                online_z = online_model.project(online_y)
                pred = online_model.predict(online_z)

                if normalise:
                    pred = F.normalize(pred, dim=-1)
                    target_z = F.normalize(target_z, dim=-1)
                if beta is None:
                    loss = 0.5 * F.mse_loss(pred, target_z)
                else:
                    loss = 0.5 * smooth_l1_loss(pred, target_z, beta)

                # Symmetrize
                with torch.no_grad():
                    target_y = target_model(v1)
                    target_z = target_model.project(target_y)

                online_y = online_model(v2)
                online_z = online_model.project(online_y)
                pred = online_model.predict(online_z)

                if normalise:
                    pred = F.normalize(pred, dim=-1)
                    target_z = F.normalize(target_z, dim=-1)
                if beta is None:
                    loss += 0.5 * F.mse_loss(pred, target_z)
                else:
                    loss += 0.5 * smooth_l1_loss(pred, target_z, beta)


            optimiser.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()

            with torch.no_grad():
                for o_param, t_param in zip(online_model.parameters(), target_model.parameters()):
                    t_param.data = taus[epoch] * t_param.data + (1 - taus[epoch]) * o_param.data

            epoch_train_losses[i] = loss.detach()
        
        with torch.no_grad():
            epoch_val_losses = torch.zeros(len(val_loader), device=next(online_model.parameters()).device)
            for i, (images, _) in enumerate(val_loader):
                with torch.cuda.amp.autocast():
                    v1 = augmentation(images)
                    v2 = augmentation(images)

                    target_y = target_model(v2)
                    target_z = target_model.project(target_y)

                    online_y = online_model(v1)
                    online_z = online_model.project(online_y)
                    pred = online_model.predict(online_z)

                    if normalise:
                        pred = F.normalize(pred, dim=-1)
                        target_z = F.normalize(target_z, dim=-1)
                    if beta is None:
                        loss = 0.5 * F.mse_loss(pred, target_z)
                    else:
                        loss = 0.5 * smooth_l1_loss(pred, target_z, beta)

                    # Symmetrize
                    with torch.no_grad():
                        target_y = target_model(v1)
                        target_z = target_model.project(target_y)

                    online_y = online_model(v2)
                    online_z = online_model.project(online_y)
                    pred = online_model.predict(online_z)

                    if normalise:
                        pred = F.normalize(pred, dim=-1)
                        target_z = F.normalize(target_z, dim=-1)
                    if beta is None:
                        loss += 0.5 * F.mse_loss(pred, target_z)
                    else:
                        loss += 0.5 * smooth_l1_loss(pred, target_z, beta)

                epoch_val_losses[i] = loss.detach()

        # single step linear classification eval
        if learn_on_ss:
            optimiser.zero_grad(set_to_none=True)
        ss_val_acc, ss_val_loss = single_step_classification_eval(online_model, ss_train_loader, ss_val_loader, scaler, learn_on_ss)
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

        if ss_val_loss < best_val_loss and save_dir is not None and epoch % save_every == 0:
            best_val_loss = ss_val_loss
            torch.save(online_model.state_dict(), save_dir)