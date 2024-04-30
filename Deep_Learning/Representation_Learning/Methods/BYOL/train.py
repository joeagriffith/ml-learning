import torch
import torch.nn.functional as F
from tqdm import tqdm


from Deep_Learning.Representation_Learning.Utils.functional import smooth_l1_loss
from Deep_Learning.Representation_Learning.Examples.MNIST.mnist_linear_1k import single_step_classification_eval, get_ss_mnist_loaders


def train(
        online_model,
        optimiser,
        train_dataset,
        val_dataset,
        num_epochs,
        batch_size,
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
    # Initialise target model and its EMA schedule
    target_model = online_model.copy()
    taus = torch.linspace(tau_0, tau_e, tau_T)
    taus_end = torch.ones(num_epochs - tau_T) * tau_e
    taus = torch.cat([taus, taus_end])

    device = next(online_model.parameters()).device
    ss_train_loader, ss_val_loader = get_ss_mnist_loaders(batch_size, device)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    scaler = torch.cuda.amp.GradScaler()

    train_options = {
        'num_epochs': num_epochs,
        'batch_size': batch_size,
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
        train_dataset.apply_transform(batch_size=batch_size)
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        loop.set_description(f'Epoch [{epoch}/{num_epochs}]')
        if epoch > 0:
            loop.set_postfix(postfix)

        # Training Pass
        epoch_train_losses = torch.zeros(len(train_loader), device=device)
        for i, (images, _) in loop:
            with torch.cuda.amp.autocast():
                x1, x2 = augmentation(images), augmentation(images)
                y1, y2 = target_model(x1), online_model(x2)
                z1, z2 = target_model.project(y1), online_model.project(y2)
                p1, p2 = target_model.predict(z1), online_model.predict(z2)

                if normalise:
                    z1, z2 = F.normalize(z1, dim=-1), F.normalize(z2, dim=-1)
                    p1, p2 = F.normalize(p1, dim=-1), F.normalize(p2, dim=-1)

                if beta is None:
                    loss = 0.5 * (F.mse_loss(p1, z2.detach()) + F.mse_loss(p2, z1.detach()))
                else:
                    loss = 0.5 * (smooth_l1_loss(p1, z2.detach(), beta) + smooth_l1_loss(p2, z1.detach(), beta))


            # Update online model
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            optimiser.zero_grad(set_to_none=True)

            # Update target model
            with torch.no_grad():
                for o_param, t_param in zip(online_model.parameters(), target_model.parameters()):
                    t_param.data = taus[epoch] * t_param.data + (1 - taus[epoch]) * o_param.data

            epoch_train_losses[i] = loss.detach()
        
        # Validation Pass
        with torch.no_grad():
            epoch_val_losses = torch.zeros(len(val_loader), device=next(online_model.parameters()).device)
            for i, (images, _) in enumerate(val_loader):
                with torch.cuda.amp.autocast():
                    x1, x2 = augmentation(images), augmentation(images)
                    y1, y2 = target_model(x1), online_model(x2)
                    z1, z2 = target_model.project(y1), online_model.project(y2)
                    p1, p2 = target_model.predict(z1), online_model.predict(z2)

                    if normalise:
                        z1, z2 = F.normalize(z1, dim=-1), F.normalize(z2, dim=-1)
                        p1, p2 = F.normalize(p1, dim=-1), F.normalize(p2, dim=-1)

                    if beta is None:
                        loss = 0.5 * (F.mse_loss(p1, z2.detach()) + F.mse_loss(p2, z1.detach()))
                    else:
                        loss = 0.5 * (smooth_l1_loss(p1, z2.detach(), beta) + smooth_l1_loss(p2, z1.detach(), beta))

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