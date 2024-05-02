import torch
import torch.nn.functional as F
from tqdm import tqdm

from Deep_Learning.Representation_Learning.Utils.functional import cosine_schedule
from Deep_Learning.Representation_Learning.Examples.MNIST.mnist_linear_1k import single_step_classification_eval, get_ss_mnist_loaders

def H(s, t, tmp_s, tmp_t, C):
    """
    Compute the loss for the given batch of samples.

    Args:
        s: torch.Tensor, projected output of the student model.
        t: torch.Tensor, projected output of the teacher model.
        tmp_s: float, the temperature scaling factor for the student.
        tmp_t: float, the temperature scaling factor for the teacher.
        C: torch.Tensor, centering term for the teacher.
    """
    s = F.softmax(s / tmp_s, dim=-1)
    t = t.detach()
    t = F.softmax((t-C) / tmp_t, dim=-1)
    return - (t * s.log()).sum(dim=-1).mean()

def train(
        student_model,
        optimiser,
        train_dataset,
        val_dataset,
        num_epochs,
        batch_size,
        augmentation,
        learn_on_ss=False,
        writer=None,
        save_dir=None,
        save_every=1,
):
    device = next(student_model.parameters()).device

#============================== Student Learning Parameters ==============================
    # LR schedule, warmup then cosine
    base_lr = optimiser.param_groups[0]['lr'] * batch_size / 256
    end_lr = 1e-6
    warm_up_lrs = torch.linspace(0, base_lr, 10)
    cosine_lrs = cosine_schedule(base_lr, end_lr, num_epochs-10)
    lrs = torch.cat([warm_up_lrs, cosine_lrs])
    assert len(lrs) == num_epochs

    # WD schedule, cosine 
    start_wd = 0.04
    end_wd = 0.4
    wds = cosine_schedule(start_wd, end_wd, num_epochs)
    
#============================== Teacher Learning Parameters ==============================
    # Initialise target model
    teacher_model = student_model.copy()
    # EMA schedule, cosine
    start_tau=0.996,
    end_tau = 1.0,
    taus = cosine_schedule(start_tau, end_tau, num_epochs)

#============================== Loss Parameters ==============================
    # Temperature schedule
    assert num_epochs >= 30, 'num_epochs must be >= 30 because of the temperature schedule'
    tmp_s = 0.1
    tmp_ts = torch.cat([torch.linspace(0.04, 0.07, 30), torch.ones(num_epochs-30) * 0.07])

    # Initialise C
    C = torch.zeros(1, student_model.num_features, device=device)
    C_mom = 0.9

# ============================== Data Handling ==============================
    # Initialise dataloaders for single step classification eval
    ss_train_loader, ss_val_loader = get_ss_mnist_loaders(batch_size, device)

    # Initialise dataloaders for training and validation
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ============================== Training Stuff ==============================
    # Initialise scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    # Log training options
    train_options = {
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'augmentation': str(augmentation),
        'learn_on_ss': learn_on_ss,
    }

    # Log training options, model details, and optimiser details
    if writer is not None:
        writer.add_text('Encoder/options', str(train_options))
        writer.add_text('Encoder/model', str(student_model).replace('\n', '<br/>').replace(' ', '&nbsp;'))
        writer.add_text('Encoder/optimiser', str(optimiser).replace('\n', '<br/>').replace(' ', '&nbsp;'))

    # Initialise training variables
    last_train_loss = -1
    last_val_loss = -1
    best_val_loss = float('inf')
    postfix = {}

# ============================== Training Loop ==============================
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
                # Augment images
                x1, x2 = augmentation(images), augmentation(images)
                
                # Encode images
                s1, s2 = student_model(x1), student_model(x2)
                t1, t2 = teacher_model(x1), teacher_model(x2)

                # Project encodings
                s1_p, s2_p = student_model.project(s1), student_model.project(s2)
                t1_p, t2_p = teacher_model.project(t1), teacher_model.project(t2)

                loss = 0.5 * (H(s1_p, t1_p, tmp_s, tmp_ts[epoch], C) + H(s2_p, t2_p, tmp_s, tmp_ts[epoch], C))

                # Update C
                batch_mean = torch.cat([t1_p, t2_p], dim=0).mean(dim=0)
                C = C_mom * C + (1 - C_mom) * batch_mean

            # Update lr
            for param_group in optimiser.param_groups:
                param_group['lr'] = lrs[epoch]
            # Update wd
            for param_group in optimiser.param_groups:
                if param_group['weight_decay'] != 0:
                    param_group['weight_decay'] = wds[epoch]

            # Update online model
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            optimiser.zero_grad(set_to_none=True)

            # Update teacher
            with torch.no_grad():
                for s_param, t_param in zip(student_model.parameters(), teacher_model.parameters()):
                    t_param.data = taus[epoch] * t_param.data + (1 - taus[epoch]) * s_param.data

            epoch_train_losses[i] = loss.detach()
        
        # Validation Pass
        with torch.no_grad():
            epoch_val_losses = torch.zeros(len(val_loader), device=device)
            for i, (images, _) in enumerate(val_loader):
                with torch.cuda.amp.autocast():
                    # Augment images
                    x1, x2 = augmentation(images), augmentation(images)

                    # Encode images
                    s1, s2 = student_model(x1), student_model(x2)
                    t1, t2 = teacher_model(x1), teacher_model(x2)

                    # Project encodings
                    s1_p, s2_p = student_model.project(s1), student_model.project(s2)
                    t1_p, t2_p = teacher_model.project(t1), teacher_model.project(t2)

                    loss = 0.5 * (H(s1_p, t1_p, tmp_s, tmp_ts[epoch], C) + H(s2_p, t2_p, tmp_s, tmp_ts[epoch], C))

                    epoch_val_losses[i] = loss.detach()

        # single step linear classification eval
        ss_val_acc, ss_val_loss = single_step_classification_eval(student_model, ss_train_loader, ss_val_loader, scaler, learn_on_ss)
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
            torch.save(student_model.state_dict(), save_dir)