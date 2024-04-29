import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from Utils.dataset import PreloadedDataset
from tqdm import tqdm


from Deep_Learning.Representation_Learning.Utils.functional import smooth_l1_loss, negative_cosine_similarity
from Deep_Learning.Representation_Learning.Examples.MNIST.mnist_linear_1k import single_step_classification_eval


def train(
        model,
        train_dataset,
        val_dataset,
        num_epochs,
        batch_size,
        lr,
        wd,
        augmentation,
        beta=None,
        normalise=True,
        learn_on_ss=False,
        writer=None,
        save_dir=None,
        save_every=1,
):
    # # Prepare data for single step classification eval
    # Load data
    device = next(model.parameters()).device
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

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    optimiser = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    scaler = torch.cuda.amp.GradScaler()

    train_options = {
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'lr': lr,
        'wd': wd,
        'augmentation': str(augmentation),
        'beta': beta,
        'normalise': normalise,
        'learn_on_ss': learn_on_ss,
    }
    if writer is not None:
        writer.add_text('Encoder/options', str(train_options))
        writer.add_text('Encoder/model', str(model).replace('\n', '<br/>').replace(' ', '&nbsp;'))
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

        epoch_train_losses = torch.zeros(len(train_loader), device=device)
        for i, (images, _) in loop:
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    x1, x2 = augmentation(images), augmentation(images)

                z1, z2 = model(x1), model(x2)
                p1, p2 = model.project(z1), model.project(z2)
                y1, y2 = model.predict(p1), model.predict(p2)

                # loss = negative_cosine_similarity(y1, p2.detach()) / 2 + negative_cosine_similarity(y2, p1.detach()) / 2
                loss = F.mse_loss(F.normalize(y1), F.normalize(p2.detach())) / 2 + F.mse_loss(F.normalize(y2), F.normalize(p1.detach())) / 2

            optimiser.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()

            epoch_train_losses[i] = loss.detach()
        
        with torch.no_grad():
            epoch_val_losses = torch.zeros(len(val_loader), device=device)
            for i, (images, _) in enumerate(val_loader):
                with torch.cuda.amp.autocast():
                    x1, x2 = augmentation(images), augmentation(images)
                    z1, z2 = model(x1), model(x2)
                    p1, p2 = model.project(z1), model.project(z2)
                    y1, y2 = model.predict(p1), model.predict(p2)
                    # loss = negative_cosine_similarity(y1, p2.detach()) / 2 + negative_cosine_similarity(y2, y1.detach()) / 2
                    loss = F.mse_loss(F.normalize(y1), F.normalize(p2.detach())) / 2 + F.mse_loss(F.normalize(y2), F.normalize(p1.detach())) / 2

                epoch_val_losses[i] = loss.detach()

        # single step linear classification eval
        if learn_on_ss:
            optimiser.zero_grad(set_to_none=True)
        ss_val_acc, ss_val_loss = single_step_classification_eval(model, ss_train_loader, ss_val_loader, scaler, learn_on_ss)
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
            writer.add_scalar('Encoder/1step_val_loss', ss_val_loss, epoch)

        if ss_val_acc < best_val_loss and save_dir is not None and epoch % save_every == 0:
            best_val_loss = ss_val_acc
            torch.save(model.state_dict(), save_dir)