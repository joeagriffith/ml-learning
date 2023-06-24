import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from Deep_Learning.Experimental.Boltzmann.transforms import add_salt_and_pepper_noise

def eval(
        model,
        data_loader,
        criterion,
        model_params,
):
    loss = 0.0
    model_params.append(True) # Unlock visible units during inference
    for batch_idx, (images, y) in data_loader:
        images = images.flatten(start_dim=1)
        noised = add_salt_and_pepper_noise(images, p=0.1)

        reconstruction = model(noised, *model_params)[0]
        loss += criterion(reconstruction, images)
    
    loss /= len(data_loader)[0]
    return loss


def train(
        model,
        train_loader,
        eval_criterion,
        learning_rate=0.01,
        epochs=20,
        neg_every=1,
        neg_num=10000,
        model_params=[100, (1.0, 5.0), False],
        # model_steps=100,
        # model_temp_range=(1.0, 5.0),
        # model_replacement=False
        eval_every=1,
        step=0
):
    writer = SummaryWriter("Deep_Learning/Experimental/Boltzmann/mnist/out/logs")

    losses = []
    loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)

    for epoch in range(epochs):

        for batch_idx, (images, y) in loop:
            images = images.flatten(start_dim=1)
            state = model(images, *model_params)
            model.update(state, learning_rate)

        # Negative phase, unlearn from random initialisation
        if epoch % neg_every == 0:
            state = model(None, *model_params)
            model.update(state, learning_rate, True)

        if epoch % eval_every == 0:
            losses.append(eval(model, train_loader, eval_criterion, model_params))
            writer.add_scalar("Reconstruction Loss", losses[-1], step)
        
        step += len(images)

    return losses, step

        

        




