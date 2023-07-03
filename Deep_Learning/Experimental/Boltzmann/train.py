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
    print(f'model_params at pre-eval: {model_params}')
    # model_params.append(True) # Unlock visible units during inference
    for batch_idx, (images, y) in enumerate(data_loader):
        images = images.flatten(start_dim=1)
        noised = add_salt_and_pepper_noise(images, p=0.1)

        reconstruction = model(noised, *model_params, True)[0]
        loss += criterion(reconstruction, images)
    
    loss /= batch_idx+1
    return loss


def train(
        model,
        positive_dataloader,
        negative_dataloader,
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

    for epoch in range(epochs):
        loop = tqdm(enumerate(positive_dataloader), total=len(positive_dataloader), leave=False)

        do_neg = epoch % neg_every == 0
        if do_neg:
            neg_it = iter(negative_dataloader)

        for batch_idx, (images, y) in loop:
            images = images.flatten(start_dim=1)

            for phase in range(do_neg+1):
                if phase == 1:
                    images = next(neg_it)[0].flatten(start_dim=1)

                state = model(images, *model_params)
                model.update(state, learning_rate, phase==1)

        if epoch % eval_every == 0:
            losses.append(eval(model, positive_dataloader, eval_criterion, model_params))
            writer.add_scalar("Reconstruction Loss", losses[-1], step)
        
        step += len(images)

    return losses, step

        

        




