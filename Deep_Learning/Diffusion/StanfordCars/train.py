from tqdm import tqdm
import torch
import torch.nn.functional as F
from Deep_Learning.Diffusion.StanfordCars.utils import (
    load_checkpoint,    
    save_checkpoint,
    check_accuracy,
    save_predictions_as_imgs,
)
from Deep_Learning.Diffusion.StanfordCars.model import DiffusionUNet
from Deep_Learning.Diffusion.StanfordCars.functional import get_loss


def train(
        model, 
        epochs, 
        train_loader, 
        optimiser, 
        loss_fn, 
        save_model=False, 
        device="cuda"
    ):

    for epoch in range(epochs):
        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        
        for batch_idx, images in loop:
            images = images.to(device)

            t = torch.randint(0, model.T, (images.shape[0],), device=images.device).long()
            # Forward
            with torch.cuda.amp.autocast():
                loss = get_loss(model, images, t, loss_fn, model.sqrt_alphas_cumprod, model.sqrt_one_minus_alphas_cumprod)

            # Backward
            optimiser.zero_grad()

            # Gradient descent
            optimiser.step()

            # Update tqdm loop
            loop.set_postfix(loss=loss.item())
            loop.set_description(f"Epoch [{epoch}/{epochs}]")
            
        
        # save model
        if save_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimiser": optimiser.state_dict(),
            }
            save_checkpoint(checkpoint)

        # # check accuracy
        # check_accuracy(val_loader, model, device)

        # # print some examples to a folder
        # save_predictions_as_imgs(
        #     val_loader, model, folder="Deep_learning/UNets/image_segmentation/saved_images/", device=device
        # )