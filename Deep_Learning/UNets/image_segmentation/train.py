from tqdm import tqdm
import torch
from Deep_Learning.UNets.image_segmentation.utils import (
    load_checkpoint,    
    save_checkpoint,
    check_accuracy,
    save_predictions_as_imgs,
)

def train(epochs, train_loader, val_loader, model, optimiser, loss_fn, scaler, save_model=False, device="cuda"):

    for epoch in range(epochs):
        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        
        for batch_idx, (images, targets) in loop:
            images = images.to(device)
            targets = targets.float().unsqueeze(1).to(device)
        
            # Forward
            with torch.cuda.amp.autocast():
                predictions = model(images)
                loss = loss_fn(predictions, targets)

            # Backward
            optimiser.zero_grad()
            scaler.scale(loss).backward()

            # Gradient descent
            scaler.step(optimiser)
            scaler.update()

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

        # check accuracy
        check_accuracy(val_loader, model, device)

        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, folder="Deep_learning/UNets/image_segmentation/saved_images/", device=device
        )