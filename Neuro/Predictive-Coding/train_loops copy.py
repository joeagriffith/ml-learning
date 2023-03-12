import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from my_funcs import topk_accuracy, evaluate
    
def train(
    model, 
    train_dataset,
    val_dataset,
    optimiser, 
    criterion, 
    model_name, 
    num_epochs, 
    flatten=False, 
    log_dir="logs", 
    step=0, 
    batch_size=100,
    device="cpu"
):
    writer = SummaryWriter(f"{log_dir}/{model_name}") # we create the log inside a folder for each model
    val_loss = 9999999.9
    val_acc = 0.0
    
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)

    for epoch in range(num_epochs):
        
        model.train()

        epoch_train_loss = 0.0
        epoch_train_acc = torch.zeros(3, device=device)

        for _, (images, y) in train_loader:
            x = images.to(device)
            if flatten:
                x = torch.flatten(x, start_dim=1)
            target = y.to(device)
            out = model(x)

            loss = criterion(out, target)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            with torch.no_grad():
                epoch_train_loss += loss.item()
                epoch_train_acc += topk_accuracy(out, target, (1,3,5))

        epoch_train_loss /= len(train_loader)
        epoch_train_acc /= len(train_loader)
        
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, flatten)

        step += len(train_dataset)
        writer.add_scalar("Training Loss", epoch_train_loss, step)
        writer.add_scalar("Training Accuracy Top1", epoch_train_acc[0], step)
        writer.add_scalar("Training Accuracy Top3", epoch_train_acc[1], step)
        writer.add_scalar("Training Accuracy Top5", epoch_train_acc[2], step)
        writer.add_scalar("Validation Loss", val_loss[-1], step)
        writer.add_scalar("Validation Accuracy Top1", val_acc[0], step)
        writer.add_scalar("Validation Accuracy Top3", val_acc[1], step)
        writer.add_scalar("Validation Accuracy Top5", val_acc[2], step)
        
    return step