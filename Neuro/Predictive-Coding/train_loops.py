import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from my_funcs import topk_accuracy, evaluate, evaluate_pc

    
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
    best="loss",
    save_model=True,
    batch_size=100,
    learning_rate=3e-4,
    weight_decay=1e-2,
    device="cpu"
):
    writer = SummaryWriter(f"{log_dir}/{model_name}")
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    
    #  For determining best model. Either can be used.
    best_val_acc = 0.0
    best_val_loss = 99999999.9

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)

    for epoch in range(num_epochs):
        
        model.train()
        train_dataset.apply_transform()
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)    

        # num_correct = 0
        epoch_train_loss = 0.0
        epoch_train_acc = torch.zeros(3)

        for batch_idx, (images, y) in loop:
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
                epoch_train_acc += torch.tensor(topk_accuracy(out, target, (1,3,5)))

                if epoch > 0:
                    loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
                    loop.set_postfix(
                        train_loss = train_loss[-1], 
                        train_acc = train_acc[-1][0].item(), 
                        val_loss = val_loss[-1], 
                        val_acc = val_acc[-1][0].item(),
                    )

        train_loss.append(epoch_train_loss / len(train_loader))
        train_acc.append(epoch_train_acc / len(train_loader))

        
        val_l, val_a = evaluate(model, val_loader, criterion, device, flatten)
        val_loss.append(val_l)
        val_acc.append(val_a)

            
        if save_model:
            if best == "loss":
                if best_val_loss > val_loss[-1]:
                    best_val_loss = val_loss[-1]
                    torch.save(model.state_dict(), f'models/{model_name}.pth')
            elif best in ["acc", "accuracy"]:
                if best_val_acc < val_acc[-1][0]:
                    best_val_acc = val_acc[-1][0]
                    torch.save(model.state_dict(), f'models/{model_name}.pth')
            else:
                raise Exception("Invalid value for 'best'")

        step += len(train_dataset)
        writer.add_scalar("Training Loss", train_loss[-1], step)
        writer.add_scalar("Training Accuracy Top1", train_acc[-1][0], step)
        writer.add_scalar("Validation Loss", val_loss[-1], step)
        writer.add_scalar("Validation Accuracy Top1", val_acc[-1][0], step)
        writer.add_scalar("Validation Accuracy Top3", val_acc[-1][1], step)
        writer.add_scalar("Validation Accuracy Top5", val_acc[-1][2], step)
        
    return torch.tensor(train_loss), torch.stack(train_acc), torch.tensor(val_loss), torch.stack(val_acc), step

def train_pc(
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
    best="loss",
    save_model=True,
    batch_size=100,
    learning_rate=3e-4,
    weight_decay=1e-2,
    mode="error",
    device="cpu"
):
    writer = SummaryWriter(f"{log_dir}/{model_name}")
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []

    assert mode == "error" or mode == "classifier", "train_pc mode must be 'error' or 'classifier'."
    if mode == "error":
        assert best == "loss", "If mode='error', then best='loss'"
    
    #  For determining best model. Either can be used.
    best_val_acc = 0.0
    best_val_loss = 99999999.9
    best_val_err = 99999999.9

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)

    if mode == "error":
        for param in model.parameters():
            param.requires_grad = True
        for param in model.classifier.parameters():
            param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True

    for epoch in range(num_epochs):
        
        model.train()
        train_dataset.apply_transform()
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)    

        # num_correct = 0
        epoch_train_loss = 0.0
        epoch_train_err = 0.0
        epoch_train_acc = torch.zeros(3)

        for batch_idx, (images, y) in loop:
            x = images.to(device)
            if flatten:
                x = torch.flatten(x, start_dim=1)
            target = y.to(device)
            out = model(x)
            epoch_train_err += out[1]

            if mode == "error":
                loss = out[1]
            else:
                loss = criterion(out[0], target)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            with torch.no_grad():
                epoch_train_loss += loss.item()
                epoch_train_acc += torch.tensor(topk_accuracy(out[0], target, (1,3,5)))

                if epoch > 0:
                    loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
                    loop.set_postfix(
                        train_loss = train_loss[-1], 
                        train_acc = train_acc[-1][0].item(), 
                        val_loss = val_loss[-1], 
                        val_acc = val_acc[-1][0].item(),
                    )

        train_loss.append(epoch_train_loss / len(train_loader))
        train_acc.append(epoch_train_acc / len(train_loader))
        epoch_train_err /= len(train_loader)
        
        val_l, val_a, val_err = evaluate_pc(model, val_loader, criterion, device, flatten, mode)
        if mode == "error":
            val_loss.append(val_err)
        else:
            val_loss.append(val_l)
            val_acc.append(val_a)
            
        if save_model:
            if best == "loss":
                if best_val_loss > val_loss[-1]:
                    best_val_loss = val_loss[-1]
                    torch.save(model.state_dict(), f'models/{model_name}.pth')
            elif best in ["acc", "accuracy"]:
                if best_val_acc < val_acc[-1][0]:
                    best_val_acc = val_acc[-1][0]
                    torch.save(model.state_dict(), f'models/{model_name}.pth')
            else:
                raise Exception("Invalid value for 'best'")

        step += len(train_dataset)
        writer.add_scalar("Training Loss", train_loss[-1], step)
        writer.add_scalar("Training Accuracy Top1", train_acc[-1][0], step)
        writer.add_scalar("Validation Loss", val_loss[-1], step)
        if mode == "classifier":
            writer.add_scalar("Validation Accuracy Top1", val_acc[-1][0], step)
            writer.add_scalar("Validation Accuracy Top3", val_acc[-1][1], step)
            writer.add_scalar("Validation Accuracy Top5", val_acc[-1][2], step)
        
    return torch.tensor(train_loss), torch.stack(train_acc), torch.tensor(val_loss), torch.stack(val_acc), step