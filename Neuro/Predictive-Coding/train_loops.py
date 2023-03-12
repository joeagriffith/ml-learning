import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from my_funcs import topk_accuracy, evaluate, evaluate_pc, evaluate_pcv3
from PCLayer import PCLayer
import torch.optim as optim


    
def train(
    model, 
    train_dataset,
    val_dataset,
    optimiser, 
    criterion, 
    model_name, 
    num_epochs, 
    flatten=False, 
    model_dir="models",
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
        epoch_train_acc = torch.zeros(3, device=device)

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
                epoch_train_acc += topk_accuracy(out, target, (1,3,5))

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
                    torch.save(model.state_dict(), f'{model_dir}/{model_name}.pth')
            elif best in ["acc", "accuracy"]:
                if best_val_acc < val_acc[-1][0]:
                    best_val_acc = val_acc[-1][0]
                    torch.save(model.state_dict(), f'models/{model_name}.pth')
            else:
                raise Exception("Invalid value for 'best'")

        step += len(train_dataset)
        writer.add_scalar("Training Loss", train_loss[-1], step)
        writer.add_scalar("Training Accuracy Top1", train_acc[-1][0], step)
        writer.add_scalar("Training Accuracy Top3", train_acc[-1][1], step)
        writer.add_scalar("Training Accuracy Top5", train_acc[-1][2], step)
        writer.add_scalar("Validation Loss", val_loss[-1], step)
        writer.add_scalar("Validation Accuracy Top1", val_acc[-1][0], step)
        writer.add_scalar("Validation Accuracy Top3", val_acc[-1][1], step)
        writer.add_scalar("Validation Accuracy Top5", val_acc[-1][2], step)
        
    return torch.tensor(train_loss), torch.stack(train_acc), torch.tensor(val_loss), torch.stack(val_acc), step

def train_pc_classification(
    model, 
    train_dataset,
    val_dataset,
    optimiser, 
    criterion, 
    model_name, 
    num_epochs, 
    flatten=False, 
    model_dir="models",
    log_dir="logs", 
    step=0, 
    best="loss",
    save_model=True,
    batch_size=100,
    learning_rate=3e-4,
    weight_decay=1e-2,
    plot_errs=False,
    device="cpu"
):
    writer = SummaryWriter(f"{log_dir}/{model_name}")
    
    #  For determining best model. Either can be used.
    best_val_acc = 0.0
    best_val_loss = 99999999.9
    best_val_err = 999999.9

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)

    epoch_val_loss = 0.0
    epoch_val_acc = torch.zeros(3)
    epoch_val_mean_err = 0.0
    epoch_val_errs = torch.zeros(model.pc_len())

    for epoch in range(num_epochs):
        
        model.train()
        train_dataset.apply_transform()
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)    

        epoch_train_loss = 0.0
        epoch_train_acc = torch.zeros(3).to(device)
        epoch_train_errs = torch.zeros(model.pc_len()).to(device)
        epoch_train_mean_err = 0.0

        for batch_idx, (images, y) in loop:
            x = images.to(device)
            if flatten:
                x = torch.flatten(x, start_dim=1)
            target = y.to(device)
            out = model(x)
            for i, e in enumerate(out[1]):
                epoch_train_errs[i] += e.detach().square().mean()
                epoch_train_mean_err += e.detach().square().mean()

            loss = criterion(out[0], target)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            epoch_train_loss += loss.item()
            epoch_train_acc += topk_accuracy(out[0], target, (1,3,5))

            if epoch > 0 and batch_idx > 0:
                loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
                loop.set_postfix(
                    train_loss = epoch_train_loss / batch_idx, 
                    val_loss = epoch_val_loss, 
                    train_acc = epoch_train_acc[0].item() / batch_idx, 
                    val_acc = epoch_val_acc[0].item(),
                )

        epoch_train_loss /= len(train_loader)
        epoch_train_acc /= len(train_loader)
        epoch_train_errs /= len(train_loader)
        epoch_train_mean_err /= len(train_loader) * model.pc_len()
        
        epoch_val_loss, epoch_val_acc, epoch_val_errs, epoch_val_mean_err = evaluate_pc(model, val_loader, criterion, device, flatten)
            
        if save_model:
            if best == "loss":
                if best_val_loss > epoch_val_loss:
                    best_val_loss = epoch_val_loss
                    torch.save(model.state_dict(), f'{model_dir}/{model_name}.pth')
            elif best in ["acc", "accuracy"]:
                if best_val_acc < epoch_val_acc[0]:
                    best_val_acc = epoch_val_acc[0]
                    torch.save(model.state_dict(), f'{model_dir}/{model_name}.pth')
            elif best == "error":
                if best_val_err > epoch_val_mean_err:
                    best_val_err = epoch_val_mean_err
                    torch.save(model.state_dict(), f'{model_dir}/{model_name}.pth')
            else:
                raise Exception("Invalid value for 'best'")

        step += len(train_dataset)
        writer.add_scalar("Training Loss", epoch_train_loss, step)
        writer.add_scalar("Validation Loss", epoch_val_loss, step)
        writer.add_scalar("Training Accuracy Top1", epoch_train_acc[0], step)
        writer.add_scalar("Training Accuracy Top3", epoch_train_acc[1], step)
        writer.add_scalar("Training Accuracy Top5", epoch_train_acc[2], step)
        writer.add_scalar("Validation Accuracy Top1", epoch_val_acc[0], step)
        writer.add_scalar("Validation Accuracy Top3", epoch_val_acc[1], step)
        writer.add_scalar("Validation Accuracy Top5", epoch_val_acc[2], step)
        writer.add_scalar("Training Mean Err", epoch_train_mean_err, step)
        writer.add_scalar("Validation Mean Err", epoch_val_mean_err, step)
        if plot_errs:
            for i in range(model.pc_len()):
                writer.add_scalar(f"Training Err layer: {i}", epoch_train_errs[i], step)
                writer.add_scalar(f"Validation Err layer: {i}", epoch_val_errs[i], step)
        
    return step


def train_pc_error(
    model, 
    train_dataset,
    val_dataset,
    optimiser, 
    criterion, 
    model_name, 
    num_epochs, 
    flatten=False, 
    model_dir="models",
    log_dir="logs", 
    step=0, 
    best="error",
    save_model=True,
    batch_size=100,
    learning_rate=3e-4,
    weight_decay=1e-2,
    plot_errs=False,
    lambdas="all",
    device="cpu"
):
    writer = SummaryWriter(f"{log_dir}/{model_name}")
    
    #  For determining best model. Either can be used.
    best_val_acc = 0.0
    best_val_mean_err = 99999999.9

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)

    epoch_val_acc = torch.zeros(3)
    epoch_val_errs = torch.zeros(model.pc_len())
    epoch_val_mean_err = 0.0

    layer_loss_weights = [0.1 for _ in range(model.pc_len())]
    if lambdas == "L_0":
        layer_loss_weights *= 0.0
    layer_loss_weights[0] = 1.0

    for epoch in range(num_epochs):
        
        model.train()
        train_dataset.apply_transform()
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)    

        epoch_train_acc = torch.zeros(3).to(device)
        epoch_train_errs = torch.zeros(model.pc_len()).to(device)
        epoch_train_mean_err = 0.0

        for batch_idx, (images, y) in loop:
            x = images.to(device)
            if flatten:
                x = torch.flatten(x, start_dim=1)
            target = y.to(device)
            # out = model.guided_forward(x, F.one_hot(target, model.num_classes).float())
            out = model(x)

            loss = torch.zeros(1).to(device)
            for i, e in enumerate(out[1]):
                epoch_train_errs[i] += e.detach().square().mean()
                epoch_train_mean_err += e.detach().square().mean()
                layer_loss = layer_loss_weights[i]*e.mean()
                if i == 0:
                    layer_loss *= 0.0
                loss += layer_loss

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            out = model.classify(out)
            

            epoch_train_acc += topk_accuracy(out[0][1], target, (1,3,5))

            if epoch > 0 and batch_idx > 0:
                loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
                loop.set_postfix(
                    train_mean_err = epoch_train_mean_err.item() / batch_idx, 
                    val_mean_err = epoch_val_mean_err.item(), 
                    train_acc = epoch_train_acc[0].item() / batch_idx, 
                    val_acc = epoch_val_acc[0].item(),
                )

        epoch_train_acc /= len(train_loader)
        epoch_train_errs /= len(train_loader)
        epoch_train_mean_err /= len(train_loader) * model.pc_len()
        
        epoch_val_acc, epoch_val_errs, epoch_val_mean_err = evaluate_pc(model, val_loader, criterion, device, flatten)
            
        if save_model:
            if best == "error":
                if best_val_mean_err > epoch_val_mean_err:
                    best_val_mean_err = epoch_val_mean_err
                    torch.save(model.state_dict(), f'{model_dir}/{model_name}.pth')
            elif best in ["acc", "accuracy"]:
                if best_val_acc < epoch_val_acc[0]:
                    best_val_acc = epoch_val_acc[0]
                    torch.save(model.state_dict(), f'models/{model_name}.pth')
            else:
                raise Exception("Invalid value for 'best'")

        step += len(train_dataset)
        writer.add_scalar("Training Mean Err", epoch_train_mean_err, step)
        writer.add_scalar("Validation Mean Err", epoch_val_mean_err, step)
        writer.add_scalar("Training Accuracy Top1", epoch_train_acc[0], step)
        writer.add_scalar("Training Accuracy Top3", epoch_train_acc[1], step)
        writer.add_scalar("Training Accuracy Top5", epoch_train_acc[2], step)
        writer.add_scalar("Validation Accuracy Top1", epoch_val_acc[0], step)
        writer.add_scalar("Validation Accuracy Top3", epoch_val_acc[1], step)
        writer.add_scalar("Validation Accuracy Top5", epoch_val_acc[2], step)
        if plot_errs:
            for i in range(model.pc_len()):
                writer.add_scalar(f"Training Err layer: {i}", epoch_train_errs[i], step)
                writer.add_scalar(f"Validation Err layer: {i}", epoch_val_errs[i], step)
        
    return step

def train_pc_either(
    model, 
    train_dataset,
    val_dataset,
    optimiser, 
    criterion, 
    model_name, 
    num_epochs, 
    flatten=False, 
    model_dir="models",
    log_dir="logs", 
    step=0, 
    best="loss",
    save_model=True,
    batch_size=100,
    learning_rate=3e-4,
    weight_decay=1e-2,
    mode="error",
    plot_err=False,
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
                    if mode == "error":
                        loop.set_postfix(
                            train_loss = train_loss[-1], 
                            val_loss = val_loss[-1], 
                        )
                    else:
                        loop.set_postfix(
                            train_loss = train_loss[-1], 
                            val_loss = val_loss[-1], 
                            train_acc = train_acc[-1][0].item(), 
                            val_acc = val_acc[-1][0].item(),
                        )

        train_loss.append(epoch_train_loss / len(train_loader))
        train_acc.append(epoch_train_acc / len(train_loader))
        epoch_train_err /= len(train_loader)
        
        val_l, val_a, val_err = evaluate_pc(model, val_loader, criterion, device, flatten)
        if mode == "error":
            val_loss.append(val_err.item())
        else:
            val_loss.append(val_l)
            val_acc.append(val_a)
            
        if save_model:
            if best == "loss":
                if best_val_loss > val_loss[-1]:
                    best_val_loss = val_loss[-1]
                    torch.save(model.state_dict(), f'{model_dir}/{model_name}.pth')
            elif best in ["acc", "accuracy"]:
                if best_val_acc < val_acc[-1][0]:
                    best_val_acc = val_acc[-1][0]
                    torch.save(model.state_dict(), f'models/{model_name}.pth')
            else:
                raise Exception("Invalid value for 'best'")

        step += len(train_dataset)
        if mode == "error" and plot_err:
            writer.add_scalar("Training Loss", train_loss[-1], step)
            writer.add_scalar("Validation Loss", val_loss[-1], step)
        if mode == "classifier":
            writer.add_scalar("Training Loss", train_loss[-1], step)
            writer.add_scalar("Validation Loss", val_loss[-1], step)
            writer.add_scalar("Training Accuracy Top1", train_acc[-1][0], step)
            writer.add_scalar("Validation Accuracy Top1", val_acc[-1][0], step)
            writer.add_scalar("Validation Accuracy Top3", val_acc[-1][1], step)
            writer.add_scalar("Validation Accuracy Top5", val_acc[-1][2], step)
        
    if mode == "error":
        return torch.tensor(train_loss), torch.tensor(val_loss), step
    else:
        return torch.tensor(train_loss), torch.stack(train_acc), torch.tensor(val_loss), torch.stack(val_acc), step

def train_pc_ec(
    model, 
    train_dataset,
    val_dataset,
    criterion, 
    model_name, 
    num_epochs,
    pc_learning_rate = 3e-4,
    pc_weight_decay = 1e-2,
    c_learning_rate = 3e-4, 
    c_weight_decay = 1e-2,
    optimiser = "AdamW",
    flatten=False, 
    model_dir="models",
    log_dir="logs", 
    step=0, 
    best="loss",
    save_model=True,
    batch_size=100,
    plot_err=True,
    lambdas="all",
    device="cpu"
):
    writer = SummaryWriter(f"{log_dir}/{model_name}")

    #  For determining best model. Either can be used.
    best_val_acc = 0.0
    best_val_loss = 99999999.9

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)

    pc_optimisers = []
    c_optimiser = None

    for module in model.children():
        if type(module) == PCLayer:
            if optimiser == "AdamW":
                opt = optim.AdamW(module.parameters(), lr=pc_learning_rate, weight_decay=pc_weight_decay )
                pc_optimisers.append(opt)
                
        else:
            c_optimiser = optim.AdamW(module.parameters(), lr=c_learning_rate, weight_decay=c_weight_decay)

    layer_loss_weights = [0.1 for _ in range(model.pc_len())]
    if lambdas == "L_0":
        layer_loss_weights *= 0.0
    if lambdas == "same":
        layer_loss_weights = [1.0 for _ in range(model.pc_len())]
    layer_loss_weights[0] = 1.0

    for epoch in range(num_epochs):
        
        #  Initialise variables and prepare data for new epoch
        model.train()
        train_dataset.apply_transform()
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)    

        #  Metric init
        train_loss = 0.0
        train_acc = torch.zeros(3, device=device)
        train_err = torch.zeros(len(pc_optimisers), device=device).to(device)
        train_mean_err = 0.0

        for batch_idx, (images, y) in loop:
            x = images.to(device)
            if flatten:
                x = torch.flatten(x, start_dim=1)
            target = y.to(device)
            out = model(x)
            # out = model.guided_forward(x, F.one_hot(target, 10))

            #  Training classifier loss
            c_optimiser.zero_grad()
            batch_loss = criterion(out[0], target)
            batch_loss.backward(retain_graph=True)
            if c_optimiser is not None:
                c_optimiser.step()
            train_loss += batch_loss.item()

            #  Training accuracy
            with torch.no_grad():
                train_acc += topk_accuracy(out[0], target, (1,3,5))

            #  Training Es
            for i, e in enumerate(out[1]):
                error = e.mean()*layer_loss_weights[i]
                error.backward(retain_graph=True)
                train_err[i] += error.detach()
                train_mean_err += e.detach().mean()
            for i in range(len(out[1])):
                pc_optimisers[i].step()
                pc_optimisers[i].zero_grad()

            #  TQDM bar update
            if epoch > 0 and batch_idx > 0:
                loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
                loop.set_postfix(
                    train_loss = train_loss/batch_idx, 
                    val_loss = val_loss, 
                    train_acc = train_acc[0].item()/batch_idx, 
                    val_acc = val_acc[0].item(),
                )

        #  Convert running totals of training metrics to epoch means
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        train_err /= len(train_loader)
        train_mean_err /= len(train_loader) * model.pc_len()
        
        #  Calculate validation metrics
        val_loss, val_acc, val_err, val_mean_err = evaluate_pc(model, val_loader, criterion, device, flatten)
            
        #  Save model if selected metric is a high score
        if save_model:
            if best == "loss":
                if best_val_loss > val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), f'{model_dir}/{model_name}.pth')
            elif best in ["acc", "accuracy"]:
                if best_val_acc < val_acc[0]:
                    best_val_acc = val_acc[0]
                    torch.save(model.state_dict(), f'{model_dir}/{model_name}.pth')
            else:
                raise Exception("Invalid value for 'best'")

        #  Log metrics for tensorboard
        step += len(train_dataset)
        writer.add_scalar("Training Loss", train_loss, step)
        writer.add_scalar("Validation Loss", val_loss, step)
        writer.add_scalar("Training Accuracy Top1", train_acc[0], step)
        writer.add_scalar("Training Accuracy Top3", train_acc[1], step)
        writer.add_scalar("Training Accuracy Top5", train_acc[2], step)
        writer.add_scalar("Validation Accuracy Top1", val_acc[0].item(), step)
        writer.add_scalar("Validation Accuracy Top3", val_acc[1].item(), step)
        writer.add_scalar("Validation Accuracy Top5", val_acc[2].item(), step)
        writer.add_scalar("Training Mean Err", train_mean_err, step)
        writer.add_scalar("Validation Mean Err", val_mean_err, step)
        if plot_err:
            for i in range(len(train_err)):
                writer.add_scalar(f"Training Err layer: {i}", train_err[i].item(), step) 
            for i in range(len(val_err)):
                writer.add_scalar(f"Validation Err layer: {i}", val_err[i].item(), step) 

    return step

def train_pc_hebbian(
    model, 
    train_dataset,
    val_dataset,
    criterion, 
    model_name, 
    num_epochs,
    pc_learning_rate = 3e-4,
    pc_weight_decay = 1e-2,
    c_learning_rate = 3e-4, 
    c_weight_decay = 1e-2,
    optimiser = "AdamW",
    flatten=False, 
    model_dir="models",
    log_dir="logs", 
    step=0, 
    best="loss",
    save_model=True,
    batch_size=100,
    plot_err=True,
    lambdas="all",
    device="cpu"
):
    writer = SummaryWriter(f"{log_dir}/{model_name}")

    #  For determining best model. Either can be used.
    best_val_acc = 0.0
    best_val_loss = 99999999.9

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)

    pc_optimisers = []
    c_optimiser = None

    for module in model.children():
        if type(module) == PCLayer:
            if optimiser == "AdamW":
                opt = optim.AdamW(module.parameters(), lr=pc_learning_rate, weight_decay=pc_weight_decay )
                pc_optimisers.append(opt)
                
        else:
            c_optimiser = optim.AdamW(module.parameters(), lr=c_learning_rate, weight_decay=c_weight_decay)

    layer_loss_weights = [0.1 for _ in range(len(model.pclayers))]
    if lambdas == "L_0":
        layer_loss_weights *= 0.0
    if lambdas == "same":
        layer_loss_weights = [1.0 for _ in range(len(model.pclayers))]
    layer_loss_weights[0] = 1.0

    time_loss_weights = [1.0 for _ in range(model.steps)]
    time_loss_weights[0] = 0.0

    for epoch in range(num_epochs):
        
        #  Initialise variables and prepare data for new epoch
        model.train()
        train_dataset.apply_transform()
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)    

        #  Metric init
        train_loss = 0.0
        train_acc = torch.zeros(3, device=device)
        train_err = torch.zeros(len(pc_optimisers), device=device).to(device)
        train_mean_err = 0.0

        for batch_idx, (images, y) in loop:
            x = images.to(device)
            if flatten:
                x = torch.flatten(x, start_dim=1)
            target = y.to(device)
            # out = model(x)
            # out = model.guided_forward(x, F.one_hot(target, 10))

            e, r = model.init_vars(len(target), device)
            for i in range(model.steps):
                e, r = model.step(x, e, r)
                

                

            out = model.classify(r[-1])

            #  Training classifier loss
            c_optimiser.zero_grad()
            batch_loss = criterion(out[0], target)
            batch_loss.backward(retain_graph=True)
            if c_optimiser is not None:
                c_optimiser.step()
            train_loss += batch_loss.item()

            #  Training accuracy
            with torch.no_grad():
                train_acc += topk_accuracy(out[0], target, (1,3,5))

            # #  Training Es
            # for i, e in enumerate(out[1]):
            #     error = e.mean()*layer_loss_weights[i]
            #     error.backward(retain_graph=True)
            #     train_err[i] += error.detach()
            #     train_mean_err += e.detach().mean()
            # for i in range(len(out[1])):
            #     pc_optimisers[i].step()
            #     pc_optimisers[i].zero_grad()

            #  TQDM bar update
            if epoch > 0 and batch_idx > 0:
                loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
                loop.set_postfix(
                    train_loss = train_loss/batch_idx, 
                    val_loss = val_loss, 
                    train_acc = train_acc[0].item()/batch_idx, 
                    val_acc = val_acc[0].item(),
                )

        #  Convert running totals of training metrics to epoch means
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        train_err /= len(train_loader)
        train_mean_err /= len(train_loader) * model.pc_len()
        
        #  Calculate validation metrics
        val_loss, val_acc, val_err, val_mean_err = evaluate_pc(model, val_loader, criterion, device, flatten)
            
        #  Save model if selected metric is a high score
        if save_model:
            if best == "loss":
                if best_val_loss > val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), f'{model_dir}/{model_name}.pth')
            elif best in ["acc", "accuracy"]:
                if best_val_acc < val_acc[0]:
                    best_val_acc = val_acc[0]
                    torch.save(model.state_dict(), f'{model_dir}/{model_name}.pth')
            else:
                raise Exception("Invalid value for 'best'")

        #  Log metrics for tensorboard
        step += len(train_dataset)
        writer.add_scalar("Training Loss", train_loss, step)
        writer.add_scalar("Validation Loss", val_loss, step)
        writer.add_scalar("Training Accuracy Top1", train_acc[0], step)
        writer.add_scalar("Training Accuracy Top3", train_acc[1], step)
        writer.add_scalar("Training Accuracy Top5", train_acc[2], step)
        writer.add_scalar("Validation Accuracy Top1", val_acc[0].item(), step)
        writer.add_scalar("Validation Accuracy Top3", val_acc[1].item(), step)
        writer.add_scalar("Validation Accuracy Top5", val_acc[2].item(), step)
        writer.add_scalar("Training Mean Err", train_mean_err, step)
        writer.add_scalar("Validation Mean Err", val_mean_err, step)
        if plot_err:
            for i in range(len(train_err)):
                writer.add_scalar(f"Training Err layer: {i}", train_err[i].item(), step) 
            for i in range(len(val_err)):
                writer.add_scalar(f"Validation Err layer: {i}", val_err[i].item(), step) 

    return step

def train_pc_bystep(
    model, 
    train_dataset,
    val_dataset,
    optimiser,
    criterion, 
    model_name, 
    num_epochs,
    steps = 5,
    learning_rate = 3e-4,
    weight_decay = 1e-2,
    flatten=False, 
    model_dir="models",
    log_dir="logs", 
    step=0, 
    best="loss",
    save_model=True,
    batch_size=100,
    plot_err=True,
    device="cpu"
):
    writer = SummaryWriter(f"{log_dir}/{model_name}")

    #  For determining best model. Either can be used.
    best_val_acc = 0.0
    best_val_loss = 99999999.9

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)

    for epoch in range(num_epochs):
        
        #  Initialise variables and prepare data for new epoch
        model.train()
        train_dataset.apply_transform()
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)    

        #  Metric init
        num_es = 0
        for module in model.children():
            if type(module) == PCLayer:
                num_es += 1

        train_loss = 0.0
        train_acc = torch.zeros(3)
        train_err = torch.zeros(num_es)

        for batch_idx, (images, y) in loop:
            x = images.to(device)
            if flatten:
                x = torch.flatten(x, start_dim=1)
            target = y.to(device)

            e0, e1, e2, r0, r1, r2 = model.init_vars(batch_size,)
            for i in range(steps):
                e0, e1, e2, r0, r1, r2 = model.step(x, e0, e1, e2, r0, r1, r2)
                error = e0.square().sum() + e1.square().sum() + e2.square().sum()
                error.backward()
            train_err += torch.tensor([e0.detach(), e1.detach(), e2.detach()])

            out = model.classify(r2)
            batch_loss = criterion(out, target)
            batch_loss.backward()
            train_loss += batch_loss.item()

            optimiser.step()
            optimiser.zero_grad()

            #  Training accuracy
            with torch.no_grad():
                train_acc += torch.tensor(topk_accuracy(out, target, (1,3,5)))

            #  TQDM bar update
            if epoch > 0:
                loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
                loop.set_postfix(
                    train_loss = train_loss/batch_idx, 
                    val_loss = val_loss, 
                    train_acc = train_acc[0].item()/batch_idx, 
                    val_acc = val_acc[0].item(),
                )

        #  Convert running totals of training metrics to epoch means
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        train_err /= len(train_loader)
        
        #  Calculate validation metrics
        val_loss, val_acc, val_err = evaluate_pc(model, val_loader, criterion, device, flatten)
            
        #  Save model if selected metric is a high score
        if save_model:
            if best == "loss":
                if best_val_loss > val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), f'{model_dir}/{model_name}.pth')
            elif best in ["acc", "accuracy"]:
                if best_val_acc < val_acc[0]:
                    best_val_acc = val_acc[0]
                    torch.save(model.state_dict(), f'models/{model_name}.pth')
            else:
                raise Exception("Invalid value for 'best'")

        #  Log metrics for tensorboard
        step += len(train_dataset)
        writer.add_scalar("Training Loss", train_loss, step)
        writer.add_scalar("Validation Loss", val_loss, step)
        writer.add_scalar("Training Accuracy Top1", train_acc[0], step)
        writer.add_scalar("Training Accuracy Top3", train_acc[1], step)
        writer.add_scalar("Training Accuracy Top3", train_acc[2], step)
        writer.add_scalar("Validation Accuracy Top1", val_acc[0].item(), step)
        writer.add_scalar("Validation Accuracy Top3", val_acc[1].item(), step)
        writer.add_scalar("Validation Accuracy Top5", val_acc[2].item(), step)
        if plot_err:
            for i in range(len(train_err)):
                writer.add_scalar(f"Training Error {i}", train_err[i].item(), step) 
            for i in range(len(val_err)):
                writer.add_scalar(f"Validation Error {i}", val_err[i].item(), step) 

    return step