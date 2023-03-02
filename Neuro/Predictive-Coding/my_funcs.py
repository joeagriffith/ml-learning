import torch

def topk_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res

def evaluate(model, data_loader, criterion, device="cpu", flatten=False):
    with torch.no_grad():
        model.eval()
        
        loss = 0.0
        acc = torch.zeros(3)

        for batch_idx, (images, y) in enumerate(data_loader):
            x = images.to(device)
            if flatten:
                x = torch.flatten(x, start_dim=1)
            target = y.to(device)
            out = model(x)
            loss += criterion(out, target).item()
            acc += torch.tensor(topk_accuracy(out, target, (1,3,5)))
        
        loss /= len(data_loader)
        acc /= len(data_loader) 

        return loss, acc

def evaluate_pc(model, data_loader, criterion, device="cpu", flatten=False):
    with torch.no_grad():
        model.eval()
        
        loss = 0.0
        acc = torch.zeros(3)
        err = 0.0
        for batch_idx, (images, y) in enumerate(data_loader):
            x = images.to(device)
            if flatten:
                x = torch.flatten(x, start_dim=1)
            target = y.to(device)
            out = model(x)
            e_mag += out[1]
            loss += criterion(out[0], target).item()
            acc += torch.tensor(topk_accuracy(out[0], target, (1,3,5)))
        
        loss /= len(data_loader)
        acc /= len(data_loader) 
        err /= len(data_loader)

        return loss, acc, err

class RandomGaussianNoise(object):
    def __init__(self, mean=0.0, std=0.001):
        self.mean = mean
        self.std = std
    
    def __call__(self, img):
        noise = (torch.randn(img.shape) * self.std + self.mean)
        if img.is_cuda:
            noise = noise.to("cuda")
        return torch.clip(img + noise, min=0.0, max=1.0)