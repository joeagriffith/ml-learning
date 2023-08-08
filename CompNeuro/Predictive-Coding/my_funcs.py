import torch
import torch.nn.functional as F

def topk_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        device = "cuda" if output.is_cuda else "cpu"
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = torch.zeros(len(topk), dtype=float, device=device)
        for i, k in enumerate(topk):
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res[i] = correct_k.mul_(100.0 / batch_size)
        return res

def evaluate(model, data_loader, criterion, device, flatten=False):
    with torch.no_grad():
        model.eval()
        
        loss = 0.0
        acc = torch.zeros(3, device=device)

        for batch_idx, (images, y) in enumerate(data_loader):
            x = images.to(device)
            if flatten:
                x = torch.flatten(x, start_dim=1)
            target = y.to(device)
            out = model(x)
            loss += criterion(out, target).item()
            acc += topk_accuracy(out, target, (1,3,5))
        
        loss /= len(data_loader)
        acc /= len(data_loader) 

        return loss, acc

def evaluate_pc(model, data_loader, criterion, device, flatten=False):
    with torch.no_grad():
        model.eval()

        loss = 0.0
        acc = torch.zeros(3, device=device)
        errs = torch.zeros(model.pc_len(), device=device)
        mean_err = 0.0

        for batch_idx, (images, y) in enumerate(data_loader):
            x = images.to(device)
            if flatten:
                x = torch.flatten(x, start_dim=1)
            target = y.to(device)
            out = model(x)

            loss += criterion(out[0], target).item()

            acc += topk_accuracy(out[0], target, (1,3,5))

            for i, e in enumerate(out[1]):
                errs[i] += e.square().mean()
                mean_err += e.square().mean()
        
        loss /= len(data_loader)
        acc /= len(data_loader) 
        errs /= len(data_loader)
        mean_err /= len(data_loader) * model.pc_len()

        return loss, acc, errs, mean_err

def evaluate_pcv3(model, data_loader, criterion, device, flatten=False):
    with torch.no_grad():
        model.eval()

        acc = torch.zeros(3, device=device)
        errs = torch.zeros(model.pc_len(), device=device)
        mean_err = 0.0

        for batch_idx, (images, y) in enumerate(data_loader):
            x = images.to(device)
            if flatten:
                x = torch.flatten(x, start_dim=1)
            target = y.to(device)
            out = model(x)

            acc += topk_accuracy(out[0][1], target, (1,3,5))

            for i, e in enumerate(out[1]):
                errs[i] += e.square().mean()
                mean_err += e.square().mean()
        
        acc /= len(data_loader) 
        errs /= len(data_loader)
        mean_err /= len(data_loader) * model.pc_len()

        return acc, errs, mean_err

class RandomGaussianNoise(object):
    def __init__(self, mean=0.0, std=0.001):
        self.mean = mean
        self.std = std
    
    def __call__(self, img):
        noise = (torch.randn(img.shape) * self.std + self.mean)
        if img.is_cuda:
            noise = noise.to("cuda")
        return torch.clip(img + noise, min=0.0, max=1.0)

class Scale(object):
    def __init__(self, k=1.0):
        self.k = k
    
    def __call__(self, img):
        return img * self.k

def SpearMax(tensor):
    assert len(tensor.shape) == 4, f"Spearmax cannot be applied on tensor of shape {tensor.shape}"
    num = tensor.shape[1]
    tensor = tensor.argmax(dim=1)
    return F.one_hot(tensor, num).transpose(3,2).transpose(2,1).float()