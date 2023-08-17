import torch
import torch.nn.functional as F

# Will mix two numbers of same class 1/num_classes of the time
class MakeNegativeSample():
    def __init__(self, dataset):
        self.dataset = dataset
        self.n = len(dataset)
    def __call__(self, x, steps=10):
        i = torch.randint(0, self.n-1, (x.shape[0],))
        return mix_images(x, self.dataset[i][0], steps=steps)

def mix_images(x1, x2, steps=10, return_mask=False):
    device = x1.device
    mask = torch.bernoulli(torch.ones((x1.shape[0],1,28,28))*0.5).to(device)
    # blur  with a filter of the form [1/4, 1/2, 1/4] in both horizontal and veritical directions
    filter_h = torch.tensor([[1/4, 1/2, 1/4]]).unsqueeze(0).to(device)
    filter_v = torch.tensor([[1/4], [1/2], [1/4]]).unsqueeze(0).to(device)
    for _ in range(steps):
        mask = F.conv2d(mask, filter_h.unsqueeze(0), padding='same')
        mask = F.conv2d(mask, filter_v.unsqueeze(0), padding='same')
    
    # threshold at 0.5
    mask_1 = mask > 0.5
    mask_2 = mask <= 0.5

    out = x1*mask_1 + x2*mask_2
    if return_mask:
        return out.squeeze(0), mask_1
    else:
        return out.squeeze(0)