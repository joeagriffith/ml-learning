import torch
import torch.nn as nn
import torch.nn.functional as F

MU = 0.70
NU = 1.0
ETA = 0.1
STEPS = 5
device = "cuda" if torch.cuda.is_available() else "cpu"
device

class PCLayer(nn.Module):
    def __init__(self, 
                 e_shape,
                 r_shape,
                 kernel,
                 
                 nu,
                 mu,
                 eta,
                 
                 device,

                 maxpool=1,
                 forw_actv=nn.ReLU(),
                 td_actv=nn.Tanh(),
                 
                 padding=0,
                 **kwargs,
                ):
        super().__init__()
        self.e_shape = e_shape
        self.r_shape = r_shape

        self.device = device
        
        self.nu = nu
        self.mu = mu
        self.eta = eta
        
        self.conv = nn.Sequential(
            nn.Conv2d(e_shape[0], r_shape[0], kernel, device=device, padding=padding, **kwargs),
            nn.MaxPool2d(kernel_size=maxpool)
        )
        self.forw_actv = forw_actv
        
        self.convT = nn.Sequential(
            nn.Upsample(scale_factor=maxpool),
            nn.ConvTranspose2d(r_shape[0], e_shape[0], kernel, padding=1, device=device, **kwargs),
            td_actv,
        )
    
    def init_vars(self, batch_size):
        e = torch.zeros((batch_size, self.e_shape[0], self.e_shape[1], self.e_shape[2])).to(self.device)
        r = torch.zeros((batch_size, self.r_shape[0], self.r_shape[1], self.r_shape[2])).to(self.device)
        return e,r
    
    def step(self, x, e, r, td_err=None):
        e = self.forw_actv(x - self.convT(r))
        r = self.nu*r + self.mu*self.conv(e)
        if td_err is not None:
            r -= self.eta*td_err
        return e, r
    
class PCCNNModel(nn.Module):
    def __init__(self, features, input_shape, num_classes, nu=1.0, mu=1.0, eta=0.1, steps=5, device="cpu"):
        super().__init__()
        self.steps = steps
        self.device = device
        self.pc_layers = [
            PCLayer( input_shape,                             (features,input_shape[1],input_shape[2]), (3,3), nu, mu, eta, device, padding="same"),
            PCLayer((features,input_shape[1],input_shape[2]), (features,input_shape[1],input_shape[2]), (3,3), nu, mu, eta, device, padding="same"),
            
            PCLayer((features,input_shape[1],input_shape[2]),       (features*2,input_shape[1]//2,input_shape[2]//2), (3,3), nu, mu, eta, device, maxpool=2, padding="same"),
            PCLayer((features*2,input_shape[1]//2,input_shape[2]//2), (features*2,input_shape[1]//2,input_shape[2]//2), (3,3), nu, mu, eta, device, padding="same"),
            
            PCLayer((features*2,input_shape[1]//2,input_shape[2]//2), (features*4,input_shape[1]//4,input_shape[2]//4), (3,3), nu, mu, eta, device, maxpool=2, padding="same"),
            PCLayer((features*4,input_shape[1]//4,input_shape[2]//4), (features*4,input_shape[1]//4,input_shape[2]//4), (3,3), nu, mu, eta, device, padding="same"),
            PCLayer((features*4,input_shape[1]//4,input_shape[2]//4), (features*4,input_shape[1]//4,input_shape[2]//4), (3,3), nu, mu, eta, device, padding="same"),
            
            PCLayer((features*4,input_shape[1]//4,input_shape[2]//4), (features*8,input_shape[1]//8,input_shape[2]//8), (3,3), nu, mu, eta, device, maxpool=2, padding="same"),
            PCLayer((features*8,input_shape[1]//8,input_shape[2]//8), (features*8,input_shape[1]//8,input_shape[2]//8), (3,3), nu, mu, eta, device, padding="same"),
            PCLayer((features*8,input_shape[1]//8,input_shape[2]//8), (features*8,input_shape[1]//8,input_shape[2]//8), (3,3), nu, mu, eta, device, padding="same"),
            
            PCLayer((features*8,input_shape[1]//8, input_shape[2]//8),  (features*8,input_shape[1]//16,input_shape[2]//16), (3,3), nu, mu, eta, device, maxpool=2, padding="same"),
            PCLayer((features*8,input_shape[1]//16,input_shape[2]//16), (features*8,input_shape[1]//16,input_shape[2]//16), (3,3), nu, mu, eta, device, padding="same"),
            PCLayer((features*8,input_shape[1]//16,input_shape[2]//16), (features*8,input_shape[1]//16,input_shape[2]//16), (3,3), nu, mu, eta, device, padding="same")
        ]
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(int(features*8*(input_shape[1]/16)*(input_shape[2]/16)), features*64, device=device),
            nn.ReLU(),
            nn.Linear(features*64, num_classes, device=device),
        )
        
    def forward(self, x, target=None, calc_e_mag=True):
        batch_size = x.shape[0]
        e, r = [], []
        
        for layer in self.pc_layers:
            layer_e, layer_r = layer.init_vars(batch_size)
            e.append(layer_e)
            r.append(layer_r)
        
        for _ in range(self.steps):
            for i, layer in enumerate(self.pc_layers):
                if i == 0:
                    e[i], r[i] = layer.step(x, e[i], r[i], e[i+1])
                elif i < len(self.pc_layers)-1:
                    e[i], r[i] = layer.step(r[i-1], e[i], r[i], e[i+1])
                else:
                    e[i], r[i] = layer.step(r[i-1], e[i], r[i])
        
        out = self.classifier(r[-1])

        e_mag = 0.0
        n = 0
        if calc_e_mag:
            for errs in e:
                e_mag += errs.abs().sum()
                n += errs.numel()
            e_mag /= n

        return [out, e_mag]