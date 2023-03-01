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
                 
                 maxpool=1,
                 forw_actv=nn.ReLU(),
                 td_actv=nn.Tanh(),
                 
                 **kwargs,
                ):
        super().__init__()
        self.e_shape = e_shape
        self.r_shape = r_shape
        
        self.nu = nu
        self.mu = mu
        self.eta = eta
        
        self.conv = nn.Sequential(
            nn.Conv2d(e_shape[0], r_shape[0], kernel, **kwargs),
            nn.MaxPool2d(kernel_size=maxpool)
        )
        self.forw_actv = forw_actv
        
        self.convT = nn.Sequential(
            nn.Upsample(scale_factor=maxpool)
            nn.ConvTranspose2d(r_shape[0], e_shape[0], kernel, **kwargs),
            td_actv,
        )
    
    
    def init_vars(self, batch_size):
        e = torch.zeros((batch_size, self.e_shape[0], self.e_shape[1], self.e_shape[2]))
        r = torch.zeros((batch_size, self.r_shape[0], self.r_shape[1], self.r_shape[2]))
        return e,r
    
    def step(self, x, e, r, td_err=None):
        e = self.forw_actv(self.x - self.convT(r))
        r = self.nu*r + self.mu*self.conv(e) - self.eta*td_err
        return e, r
    
class PCCNNModel(nn.Module):
    def __init__(self, features, input_channels, num_classes, nu, mu, eta, steps=5):
        super().__init__()
        self.steps = steps
        
        self.pc0 = PCLayer((3,32,32), (features,32,32), (3,3), nu, mu, eta)
        self.pc1 = PCLayer((features,32,32), (features,32,32), (3,3), nu, mu, eta)
        
        self.pc2 = PCLayer((features,32,32), (features*2,16,16), (3,3), nu, mu, eta, maxpool=2)
        self.pc3 = PCLayer((features*2,16,16), (features*2,16,16), (3,3), nu, mu, eta)
        
        self.pc4 = PCLayer((features*2,16,16), (features*4,8,8), (3,3), nu, mu, eta, maxpool=2)
        self.pc5 = PCLayer((features*4,8,8), (features*4,8,8), (3,3), nu, mu, eta)
        self.pc6 = PCLayer((features*4,8,8), (features*4,8,8), (3,3), nu, mu, eta)
        
        self.pc4 = PCLayer((features*4,8,8), (features*8,4,4), (3,3), nu, mu, eta, maxpool=2)
        self.pc5 = PCLayer((features*8,4,4), (features*8,4,4), (3,3), nu, mu, eta)
        self.pc6 = PCLayer((features*8,4,4), (features*8,4,4), (3,3), nu, mu, eta)
        
        self.pc4 = PCLayer((features*8,4,4), (features*8,2,2), (3,3), nu, mu, eta, maxpool=2)
        self.pc5 = PCLayer((features*8,2,2), (features*8,2,2), (3,3), nu, mu, eta)
        self.pc6 = PCLayer((features*8,2,2), (features*8,2,2), (3,3), nu, mu, eta)
        
        self.classifier = nn.Sequential(
            nn.Flatten()
            nn.Linear(features*8*2*2, features*64),
            nn.ReLU(),
            nn.Linear(features*64, num_classes),
        )
        
    def forward(self, x, target=None, calc_e_mag=False):
        batch_size = x.shape[0]
        pc0_e, pc0_r = self.pc0.init_vars(batch_size)
        pc1_e, pc1_r = self.pc1.init_vars(batch_size)
        pc2_e, pc2_r = self.pc2.init_vars(batch_size)
        pc3_e, pc3_r = self.pc3.init_vars(batch_size)
        pc4_e, pc4_r = self.pc4.init_vars(batch_size)
        pc5_e, pc5_r = self.pc5.init_vars(batch_size)
        pc6_e, pc6_r = self.pc6.init_vars(batch_size)
        
        for _ in range(self.steps):
            pc0_e, pc0_r = self.pc0.step(x, pc0_e, pc0_r, pc1_e)
            pc1_e, pc1_r = self.pc1.step(pc0_r, pc1_e, pc1_r, pc2_e)
            pc2_e, pc2_r = self.pc2.step(pc1_r, pc2_e, pc2_r, pc3_e)
            pc3_e, pc3_r = self.pc3.step(pc2_r, pc3_e, pc3_r, pc4_e)
            pc4_e, pc4_r = self.pc4.step(pc3_r, pc4_e, pc4_r, pc5_e)
            pc5_e, pc5_r = self.pc5.step(pc4_r, pc5_e, pc5_r, pc6_e)
            pc6_e, pc6_r = self.pc6.step(pc5_r, pc6_e, pc6_r)
        
        out = self.classifier(pc6_r)

            fin_r = self.fin_layer.step(hid4_bu_e, fin_r, target)

        out = self.fc_layer(torch.flatten(F.relu(fin_r), start_dim=1))
        
        e_mag = 0.0
        n = 0
        if calc_e_mag:
            e_mag += pc0_e.abs().sum()
            e_mag += pc1_e.abs().sum()
            e_mag += pc2_e.abs().sum()
            e_mag += pc3_e.abs().sum()
            e_mag += pc4_e.abs().sum()
            e_mag += pc5_e.abs().sum()
            e_mag += pc6_e.abs().sum()
            
            n += pc0_e.numel()
            n += pc1_e.numel()
            n += pc2_e.numel()
            n += pc3_e.numel()
            n += pc4_e.numel()
            n += pc5_e.numel()
            n += pc6_e.numel()
            
            e_mag /= n

        return [out, e_mag]