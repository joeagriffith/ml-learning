import torch
import torch.nn as nn
import torch.nn.functional as F
from PCLayer import PCLayer

MU = 0.70
NU = 1.0
ETA = 0.1
STEPS = 5
device = "cuda" if torch.cuda.is_available() else "cpu"
device

    
class PCCNNModel(nn.Module):
    def __init__(self, features, input_shape, num_classes, nu=1.0, mu=1.0, eta=0.1, steps=5, device="cpu"):
        super().__init__()
        self.steps = steps
        self.device = device
        features = [
            features,
            int(features*1.5),
            int(features*2.25),
            features*4,
            features*4,
        ]
            
        self.pc0 = PCLayer( input_shape,                                (features[0],input_shape[1],input_shape[2]), (3,3), nu, mu, eta, device, padding="same"),
        self.pc1 = PCLayer((features[0],input_shape[1],input_shape[2]), (features[0],input_shape[1],input_shape[2]), (3,3), nu, mu, eta, device, padding="same"),
            
        self.pc2 = PCLayer((features[0],input_shape[1],input_shape[2]),       (features[1],input_shape[1]//2,input_shape[2]//2), (3,3), nu, mu, eta, device, maxpool=2, padding="same"),
        self.pc3 = PCLayer((features[1],input_shape[1]//2,input_shape[2]//2), (features[1],input_shape[1]//2,input_shape[2]//2), (3,3), nu, mu, eta, device, padding="same"),
            
        self.pc4 = PCLayer((features[1],input_shape[1]//2,input_shape[2]//2), (features[2],input_shape[1]//4,input_shape[2]//4), (3,3), nu, mu, eta, device, maxpool=2, padding="same"),
        self.pc5 = PCLayer((features[2],input_shape[1]//4,input_shape[2]//4), (features[2],input_shape[1]//4,input_shape[2]//4), (3,3), nu, mu, eta, device, padding="same"),
        self.pc6 = PCLayer((features[2],input_shape[1]//4,input_shape[2]//4), (features[2],input_shape[1]//4,input_shape[2]//4), (3,3), nu, mu, eta, device, padding="same"),
            
        self.pc7 = PCLayer((features[2],input_shape[1]//4,input_shape[2]//4), (features[3],input_shape[1]//8,input_shape[2]//8), (3,3), nu, mu, eta, device, maxpool=2, padding="same"),
        self.pc8 = PCLayer((features[3],input_shape[1]//8,input_shape[2]//8), (features[3],input_shape[1]//8,input_shape[2]//8), (3,3), nu, mu, eta, device, padding="same"),
        self.pc9 = PCLayer((features[3],input_shape[1]//8,input_shape[2]//8), (features[3],input_shape[1]//8,input_shape[2]//8), (3,3), nu, mu, eta, device, padding="same"),
            
        self.pc10 = PCLayer((features[3],input_shape[1]//8, input_shape[2]//8),  (features[4],input_shape[1]//16,input_shape[2]//16), (3,3), nu, mu, eta, device, maxpool=2, padding="same"),
        self.pc11 = PCLayer((features[4],input_shape[1]//16,input_shape[2]//16), (features[4],input_shape[1]//16,input_shape[2]//16), (3,3), nu, mu, eta, device, padding="same"),
        self.pc12 = PCLayer((features[4],input_shape[1]//16,input_shape[2]//16), (features[4],input_shape[1]//16,input_shape[2]//16), (3,3), nu, mu, eta, device, padding="same")
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(features[4]*(input_shape[1]//16)*(input_shape[2]//16), features[0]*64, device=device),
            nn.ReLU(),
            nn.Linear(features[0]*64, num_classes, device=device),
        )

    def to(self, device):
        # for pc_layer in self.pc_layers:
        #     pc_layer.to(device)
        self.pc0.to(device)
        self.pc1.to(device)
        self.pc2.to(device)
        self.pc3.to(device)
        self.pc4.to(device)
        self.pc5.to(device)
        self.pc6.to(device)
        self.pc7.to(device)
        self.pc8.to(device)
        self.pc9.to(device)
        self.pc10.to(device)
        self.pc11.to(device)
        self.pc12.to(device)
        self.classifier = self.classifier.to(device)
        
    def forward(self, x, target=None, calc_e_mag=True):
        batch_size = x.shape[0]

        # e, r = [], []
        # for layer in self.pc_layers:
        #     layer_e, layer_r = layer.init_vars(batch_size)
        #     e.append(layer_e)
        #     r.append(layer_r)
        
        # for _ in range(self.steps):
        #     for i, layer in enumerate(self.pc_layers):
        #         if i == 0:
        #             e[i], r[i] = layer.step(x, e[i], r[i], e[i+1])
        #         elif i < len(self.pc_layers)-1:
        #             e[i], r[i] = layer.step(r[i-1], e[i], r[i], e[i+1])
        #         else:
        #             e[i], r[i] = layer.step(r[i-1], e[i], r[i])

        e0, r0 = self.pc0.init_vars(batch_size)
        e1, r1 = self.pc1.init_vars(batch_size)
        e2, r2 = self.pc2.init_vars(batch_size)
        e3, r3 = self.pc3.init_vars(batch_size)
        e4, r4 = self.pc4.init_vars(batch_size)
        e5, r5 = self.pc5.init_vars(batch_size)
        e6, r6 = self.pc6.init_vars(batch_size)
        e7, r7 = self.pc7.init_vars(batch_size)
        e8, r8 = self.pc8.init_vars(batch_size)
        e9, r9 = self.pc9.init_vars(batch_size)
        e10, r10 = self.pc10.init_vars(batch_size)
        e11, r11 = self.pc11.init_vars(batch_size)
        e12, r12 = self.pc12.init_vars(batch_size)
        
        for _ in range(self.steps):
            e0, r0 = self.pc0.step(x, e0, r0, e1)
            e1, r1 = self.pc.step(r0, e1, r1, e2)
            e2, r2 = self.pc.step(r1, e2, r2, e3)
            e3, r3 = self.pc.step
            e4, r4 = self.pc.step
            e5, r5 = self.pc.step
            e6, r6 = self.pc.step
            e7, r7 = self.pc.step
            e8, r8 = self.pc.step
            self.pc.step
            self.pc.step
            self.pc.step
            self.pc.step
        
        out = self.classifier(r[-1])

        e_mag = 0.0
        n = 0
        if calc_e_mag:
            for errs in e:
                e_mag += errs.abs().sum()
                n += errs.numel()
            e_mag /= n

        return [out, e_mag]