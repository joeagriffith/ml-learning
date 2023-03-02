import torch.nn as nn
from PCLayer import PCLayer

class PCLeNet(nn.Module):
    def __init__(self, nu=1.0, mu=1.0, eta=0.1, steps=5):
        super().__init__()

        shape = [
            (1, 32, 32),
            (6, 14, 14),
            (16, 5, 5),
            (120, 1, 1),
        ]

        self.steps = steps

        self.pc0 = PCLayer(shape[0], shape[1], (5,5), nu, mu, eta, maxpool=2, padding=2)
        self.pc1 = PCLayer(shape[1], shape[2], (5,5), nu, mu, eta, maxpool=2)
        self.pc2 = PCLayer(shape[2], shape[3], (5,5), nu, mu, eta)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        device = "cuda" if x.is_cuda else "cpu"


        e0, r0 = self.pc0.init_vars(batch_size)
        e1, r1 = self.pc1.init_vars(batch_size)
        e2, r2 = self.pc2.init_vars(batch_size)

        e0 = e0.to(device)
        e1 = e1.to(device)
        e2 = e2.to(device)

        r0 = r0.to(device)
        r1 = r1.to(device)
        r2 = r2.to(device)

        for _ in range(self.steps):
            e0, r0 = self.pc0(x, e0, r0, e1)
            e1, r1 = self.pc1(r0, e1, r1, e2)
            e2, r2 = self.pc2(r1, e2, r2)

        out = self.classifier(r2)

        e_mag = e0.square().sum().item() + e1.square().sum().item() + e2.square().sum().item()
        e_mag /= e0.numel() + e1.numel() + e2.numel()

        return [out, e_mag]
