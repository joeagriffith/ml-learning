import torch.nn as nn
from PCLayer import PCLayer

class PCLeNetv2(nn.Module):
    def __init__(self, input_shape, num_classes, nu=1.0, mu=1.0, eta=0.1, steps=5):
        super().__init__()

        shape = [
            input_shape,
            (6, input_shape[1]//2, input_shape[2]//2),
            (24, ((input_shape[1]//2)-4)//2, ((input_shape[2]//2)-4)//2),
            (120, ((input_shape[1]//2)-4)//2 - 4, ((input_shape[2]//2)-4)//2 - 4),
        ]

        self.steps = steps

        self.pc0 = PCLayer(shape[0], shape[1], (5,5), nu, mu, eta, maxpool=2, padding=2)
        self.pc1 = PCLayer(shape[1], shape[2], (5,5), nu, mu, eta, maxpool=2)
        self.pc2 = PCLayer(shape[2], shape[3], (5,5), nu, mu, eta)

        self.classifier = nn.Sequential(
            # nn.AdaptiveMaxPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def pc_len(self):
        return 3

    def init_vars(self, batch_size, device="cpu"):
        e0, r0 = self.pc0.init_vars(batch_size)
        e1, r1 = self.pc1.init_vars(batch_size)
        e2, r2 = self.pc2.init_vars(batch_size)

        e0 = e0.to(device)
        e1 = e1.to(device)
        e2 = e2.to(device)

        r0 = r0.to(device)
        r1 = r1.to(device)
        r2 = r2.to(device)

        return e0, e1, e2, r0, r1, r2

    def forward(self, x):
        batch_size = x.shape[0]
        device = "cuda" if x.is_cuda else "cpu"

        e0, e1, e2, r0, r1, r2 = self.init_vars(x.shape[0], device)

        for _ in range(self.steps):
            e0, r0 = self.pc0(x, e0, r0, e1)
            e1, r1 = self.pc1(r0, e1, r1, e2)
            e2, r2 = self.pc2(r1, e2, r2)

        out = self.classifier(r2)

        return out, [e0, e1, e2]
    
    

    def step(self, x, e0, e1, e2, r0, r1, r2):
        e0, r0 = self.pc0(x, e0, r0, e1)
        e1, r1 = self.pc1(r0, e1, r1, e2)
        e2, r2 = self.pc2(r1, e2, r2)
        
        return e0, e1, e2, r0, r1, r2

    def classify(self, r2):
        return self.classifier(r2)


