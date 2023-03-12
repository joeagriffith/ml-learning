import torch
import torch.nn as nn
import torch.nn.functional as F
from PCLayer import PCLayer
from PCFCLayer import PCFCLayer

class PCLeNet(nn.Module):
    def __init__(self, input_shape, num_classes, nu=1.0, mu=1.0, eta=0.1, steps=5, relu_errs=False):
        super().__init__()

        shape = [
            input_shape,
            (6, input_shape[1]//2, input_shape[2]//2),
            (16, ((input_shape[1]//2)-4)//2, ((input_shape[2]//2)-4)//2),
            (120, ((input_shape[1]//2)-4)//2 - 4, ((input_shape[2]//2)-4)//2 - 4),
        ]

        self.steps = steps

        self.pc0 = PCLayer(shape[0], shape[1], (5,5), nu, mu, eta, maxpool=2, padding=2, relu_errs=relu_errs)
        self.pc1 = PCLayer(shape[1], shape[2], (5,5), nu, mu, eta, maxpool=2, relu_errs=relu_errs)
        self.pc2 = PCLayer(shape[2], shape[3], (5,5), nu, mu, eta, relu_errs=relu_errs)

        self.classifier = nn.Sequential(
            nn.AdaptiveMaxPool2d((1,1)),
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

        e0, e1, e2, r0, r1, r2 = self.init_vars(batch_size, device)

        for _ in range(self.steps):
            e0, r0 = self.pc0(x, e0, r0, e1)
            e1, r1 = self.pc1(r0, e1, r1, e2)
            e2, r2 = self.pc2(r1, e2, r2)
        
        out = self.classify(r2)

        return out, [e0, e1, e2]

    def step(self, x, e0, e1, e2, r0, r1, r2):
        e0, r0 = self.pc0(x, e0, r0, e1)
        e1, r1 = self.pc1(r0, e1, r1, e2)
        e2, r2 = self.pc2(r1, e2, r2)
        
        return e0, e1, e2, r0, r1, r2

    def classify(self, r2):
        return self.classifier(r2)

class PCLeNetTidy(nn.Module):
    def __init__(self, input_shape, num_classes, nu=1.0, mu=1.0, eta=0.1, steps=5, relu_errs=False):
        super().__init__()

        shape = [
            input_shape,
            (6, input_shape[1]//2, input_shape[2]//2),
            (16, ((input_shape[1]//2)-4)//2, ((input_shape[2]//2)-4)//2),
            (120, ((input_shape[1]//2)-4)//2 - 4, ((input_shape[2]//2)-4)//2 - 4),
        ]

        self.steps = steps

        self.pclayers = [
            PCLayer(shape[0], shape[1], (5,5), nu, mu, eta, maxpool=2, padding=2, relu_errs=relu_errs),
            PCLayer(shape[1], shape[2], (5,5), nu, mu, eta, maxpool=2, relu_errs=relu_errs),
            PCLayer(shape[2], shape[3], (5,5), nu, mu, eta, relu_errs=relu_errs),
        ]

        self.classifier = nn.Sequential(
            nn.AdaptiveMaxPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def init_vars(self, batch_size, device="cpu"):
        e0, r0 = self.pclayers[0].init_vars(batch_size)
        e1, r1 = self.pclayers[1].init_vars(batch_size)
        e2, r2 = self.pclayers[2].init_vars(batch_size)

        if device != "cpu":
            e0 = e0.to(device)
            e1 = e1.to(device)
            e2 = e2.to(device)

            r0 = r0.to(device)
            r1 = r1.to(device)
            r2 = r2.to(device)

        return [e0, e1, e2], [r0, r1, r2]

    def hebbian_update(self, e, r, weight):
        for i in range(len(e)):
            forward_grad =

    def forward(self, x):
        batch_size = x.shape[0]
        device = "cuda" if x.is_cuda else "cpu"

        e0, e1, e2, r0, r1, r2 = self.init_vars(batch_size, device)

        for _ in range(self.steps):
            e0, r0 = self.pclayers[0](x, e0, r0, e1)
            e1, r1 = self.pclayers[1](r0, e1, r1, e2)
            e2, r2 = self.pclayers[2](r1, e2, r2)
        
        out = self.classify(r2)

        return out, [e0, e1, e2]

    def step(self, x, e, r):
        e[0], r[0] = self.pclayers[0](x, e[0], r[0], e[1])
        e[1], r[1] = self.pclayers[1](r[0], e[1], r[1], e[2])
        e[2], r[2] = self.pclayers[2](r[1], e[2], r[2])
        
        return e, r

    def classify(self, r_final):
        return self.classifier(r_final)


#  Uses PC layers for classifier
class PCLeNetv3(nn.Module):
    def __init__(self, input_shape, num_classes, nu=1.0, mu=1.0, eta=0.1, steps=5, relu_errs=False):
        super().__init__()

        shape = [
            input_shape,
            (6, input_shape[1]//2, input_shape[2]//2),
            (16, ((input_shape[1]//2)-4)//2, ((input_shape[2]//2)-4)//2),
            (120, ((input_shape[1]//2)-4)//2 - 4, ((input_shape[2]//2)-4)//2 - 4),
        ]
        self.num_classes = num_classes
        self.steps = steps

        self.pc0 = PCLayer(shape[0], shape[1], (5,5), nu, mu, eta, maxpool=2, padding=2, relu_errs=relu_errs)
        self.pc1 = PCLayer(shape[1], shape[2], (5,5), nu, mu, eta, maxpool=2, relu_errs=relu_errs)
        self.pc2 = PCLayer(shape[2], shape[3], (5,5), nu, mu, eta, relu_errs=relu_errs)

        self.pcfc1 = PCFCLayer(120, 84, nu, mu, eta, relu_errs=relu_errs)
        self.pcfc2 = PCFCLayer(84, num_classes, nu, mu, eta, relu_errs=relu_errs)


    def pc_len(self):
        return 5


    def init_vars(self, batch_size, device="cpu"):
        e0, r0 = self.pc0.init_vars(batch_size)
        e1, r1 = self.pc1.init_vars(batch_size)
        e2, r2 = self.pc2.init_vars(batch_size)

        e3, r3 = self.pcfc1.init_vars(batch_size)
        e4, r4 = self.pcfc2.init_vars(batch_size)

        e0 = e0.to(device)
        e1 = e1.to(device)
        e2 = e2.to(device)
        e3 = e3.to(device)
        e4 = e4.to(device)

        r0 = r0.to(device)
        r1 = r1.to(device)
        r2 = r2.to(device)
        r3 = r3.to(device)
        r4 = r4.to(device)

        return e0, e1, e2, e3, e4, r0, r1, r2, r3, r4


    def step(self, x, e0, e1, e2, e3, e4, r0, r1, r2, r3, r4):
        e0, r0 = self.pc0(x, e0, r0, e1)
        e1, r1 = self.pc1(r0, e1, r1, e2)
        e2, r2 = self.pc2(r1, e2, r2, e3.reshape(r2.shape))

        e3, r3 = self.pcfc1(torch.flatten(r2, start_dim=1), e3, r3, e4)
        e4, r4 = self.pcfc2(r3, e4, r4)
        
        return e0, e1, e2, e3, e4, r0, r1, r2, r3, r4


    def guided_forward(self, x, y):
        batch_size = x.shape[0]
        device = "cuda" if x.is_cuda else "cpu"

        e0, e1, e2, e3, e4, r0, r1, r2, r3, r4 = self.init_vars(batch_size, device)

        for _ in range(self.steps):
            r4 = y
            e0, e1, e2, e3, e4, r0, r1, r2, r3, r4 = self.step(x, e0, e1, e2, e3, e4, r0, r1, r2, r3, r4)

        return r4, [e0, e1, e2, e3, e4]


    def forward(self, x):
        batch_size = x.shape[0]
        device = "cuda" if x.is_cuda else "cpu"

        e0, e1, e2, e3, e4, r0, r1, r2, r3, r4 = self.init_vars(batch_size, device)

        for _ in range(self.steps):
            e0, e1, e2, e3, e4, r0, r1, r2, r3, r4 = self.step(x, e0, e1, e2, e3, e4, r0, r1, r2, r3, r4)

        return r4, [e0, e1, e2, e3, e4]

    def predict(self, y):
        batch_size = y.shape[0]
        device = "cuda" if y.is_cuda else "cpu"

        e0, e1, e2, e3, e4, r0, r1, r2, r3, _ = self.init_vars(batch_size, device)
        x = torch.zeros(e0.shape, device=device)

        for _ in range(self.steps):
            e0, e1, e2, e3, e4, r0, r1, r2, r3, _ = self.step(x, e0, e1, e2, e3, e4, r0, r1, r2, r3, y)
            x -= e0

        return x, [e0, e1, e2, e3, e4]

class PCLeNetv3Deep(nn.Module):
    def __init__(self, input_shape, num_classes, nu=1.0, mu=1.0, eta=0.1, steps=5, relu_errs=False):
        super().__init__()

        shape = [input_shape] # (1,28,28)
        shape.append((6,  shape[-1][1]//2, shape[-1][2]//2)) # (6,14,14)
        shape.append((16, shape[-1][1]//2, shape[-1][2]//2)) # (16,7,7)
        shape.append((64, shape[-1][1]-4, shape[-1][2]-4))   # (64,3,3)
        shape.append((120, shape[-1][1]-2, shape[-1][2]-2))  # (120,1,1) 

        self.num_classes = num_classes
        self.steps = steps

        self.pc0 = PCLayer(shape[0], shape[1], (5,5), nu, mu, eta, maxpool=2, padding=2, relu_errs=relu_errs)
        self.pc1 = PCLayer(shape[1], shape[2], (5,5), nu, mu, eta, maxpool=2, padding=2, relu_errs=relu_errs)
        self.pc2 = PCLayer(shape[2], shape[3], (5,5), nu, mu, eta, relu_errs=relu_errs)
        self.pc3 = PCLayer(shape[3], shape[4], (3,3), nu, mu, eta, relu_errs=relu_errs)

        self.pcfc1 = PCFCLayer(120, 84, nu, mu, eta, relu_errs=relu_errs)
        self.pcfc2 = PCFCLayer(84, num_classes, nu, mu, eta, relu_errs=relu_errs)


    def pc_len(self):
        return 6


    def init_vars(self, batch_size, device="cpu"):
        e0, r0 = self.pc0.init_vars(batch_size)
        e1, r1 = self.pc1.init_vars(batch_size)
        e2, r2 = self.pc2.init_vars(batch_size)
        e3, r3 = self.pc3.init_vars(batch_size)
        e4, r4 = self.pcfc1.init_vars(batch_size)
        e5, r5 = self.pcfc2.init_vars(batch_size)

        if device != "cpu":
            e0 = e0.to(device)
            e1 = e1.to(device)
            e2 = e2.to(device)
            e3 = e3.to(device)
            e4 = e4.to(device)
            e5 = e5.to(device)

            r0 = r0.to(device)
            r1 = r1.to(device)
            r2 = r2.to(device)
            r3 = r3.to(device)
            r4 = r4.to(device)
            r5 = r5.to(device)

        return e0, e1, e2, e3, e4, e5, r0, r1, r2, r3, r4, r5


    def step(self, x, e0, e1, e2, e3, e4, e5, r0, r1, r2, r3, r4, r5):
        e0, r0 = self.pc0(x, e0, r0, e1)
        e1, r1 = self.pc1(r0, e1, r1, e2)
        e2, r2 = self.pc2(r1, e2, r2, e3)
        e3, r3 = self.pc3(r2, e3, r3, e4.reshape(r3.shape))

        e4, r4 = self.pcfc1(torch.flatten(r3, start_dim=1), e4, r4, e5)
        e5, r5 = self.pcfc2(r4, e5, r5)
        
        return e0, e1, e2, e3, e4, e5, r0, r1, r2, r3, r4, r5


    def guided_forward(self, x, y):
        batch_size = x.shape[0]
        device = "cuda" if x.is_cuda else "cpu"

        e0, e1, e2, e3, e4, e5, r0, r1, r2, r3, r4, r5 = self.init_vars(batch_size, device)
        r5 = y

        for _ in range(self.steps):
            e0, e1, e2, e3, e4, e5, r0, r1, r2, r3, r4, _ = self.step(x, e0, e1, e2, e3, e4, e5, r0, r1, r2, r3, r4, r5)

        return [x,r5], [e0, e1, e2, e3, e4, e5]


    def forward(self, x):
        batch_size = x.shape[0]
        device = "cuda" if x.is_cuda else "cpu"

        e0, e1, e2, e3, e4, e5, r0, r1, r2, r3, r4, r5 = self.init_vars(batch_size, device)

        for _ in range(self.steps):
            e0, e1, e2, e3, e4, e5, r0, r1, r2, r3, r4, r5 = self.step(x, e0, e1, e2, e3, e4, e5, r0, r1, r2, r3, r4, r5)

        return r4, [e0, e1, e2, e3, e4, e5]

    def predict(self, y):
        batch_size = y.shape[0]
        device = "cuda" if y.is_cuda else "cpu"

        e0, e1, e2, e3, e4, e5, r0, r1, r2, r3, r4, _ = self.init_vars(batch_size, device)
        x = torch.zeros(e0.shape, device=device)

        for _ in range(self.steps):
            e0, e1, e2, e3, e4, e5, r0, r1, r2, r3, r4, _ = self.step(x, e0, e1, e2, e3, e4, e5, r0, r1, r2, r3, e4, y)
            x -= e0

        return x, [e0, e1, e2, e3, e4, e5]