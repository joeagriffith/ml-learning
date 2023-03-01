import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 6, (5,5), padding=2),
            nn.MaxPool2d(2),
            nn.ReLU(),

            nn.Conv2d(6, 16, (5,5)),
            nn.MaxPool2d(2),
            nn.ReLU(),

            nn.Conv2d(16, 120, (5,5)),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        return self.net(x)

    