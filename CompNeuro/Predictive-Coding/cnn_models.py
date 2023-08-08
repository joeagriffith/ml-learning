
from torch import nn
from my_funcs import SpearMax

class LeNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 6, (5,5), padding=2),
            nn.MaxPool2d(2),
            nn.ReLU(),

            nn.Conv2d(6, 16, (5,5)),
            nn.MaxPool2d(2),
            nn.ReLU(),

            nn.Conv2d(16, 120, (5,5)),
            nn.AdaptiveMaxPool2d((1,1)),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        return self.net(x)

class LeNet_spearmax(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 6, (5,5), padding=2),
            nn.MaxPool2d(2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, (5,5)),
            nn.MaxPool2d(2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 120, (5,5)),
            nn.AdaptiveMaxPool2d((1,1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = SpearMax(out)
        out = self.conv2(out)
        out = SpearMax(out)
        out = self.conv3(out)
        return self.classifier(out) 