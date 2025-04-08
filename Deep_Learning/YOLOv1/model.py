import torch
import torch.nn as nn
from torchvision.models import alexnet

class YOLOv1(nn.Module):
    def __init__(self, num_classes: int, S: int, B: int, weights: str = None):
        super(YOLOv1, self).__init__()
        self.num_classes = num_classes
        self.S = S
        self.B = B
        self.cnn = alexnet(weights=weights).features
        self.avgpool = nn.AdaptiveAvgPool2d((S, S))
        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * S * S, 512 * S * S),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512 * S * S, S * S * (B * 5 + num_classes))
        )

    def apply_activations(self, x: torch.Tensor) -> torch.Tensor:
        # Apply sigmoid to the bounding box coordinates and objectness score
        x[..., :self.B * 5] = torch.sigmoid(x[..., :self.B * 5])
        # Apply softmax to the class probabilities
        x[..., self.B * 5:] = torch.softmax(x[..., self.B * 5:], dim=-1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x).view(x.size(0), self.S, self.S, self.B * 5 + self.num_classes)
        x = self.apply_activations(x)
        return x