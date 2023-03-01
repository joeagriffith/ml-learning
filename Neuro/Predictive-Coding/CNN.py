import torch.nn as nn
import torch

class CNNModel(nn.Module):
    def __init__(self, input_channels, num_classes):
        super().__init__()
        # [Batch_size x Channels x 32 x 32]
        self.network = nn.Sequential(
            
            # [Batch_size x 64 x 32 x 32]
            nn.Conv2d(input_channels, 64, (3,3), padding='same'),
            nn.Dropout(0.1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3,3), padding='same'),
            nn.Dropout(0.1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=2), # [Batch_size x 128 x 16 x 16]
            nn.Conv2d(64, 128, (3,3), padding='same'),
            nn.Dropout(0.1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, (3,3), padding='same'),
            nn.Dropout(0.1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=2), # [Batch_size x 256 x 8 x 8]
            nn.Conv2d(128, 256, (3,3), padding='same'),
            nn.Dropout(0.1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3,3), padding='same'),
            nn.Dropout(0.1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3,3), padding='same'),
            nn.Dropout(0.1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
#             nn.MaxPool2d(kernel_size=2), # [Batch_size x 512 x 4 x 4]
#             nn.Conv2d(256, 512, (3,3), padding='same'),
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#             nn.Conv2d(512, 512, (3,3), padding='same'),
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#             nn.Conv2d(512, 512, (3,3), padding='same'),
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
            
#             nn.MaxPool2d(kernel_size=2), # [Batch_size x 512 x 2 x 2]
#             nn.Conv2d(512, 512, (3,3), padding='same'),
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#             nn.Conv2d(512, 512, (3,3), padding='same'),
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#             nn.Conv2d(512, 512, (3,3), padding='same'),
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
            
#             nn.MaxPool2d(kernel_size=2), # [Batch_size x 512 x 1 x 1]
        )
        
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 8 * 8, 256 * 8 * 8),
            nn.ReLU(),
            nn.Linear(256 * 8 * 8, num_classes),
        )

    def forward(self, x):
        out = self.network(x)
        out = torch.flatten(out, start_dim=1)
        out = self.fc(out)
        return [out]