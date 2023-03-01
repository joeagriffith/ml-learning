import torch.nn as nn


def block(in_channels, out_channels, kernel_size, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs),
        nn.MaxPool2d(kernel_size=2),
        nn.ReLU(),
    )

class CNN(nn.Module):
    def __init__(self, input_shape, features, num_classes):
        super().__init__()

        features = [
            features,
            int(features*1.5),
            int(features*2.25),
            int(features*4),
            int(features*4),
        ]


        self.conv0 = block(input_shape[0], features[0], (5,5), padding="same")
        self.conv1 = block(features[0],    features[1], (3,3), padding="same")
        self.conv2 = block(features[1],    features[2], (3,3), padding="same")
        self.conv3 = block(features[2],    features[3], (3,3), padding="same")
        self.conv4 = block(features[3],    features[4], (3,3), padding="same")

        final_size = input_shape[1]//(2**5)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(features[4]*final_size*final_size, features[0]**2//2),
            nn.ReLU(),
            nn.Linear(features[0]**2//2, num_classes)
        )

    def forward(self, x):
        out = self.conv0(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.classifier(out)
        
        return out
