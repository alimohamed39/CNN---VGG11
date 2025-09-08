import torch
import torch.nn as nn
import torch.nn.functional as F


class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.features = nn.Sequential(

            nn.Conv2d(1, 32, kernel_size=3, padding="same"),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding="same"),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding="same"),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

         #   nn.Conv2d(256, 512, kernel_size=3, padding="same"),
          #  nn.BatchNorm2d(512),
           # nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Dropout(0.3)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 1000),
            nn.ReLU(),
            nn.Dropout(0.3),
        #    nn.Linear(4096, 1000),
            nn.ReLU(),
            nn.Linear(1000, 20)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits


model = MyCNN()
