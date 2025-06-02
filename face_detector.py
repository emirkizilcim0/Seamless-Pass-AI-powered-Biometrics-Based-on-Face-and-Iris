import torch
import torch.nn as nn

class FaceDetectorCNN(nn.Module):
    def __init__(self):
        super(FaceDetectorCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),  # Input channels: 3 (RGB), Output: 16
            nn.ReLU(),
            nn.MaxPool2d(2),  # Downsample by 2
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 12 * 12, 128),  # Adjust input size based on image dimensions
            nn.ReLU(),
            nn.Linear(128, 5)  # Output: [x_center, y_center, width, height, confidence]
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x
