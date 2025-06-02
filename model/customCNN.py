import torch
import torch.nn as nn
import torch.nn.functional as F

class customCNN(nn.Module):
    def __init__(self):
        super(customCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 12 * 12, 1024)
        self.fc2 = nn.Linear(1024, 136)  # 68 landmarks, each with (x, y) => 68 * 2 = 136
    
    def forward(self, x):
        # Conv layers
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # Max pooling (downsample)
        
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        
        # Flatten before passing to fully connected layers
        x = x.view(x.size(0), -1)  # Flatten the output of the last convolutional layer
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # Output 136 values (68 landmarks, 2 values for each: x and y)

        return x

