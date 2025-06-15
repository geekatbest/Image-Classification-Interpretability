# filepath: d:\ml_projects\Image-Classification-Interpretability\model.py
import torch.nn as nn
import torch.nn.functional as F

# LeNet-style CNN for grayscale FashionMNIST
class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)   # (1, 28, 28) -> (32, 28, 28)
        self.pool = nn.MaxPool2d(2, 2)                            # -> (32, 14, 14)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # -> (64, 14, 14)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 -> ReLU -> Pool
        x = self.pool(F.relu(self.conv2(x)))  # Conv2 -> ReLU -> Pool
        x = x.view(-1, 64 * 7 * 7)            # Flatten
        x = F.relu(self.fc1(x))               # FC1 + ReLU
        x = self.fc2(x)                       # FC2 (no activation here)
        return x