import torch
import torch.nn as nn
import torch.nn.functional as F

class ChessNet(nn.Module):
    def __init__(self, num_unique_moves):
        super(ChessNet, self).__init__()
        
        # Convolutional Layers (Feature Extraction)
        # Input: 12 channels (pieces). Output: 64 features.
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Fully Connected Layers (Decision Making)
        self.fc1 = nn.Linear(128 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, num_unique_moves) 

    def forward(self, x):
        # x shape: (Batch, 12, 8, 8)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = x.view(-1, 128 * 8 * 8) # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # Output raw logits (CrossEntropyLoss handles Softmax)
        
        return x