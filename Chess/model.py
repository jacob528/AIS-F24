import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

class ThunderByteCNN(nn.Module):
    def __init__(self):
        super(ThunderByteCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(14, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        # Policy head output size should match the number of squares on the board or legal moves
        self.policy_head = nn.Conv2d(256, 64, kernel_size=1)  
        
        # Value head for scalar evaluation 
        self.value_head = nn.Linear(256 * 8 * 8, 1)
            
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        
        # Policy head (output probabilities for each move)
        policy = self.policy_head(x).view(-1, 64)  # Flattened to 64 outputs (for 64 squares)
        
        # Value head (output scalar evaluation)
        value = self.value_head(x.view(x.size(0), -1))  # Flatten the conv output
        
        return policy, value
