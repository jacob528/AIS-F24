import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

class ThunderByteCNN(nn.Module):
    def __init__(self):
        #Initialize model
        super(ThunderByteCNN, self).__init__()
        
        #Initialize Convolutional layers
        self.conv1 = nn.Conv2d(14, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        #Fully connected layers for policy and value heads
        self.policy_head = nn.Conv2d(256, 2, kernel_size=1)  # 2 possible move outputs
        self.value_head = nn.Linear(256 * 8 * 8, 1)  # Output a single scalar for the value
            
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        
        # Policy head (output probabilities for each move)
        policy = self.policy_head(x).view(-1, 2)  # Flattened to 2 outputs
        
        # Value head (output a scalar evaluation)
        value = self.value_head(x.view(x.size(0), -1))  # Flatten the conv output
        
        return policy, value