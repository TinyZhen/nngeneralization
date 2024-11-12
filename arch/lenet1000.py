import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet1000(nn.Module):
    def __init__(self, input_size=784, num_classes=10):
        super(LeNet1000, self).__init__()
        self.fc1 = nn.Linear(1024, 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x