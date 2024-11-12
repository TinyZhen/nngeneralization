import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """A basic block for WRN with a width multiplier."""
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.0):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout_rate = dropout_rate

        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        if self.dropout_rate > 0:
            out = F.dropout(out, p=self.dropout_rate, training=self.training)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class WRN(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.3):
        super(WRN, self).__init__()
        self.in_channels = 16

        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # Define the layers with (4, 4, 4) blocks in each group
        self.layer1 = self._make_layer(BasicBlock, 16 * 10, 4, stride=1, dropout_rate=dropout_rate)  # Width multiplier of 10
        self.layer2 = self._make_layer(BasicBlock, 32 * 10, 4, stride=2, dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(BasicBlock, 64 * 10, 4, stride=2, dropout_rate=dropout_rate)

        # Fully connected layer
        self.fc = nn.Linear(64 * 10, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride, dropout_rate):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride=stride, dropout_rate=dropout_rate))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels, dropout_rate=dropout_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))  # Global average pooling
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x
