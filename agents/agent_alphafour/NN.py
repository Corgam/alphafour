import torch
import torch.nn as nn
import torch.nn.functional as F

class Convblock(nn.Module):
    def __init__(self):
        super(Convblock, self).__init__()

        self.conv = nn.Conv2d(3, 128, 3, stride=(1, 1), padding=1)

    def forward(self, value):
        return nn.ReLU(self.conv(value))

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu'):
        super().__init__()
        self.in_channels, self.out_channels, self.activation = in_channels, out_channels, activation
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels*4, kernel_size=1, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels*4)

    def forward(self, value):
        copy = value
        value = self.conv1(value)
        value = self.bn1(value)
        value = nn.ReLU(value)
        value = self.conv2(value)
        value = self.bn2(value)
        value = nn.ReLU(value)
        value = self.conv3(value)
        value = self.bn3(value)

