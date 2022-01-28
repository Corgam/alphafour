import torch
import torch.nn as nn
import torch.nn.functional as F

class Convblock(nn.Module):
    def __init__(self):
        super(Convblock, self).__init__()
        bn = nn.BatchNorm2d(42)
        self.conv = nn.Conv2d(1, 42, 3, stride=(1, 1), padding=1)

    def forward(self, value):
        value = self.bn(value)
        return nn.ReLU(self.conv(value))

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu'):
        super().__init__()
        self.expansion = 4
        self.in_channels, self.out_channels, self.activation = in_channels, out_channels, activation
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), stride=(1,1), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), stride=(1,1), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=(1,1), stride=(1,1), padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)

    def forward(self, value):
        value = self.conv1(value)
        value = self.bn1(value)
        value = nn.ReLU(value)
        value = self.conv2(value)
        value = self.bn2(value)
        value = nn.ReLU(value)
        value = self.conv3(value)
        value = self.bn3(value)
        return value

class OutBlock(nn.Module):
    def init(self):
        super(OutBlock, self).init()
        self.conv = nn.Conv2d(42, 3, kernel_size=(1,1)) # value head
        self.bn = nn.BatchNorm2d(3)
        self.fc1 = nn.Linear(367, 32)
        self.fc2 = nn.Linear(32, 1)

        self.conv1 = nn.Conv2d(42, 32, kernel_size=(1,1)) # policy head
        self.bn1 = nn.BatchNorm2d(32)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc = nn.Linear(6732, 7)

    def forward(self,s):
        v = F.relu(self.bn(self.conv(s))) # value head
        v = v.view(-1, 367)  # batch_size X channel X height X width
        v = F.relu(self.fc1(v))
        v = torch.tanh(self.fc2(v))

        p = F.relu(self.bn1(self.conv1(s))) # policy head
        p = p.view(-1, 6732)
        p = self.fc(p)
        p = self.logsoftmax(p).exp()
        return p, v

class AlphaNet(torch.nn.Module):
    '''
    Main class for the deep convolutional residual neural network for the connect four agent.
    Consists of one convolutional layer, followed by 19 residual layers and a fully connected layer at the end.
    '''

    def __init__(self) -> None:
        super(AlphaNet, self).__init__()
        self.convLayer = Convblock()
        self.resLayers = [ResBlock()] * 19
        self.fullLayer = OutBlock()

    # Parameters

    # Methods
    def forward(self, values):
        values = self.convLayer(values)
        for layerID in range(19):
            values = self.resLayers[layerID](values)
        values = self.fullLayer(values)
        return values

