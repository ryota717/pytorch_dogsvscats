from torch import nn
import torch.nn.functional as F
import numpy as np

class conv_unit(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(conv_unit, self).__init__()
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        y = F.leaky_relu(self.bn1(self.c1(x)), negative_slope=0.1)
        return y

class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch, make_last=True):
        super(conv_block, self).__init__()
        self.make_last = make_last
        self.c1 = conv_unit(in_ch, out_ch, stride=1)
        self.c2 = conv_unit(out_ch, out_ch, stride=1)
        self.c3 = conv_unit(out_ch, out_ch, stride=1)
        if self.make_last:
            self.c4 = conv_unit(out_ch, out_ch, stride=2)
    def forward(self, x):
        y = self.c1(x)
        y = self.c2(y)
        y = self.c3(y)
        if self.make_last:
            y = self.c4(y)
        return y

class CNN(nn.Module):
    def __init__(self,):
        super(CNN, self).__init__()
        self.c1 = conv_block(3, 8)
        self.c2 = conv_block(8, 16)
        self.c3 = conv_block(16, 32)
        self.c4 = conv_block(32, 64, False)
        self.l1 = nn.Linear(64, 2)

    def forward(self, x):
        y = self.c1(x)
        y = self.c2(y)
        y = self.c3(y)
        y = self.c4(y)
        y = F.max_pool2d(y, y.size()[2:])
        y = y.view(-1,64)
        y = self.l1(y)
        return y

if __name__ == '__main__':
    model = CNN()
    print(model)
