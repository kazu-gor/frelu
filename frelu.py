import torch
import torch.nn as nn

class FReLU(nn.Module):
    def __init__(self, in_c, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.comv = nn.Conv2d(in_c, in_c, kernel_size, stride, padding, groups=in_c)
        self.bn = nn.BatchNorm2d(in_c)

    def forward(self, x):
        tx = self.bn(self.conv(x))
        return torch.max(x, tx)