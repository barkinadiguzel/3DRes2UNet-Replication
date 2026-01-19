import torch
import torch.nn as nn
from .blocks import DownBlock, UpBlock, Res2Block3D


class Res2UNet3D(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, base_channels=32):
        super().__init__()

        self.enc1 = DownBlock(in_channels, base_channels)
        self.enc2 = DownBlock(base_channels, base_channels * 2)
        self.enc3 = DownBlock(base_channels * 2, base_channels * 4)
        self.enc4 = DownBlock(base_channels * 4, base_channels * 8)

        self.bottleneck = Res2Block3D(base_channels * 8, base_channels * 16)

        self.dec4 = UpBlock(base_channels * 16, base_channels * 8)
        self.dec3 = UpBlock(base_channels * 8, base_channels * 4)
        self.dec2 = UpBlock(base_channels * 4, base_channels * 2)
        self.dec1 = UpBlock(base_channels * 2, base_channels)

        self.final = nn.Conv3d(base_channels, num_classes, kernel_size=1)

    def forward(self, x):
        s1, p1 = self.enc1(x)
        s2, p2 = self.enc2(p1)
        s3, p3 = self.enc3(p2)
        s4, p4 = self.enc4(p3)

        b = self.bottleneck(p4)

        d4 = self.dec4(b, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        out = self.final(d1)
        return out
