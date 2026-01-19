import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=k, padding=p),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class SEBlock3D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y


class Res2Block3D(nn.Module):
    def __init__(self, in_channels, out_channels, scale=4):
        super().__init__()
        assert out_channels % scale == 0
        self.scale = scale
        self.width = out_channels // scale

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)

        self.convs = nn.ModuleList([
            nn.Conv3d(self.width, self.width, kernel_size=3, padding=1, bias=False)
            for _ in range(scale - 1)
        ])

        self.bns = nn.ModuleList([
            nn.BatchNorm3d(self.width) for _ in range(scale - 1)
        ])

        self.conv3 = nn.Conv3d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels)

        self.se = SEBlock3D(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        splits = torch.split(out, self.width, dim=1)

        ys = []
        ys.append(splits[0])

        for i in range(1, self.scale):
            if i == 1:
                y = self.convs[i-1](splits[i])
            else:
                y = self.convs[i-1](splits[i] + ys[i-1])

            y = self.bns[i-1](y)
            y = self.relu(y)
            ys.append(y)

        out = torch.cat(ys, dim=1)
        out = self.conv3(out)
        out = self.bn3(out)

        out = self.se(out)
        out += identity
        out = self.relu(out)

        return out


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = Res2Block3D(in_ch, out_ch)
        self.pool = nn.MaxPool3d(2)

    def forward(self, x):
        x = self.block(x)
        p = self.pool(x)
        return x, p


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.block = Res2Block3D(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.block(x)
        return x
