import torch
import torch.nn as nn


class ConvBlock(nn.Module):

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet3D(nn.Module):
    """
    3-level 3D U-Net.
    Input:  (B, 2, D, H, W)  — normalized image + coarse mask
    Output: (B, 1, D, H, W)  — boundary probability map in [0, 1]
    """

    def __init__(self, in_channels: int = 2, base_channels: int = 16):
        super().__init__()
        c = base_channels

        # Encoder
        self.enc1 = ConvBlock(in_channels, c)
        self.enc2 = ConvBlock(c,     c * 2)
        self.enc3 = ConvBlock(c * 2, c * 4)

        self.pool = nn.MaxPool3d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(c * 4, c * 8)

        # Decoder
        self.up3   = nn.ConvTranspose3d(c * 8, c * 4, kernel_size=2, stride=2)
        self.dec3  = ConvBlock(c * 8, c * 4)

        self.up2   = nn.ConvTranspose3d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2  = ConvBlock(c * 4, c * 2)

        self.up1   = nn.ConvTranspose3d(c * 2, c, kernel_size=2, stride=2)
        self.dec1  = ConvBlock(c * 2, c)

        self.head  = nn.Conv3d(c, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Bottleneck
        b = self.bottleneck(self.pool(e3))

        # Decoder with skip connections
        d3 = self.dec3(torch.cat([self.up3(b),  e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return torch.sigmoid(self.head(d1))
