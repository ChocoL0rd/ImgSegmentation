from .model_parts.unet_parts import *
from .model_parts.my_parts import *

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class ResUNet(nn.Module):
    def __init__(self, cfg):
        super(ResUNet, self).__init__()

        self.inc = DoubleConv(3, cfg.first_ch)
        self.down1 = Down(cfg.first_ch, cfg.down_ch[0])
        self.down_neck1 = NeckSequence(cfg.down_ch[0], cfg.down_depth[0])
        self.down2 = Down(cfg.down_ch[0], cfg.down_ch[1])
        self.down_neck2 = NeckSequence(cfg.down_ch[1], cfg.down_depth[1])
        self.down3 = Down(cfg.down_ch[1], cfg.down_ch[2])
        self.down_neck3 = NeckSequence(cfg.down_ch[2], cfg.down_depth[2])
        factor = 2 if cfg.bilinear else 1
        self.down4 = Down(cfg.down_ch[2], cfg.middle_ch // factor)
        self.up1 = Up(cfg.middle_ch, cfg.up_ch[0] // factor, cfg.bilinear)
        self.up_neck1 = NeckSequence(cfg.up_ch[0] // factor, cfg.up_depth[0])
        self.up2 = Up(cfg.up_ch[0], cfg.up_ch[1] // factor, cfg.bilinear)
        self.up_neck2 = NeckSequence(cfg.up_ch[1] // factor, cfg.up_depth[1])
        self.up3 = Up(cfg.up_ch[1], cfg.up_ch[2] // factor, cfg.bilinear)
        self.up_neck3 = NeckSequence(cfg.up_ch[2] // factor, cfg.up_depth[2])
        self.up4 = Up(cfg.up_ch[2], cfg.last_ch, cfg.bilinear)
        self.outc = OutConv(cfg.last_ch, 1)

        if cfg.checkpoint:
            self.inc = checkpoint(self.inc)
            self.down1 = checkpoint(self.down1)
            self.down_neck1 = checkpoint(self.down_neck1)
            self.down2 = checkpoint(self.down2)
            self.down_neck2 = checkpoint(self.down_neck2)
            self.down3 = checkpoint(self.down3)
            self.down_neck3 = checkpoint(self.down_neck3)
            self.down4 = checkpoint(self.down4)
            self.up1 = checkpoint(self.up1)
            self.up_neck1 = checkpoint(self.up_neck1)
            self.up2 = checkpoint(self.up2)
            self.up_neck2 = checkpoint(self.up_neck2)
            self.up3 = checkpoint(self.up3)
            self.up_neck3 = checkpoint(self.up_neck3)
            self.up4 = checkpoint(self.up4)
            self.outc = checkpoint(self.outc)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2 = self.down_neck1(x2)
        x3 = self.down2(x2)
        x3 = self.down_neck2(x3)
        x4 = self.down3(x3)
        x4 = self.down_neck3(x4)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up_neck1(x)
        x = self.up2(x, x3)
        x = self.up_neck2(x)
        x = self.up3(x, x2)
        x = self.up_neck3(x)
        x = self.up4(x, x1)

        return self.outc(x)

    def inference(self, x):
        return torch.sigmoid(self.forward(x))