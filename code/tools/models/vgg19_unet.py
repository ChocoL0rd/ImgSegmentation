import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .model_parts.upsample_block import UpsampleBlock


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)


class VGG19Unet(nn.Module):
    def __init__(self, cfg):
        super(VGG19Unet, self).__init__()

        # Encoder
        if cfg.pretrained:
            vgg19 = models.vgg19_bn(weights="VGG19_BN_Weights.DEFAULT")
        else:
            vgg19 = models.vgg19_bn()

        self.backbone = nn.ModuleList()
        self.backbone.append(nn.Sequential(*list(vgg19.features.children())[:6]))  # 64
        self.backbone.append(nn.Sequential(*list(vgg19.features.children())[6:13]))  # 128
        self.backbone.append(nn.Sequential(*list(vgg19.features.children())[13:26]))  # 256
        self.backbone.append(nn.Sequential(*list(vgg19.features.children())[26:39]))  # 512
        self.backbone.append(nn.Sequential(*list(vgg19.features.children())[39:52]))  # 512
        self.backbone.append(nn.Sequential(*list(vgg19.features.children())[-1:]))  # maxpool

        for i in range(len(self.backbone)):
            if cfg.encoder_freeze:
                for param in self.backbone[i].parameters():
                    param.requires_grad = False

        self.bottleneck = DoubleConv(512, cfg.middle_chnls)

        self.upsample_blocks = nn.ModuleList(
            [
                UpsampleBlock(cfg.middle_chnls, cfg.up_chnls[0], 512, cfg.modes[0],
                              use_bn=cfg.use_bn[0], drop_value=cfg.drop_values[0]),
                UpsampleBlock(cfg.up_chnls[0], cfg.up_chnls[1], 512, cfg.modes[1],
                              use_bn=cfg.use_bn[1], drop_value=cfg.drop_values[1]),
                UpsampleBlock(cfg.up_chnls[1], cfg.up_chnls[2], 256, cfg.modes[2],
                              use_bn=cfg.use_bn[2], drop_value=cfg.drop_values[2]),
                UpsampleBlock(cfg.up_chnls[2], cfg.up_chnls[3], 128, cfg.modes[3],
                              use_bn=cfg.use_bn[3], drop_value=cfg.drop_values[3]),
                UpsampleBlock(cfg.up_chnls[3], cfg.up_chnls[4], 64, cfg.modes[4],
                              use_bn=cfg.use_bn[4], drop_value=cfg.drop_values[4])
            ]
        )

        self.final_conv = nn.Conv2d(cfg.up_chnls[4], 1, kernel_size=1)

    def encode(self, x0):
        """
        :param x0: original image
        :return: last feature map in encoder (middle), skip feature maps and their sizes
        """
        x0 = self.backbone[0](x0)
        x1 = self.backbone[1](x0)
        x2 = self.backbone[2](x1)
        x3 = self.backbone[3](x2)
        x4 = self.backbone[4](x3)
        middle = self.backbone[5](x4)
        middle = self.bottleneck(middle)

        # sizes to up properly
        size0 = x0.shape[-2:]
        size1 = x1.shape[-2:]
        size2 = x2.shape[-2:]
        size3 = x3.shape[-2:]
        size4 = x4.shape[-2:]

        return middle, [x4, x3, x2, x1, x0], [size4, size3, size2, size1, size0]

    def decode(self, middle, skips, skip_sizes):
        """
        :param middle: fmap after encoder
        :param skips: skip fmaps
        :param skip_sizes: sizes of skip fmaps
        :return: last feature map, that going to be conv-ed in 1 channel feature map to make predict
        """
        dec_x = middle
        for i in range(len(skips)):
            dec_x = self.upsample_blocks[i](dec_x, skip_sizes[i], skips[i])
        return dec_x

    def pre_forward(self, x):
        middle, skips, skip_sizes = self.encode(x)
        # Decoder
        x = self.decode(middle, skips, skip_sizes)
        return x

    def forward(self, x):
        x = self.pre_forward(x)
        # Predicts
        x = self.final_conv(x)
        return x

    def inference(self, x):
        x = self.forward(x)
        return nn.Sigmoid()(x)


"""
VGG(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace=True)
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace=True)
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace=True)
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace=True)
    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace=True)
    (16): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (17): ReLU(inplace=True)
    (18): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (19): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU(inplace=True)
    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU(inplace=True)
    (23): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (24): ReLU(inplace=True)
    (25): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (26): ReLU(inplace=True)
    (27): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): ReLU(inplace=True)
    (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (31): ReLU(inplace=True)
    (32): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (33): ReLU(inplace=True)
    (34): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (35): ReLU(inplace=True)
    (36): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
)
"""