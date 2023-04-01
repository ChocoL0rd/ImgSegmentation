import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F
from .model_parts.my_parts import NeckSequence


class UpsampleBlock(nn.Module):
    def __init__(self, ch_in, ch_out, skip_in, mode, length, use_bn=True):
        super(UpsampleBlock, self).__init__()

        self.mode = mode
        ch_in = ch_in + skip_in
        self.neck_sequence = NeckSequence(ch_in, length, use_bn)

        if use_bn:
            self.final = nn.Sequential(
                nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=3, bias=False),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True)
            )
        else:
            self.final = nn.Sequential(
                nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=3, bias=True),
                nn.ReLU(inplace=True)
            )

    def forward(self, x, size, skip_connection=None):
        x = F.interpolate(x, size=size, mode=self.mode, align_corners=None)

        if skip_connection is not None:
            x = torch.cat([x, skip_connection], dim=1)

        x = self.neck_sequence(x)
        x = self.final(x)
        return x


class MiddleBlock(nn.Sequential):
    def __init__(self, ch_in, ch_out, length, use_bn):
        super(MiddleBlock, self).__init__()

        self.neck_sequence = NeckSequence(ch_in, length, use_bn)

        if use_bn:
            self.final = nn.Sequential(
                nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=3, bias=False),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True)
            )
        else:
            self.final = nn.Sequential(
                nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=3, bias=True),
                nn.ReLU(inplace=True)
            )


class DeepResUnet(nn.Module):
    def __init__(self, cfg):
        super(DeepResUnet, self).__init__()
        if cfg.pretrained:
            self.resnet = models.resnet101(weights="ResNet101_Weights.DEFAULT")
        else:
            self.resnet = models.resnet101()

        self.layer0 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
        )
        self.max_pool = self.resnet.maxpool

        if cfg.freeze_encoder[0]:
            for param in self.layer0.parameters():
                param.requires_grad = False

        self.layer1 = self.resnet.layer1
        if cfg.freeze_encoder[1]:
            for param in self.layer1.parameters():
                param.requires_grad = False

        self.layer2 = self.resnet.layer2
        if cfg.freeze_encoder[2]:
            for param in self.layer2.parameters():
                param.requires_grad = False

        self.layer3 = self.resnet.layer3
        if cfg.freeze_encoder[3]:
            for param in self.layer3.parameters():
                param.requires_grad = False

        self.layer4 = self.resnet.layer4
        if cfg.freeze_encoder[4]:
            for param in self.layer4.parameters():
                param.requires_grad = False

        self.middle_work = MiddleBlock(2048, cfg.mid_chnl, cfg.mid_length, cfg.mid_use_bn)

        self.up0 = UpsampleBlock(cfg.mid_chnl, cfg.up_chnls[0], 1024,
                                 cfg.modes[0], cfg.up_length[0], use_bn=cfg.use_bn[0])
        self.up1 = UpsampleBlock(cfg.up_chnls[0], cfg.up_chnls[1], 512,
                                 cfg.modes[1], cfg.up_length[1], use_bn=cfg.use_bn[1])
        self.up2 = UpsampleBlock(cfg.up_chnls[1], cfg.up_chnls[2], 256,
                                 cfg.modes[2], cfg.up_length[2], use_bn=cfg.use_bn[2])
        self.up3 = UpsampleBlock(cfg.up_chnls[2], cfg.up_chnls[3], 64,
                                 cfg.modes[3], cfg.up_length[3], use_bn=cfg.use_bn[3])
        self.up4 = UpsampleBlock(cfg.up_chnls[3], cfg.up_chnls[4], 64,
                                 cfg.modes[4], cfg.up_length[4], use_bn=cfg.use_bn[4])

        # self.up3 = nn.Conv2d(2048, 1024, kernel_size=1)
        # self.up2 = nn.Conv2d(1024, 512, kernel_size=1)
        # self.up1 = nn.Conv2d(512, 256, kernel_size=1)
        # self.up0 = nn.Conv2d(256, 64, kernel_size=1)

        self.conv_last = nn.Conv2d(cfg.up_chnls[4], 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        x = self.layer0(x)
        x0 = self.max_pool(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        # sizes to up properly
        original_size = x.shape[-2:]
        size0 = x0.shape[-2:]
        size1 = x1.shape[-2:]
        size2 = x2.shape[-2:]
        size3 = x3.shape[-2:]

        # middle
        x4 = self.middle_work(x4)

        # Decoder
        dec_x = self.up0(x4, size3, x3)
        dec_x = self.up1(dec_x, size2, x2)
        dec_x = self.up2(dec_x, size1, x1)
        dec_x = self.up3(dec_x, size0, x0)
        dec_x = self.up4(dec_x, original_size, x)

        # Predictions
        x = self.conv_last(dec_x)
        return x

    def inference(self, x):
        x = self.forward(x)
        return nn.Sigmoid()(x)

