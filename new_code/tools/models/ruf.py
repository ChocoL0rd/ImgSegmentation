import torch
import torch.nn as nn
import torchvision.models as models

from torch.nn import functional as F
from .model_parts.upsample_block import UpsampleBlock

from collections import OrderedDict
import os


class RUF(nn.Module):
    """ Resnet101 Unet + FPN"""
    def __init__(self, cfg):
        super(RUF, self).__init__()

        if cfg.pretrained_backbone:
            backbone = models.resnet101(weights="ResNet101_Weights.DEFAULT")
        else:
            backbone = models.resnet101()

        self.encoder = nn.ModuleDict({
            "layer0": nn.Sequential(OrderedDict([
                ("conv1", backbone.conv1),
                ("bn1", backbone.bn1),
                ("relu", backbone.relu),
            ])),
            "layer1": nn.Sequential(OrderedDict([
                ("maxpool", backbone.maxpool),
                ("backbone_layer1", backbone.layer1)
            ])),
            "layer2": backbone.layer2,
            "layer3": backbone.layer3,
            "layer4": backbone.layer4
        })

        self.encoder_params = []
        for i in range(5):
            if cfg.freeze_encoder[i]:
                for param in self.encoder[f"layer{i}"].parameters():
                    param.requires_grad = False
            else:
                self.encoder_params.append(
                    self.encoder[f"layer{i}"].parameters()
                )

        self.decoder = nn.ModuleDict(
            {
                "upsample0": UpsampleBlock(2048, cfg.up_chnls[0], 1024, cfg.modes[0],
                                           use_bn=cfg.use_bn[0], drop_value=cfg.drop_values[0]),
                "upsample1": UpsampleBlock(cfg.up_chnls[0], cfg.up_chnls[1], 512, cfg.modes[1],
                                           use_bn=cfg.use_bn[1], drop_value=cfg.drop_values[1]),
                "upsample2": UpsampleBlock(cfg.up_chnls[1], cfg.up_chnls[2], 256, cfg.modes[2],
                                           use_bn=cfg.use_bn[2], drop_value=cfg.drop_values[2]),
                "upsample3": UpsampleBlock(cfg.up_chnls[2], cfg.up_chnls[3], 64, cfg.modes[3],
                                           use_bn=cfg.use_bn[3], drop_value=cfg.drop_values[3]),
                "upsample4": UpsampleBlock(cfg.up_chnls[3], cfg.up_chnls[4], 0, cfg.modes[4],
                                           use_bn=cfg.use_bn[4], drop_value=cfg.drop_values[4])
            }
        )

        self.decoder_params = []
        for i in range(5):
            if cfg.freeze_decoder[i]:
                for param in self.decoder[f"upsample{i}"].parameters():
                    param.requires_grad = False
            else:
                self.decoder_params.append(
                    self.decoder[f"upsample{i}"].parameters()
                )

        s = sum([i for i in cfg.pyramid_chnls])
        self.pyramid = nn.ModuleDict([
            (
                "convs",
                nn.ModuleList(
                    [
                        nn.Conv2d(cfg.up_chnls[0], cfg.pyramid_chnls[0], kernel_size=3, padding=1, stride=1),
                        nn.Conv2d(cfg.up_chnls[1], cfg.pyramid_chnls[1], kernel_size=3, padding=1, stride=1),
                        nn.Conv2d(cfg.up_chnls[2], cfg.pyramid_chnls[2], kernel_size=3, padding=1, stride=1),
                        nn.Conv2d(cfg.up_chnls[3], cfg.pyramid_chnls[3], kernel_size=3, padding=1, stride=1),
                        nn.Conv2d(cfg.up_chnls[4], cfg.pyramid_chnls[4], kernel_size=3, padding=1, stride=1),
                    ]
                )
             ),
            (
                "final_conv",
                nn.Conv2d(s, cfg.post_pyr_chnls, kernel_size=3, padding=1, stride=1)
            )
        ])

        self.pyramid_convs_params = []
        for i in range(5):
            if cfg.freeze_pyramid_convs[i]:
                for param in self.pyramid["convs"][i].parameters():
                    param.requires_grad = False
            else:
                self.pyramid_convs_params.append(self.pyramid["convs"][i].parameters())

        self.pyramid_final_conv_params = []
        if cfg.freeze_pyramid_final_conv:
            for param in self.pyramid["final_conv"].parameters():
                param.requires_grad = False
        else:
            self.pyramid_final_conv_params.append(self.pyramid["final_conv"].parameters())

        self.final_conv = nn.Conv2d(cfg.post_pyr_chnls, 1, kernel_size=1)
        self.final_conv_params = []
        if cfg.freeze_final_conv:
            for param in self.final_conv.parameters():
                param.requires_grad = False
        else:
            self.final_conv_params.append(self.final_conv.parameters())

    def encode(self, x0):
        """
        :param x0: original image
        :return: last feature map in encoder (middle), skip feature maps and their sizes
        """

        x0 = self.encoder["layer0"](x0)  # 330
        x1 = self.encoder["layer1"](x0)  # 165
        x2 = self.encoder["layer2"](x1)  # 82
        x3 = self.encoder["layer3"](x2)  # 41

        middle = self.encoder["layer4"](x3)  # 20

        # sizes to up properly
        size0 = x0.shape[-2:]
        size1 = x1.shape[-2:]
        size2 = x2.shape[-2:]
        size3 = x3.shape[-2:]

        return middle, [x3, x2, x1, x0], [size3, size2, size1, size0]

    def decode(self, middle, skips, sizes, original_size):
        """
        :param middle: fmap after encoder
        :param skips: skip fmaps
        :param sizes: sizes of skips
        :param original_size: size of input image
        :return: last feature map, that going to be conv-ed in 1 channel feature map to make predict
        """
        dec_outs = [None, None, None, None, None]
        dec_outs[0] = self.decoder["upsample0"](middle, sizes[0], skips[0])  # 41
        dec_outs[1] = self.decoder["upsample1"](dec_outs[0], sizes[1], skips[1])  # 82
        dec_outs[2] = self.decoder["upsample2"](dec_outs[1], sizes[2], skips[2])  # 165
        dec_outs[3] = self.decoder["upsample3"](dec_outs[2], sizes[3], skips[3])  # 330
        dec_outs[4] = self.decoder["upsample4"](dec_outs[3], original_size)

        return dec_outs

    def apply_pyramid(self, dec_outs, original_size):
        for i, dec_out in enumerate(dec_outs):
            dec_outs[i] = self.pyramid["convs"][i](F.interpolate(dec_out, original_size, mode="bilinear", align_corners=None))

        dec_outs = torch.cat(dec_outs, dim=1)
        return self.pyramid["final_conv"](dec_outs)

    def pre_forward(self, x):
        original_size = x.shape[-2:]
        # Encoder
        middle, skips, sizes = self.encode(x)
        # Decoder
        decoder_outs = self.decode(middle, skips, sizes, original_size)

        for i, out in enumerate(decoder_outs):
            decoder_outs[i] = F.interpolate(out, size=original_size, mode="bilinear", align_corners=None)

        return self.apply_pyramid(decoder_outs, original_size)

    def forward(self, x):
        x = self.pre_forward(x)
        # Predicts
        x = self.final_conv(x)
        return x

    def inference(self, x):
        x = self.forward(x)
        return nn.Sigmoid()(x)

    def get_params(self):
        return self.encoder_params + self.decoder_params + \
            self.pyramid_convs_params + self.pyramid_final_conv_params + \
            self.final_conv_params

