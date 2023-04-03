import torch
import torch.nn as nn
import torchvision.models as models

from torch.nn import functional as F
from .model_parts.upsample_block import UpsampleBlock

from collections import OrderedDict


class PERU(nn.Module):
    """ Pyramid Easy Resnet101 Unet"""
    def __init__(self, cfg):
        super(PERU, self).__init__()

        if cfg.pretrained_encoder:
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

        for i in range(5):
            if cfg.freeze_encoder[i]:
                for param in self.encoder[f"layer{i}"].parameters():
                    param.requires_grad = False

        self.decoder = nn.ModuleList(
            [
                UpsampleBlock(2048, cfg.up_chnls[0], 1024, cfg.modes[0],
                              use_bn=cfg.use_bn[0], drop_value=cfg.drop_values[0]),
                UpsampleBlock(cfg.up_chnls[0], cfg.up_chnls[1], 512, cfg.modes[1],
                              use_bn=cfg.use_bn[1], drop_value=cfg.drop_values[1]),
                UpsampleBlock(cfg.up_chnls[1], cfg.up_chnls[2], 256, cfg.modes[2],
                              use_bn=cfg.use_bn[2], drop_value=cfg.drop_values[2]),
                UpsampleBlock(cfg.up_chnls[2], cfg.up_chnls[3], 0, cfg.modes[3],
                              use_bn=cfg.use_bn[3], drop_value=cfg.drop_values[3]),
                UpsampleBlock(cfg.up_chnls[3], cfg.up_chnls[4], 0, cfg.modes[4],
                              use_bn=cfg.use_bn[4], drop_value=cfg.drop_values[4])
            ]
        )

        for i in range(5):
            if cfg.freeze_decoder[i]:
                for param in self.upsample_blocks[i].parameters():
                    param.requires_grad = False
        s = cfg.up_chnls[1] + cfg.up_chnls[2] + cfg.up_chnls[3] + cfg.up_chnls[4]
        self.final_conv = nn.Conv2d(s, 1, kernel_size=1)
        # self.final_conv = nn.Conv2d(sum([i for i in cfg.up_chnls][1:]), 1, kernel_size=1)

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

        return middle, [x3, x2, x1], [size3, size2, size1, size0]

    def decode(self, middle, skips, sizes, original_size):
        """
        :param middle: fmap after encoder
        :param skips: skip fmaps
        :param sizes: sizes of skips
        :param original_size: size of input image
        :return: last feature map, that going to be conv-ed in 1 channel feature map to make predict
        """
        dec_outs = [None, None, None, None, None]
        dec_outs[0] = self.decoder[0](middle, sizes[0], skips[0])  # 41
        dec_outs[1] = self.decoder[1](dec_outs[0], sizes[1], skips[1])  # 82
        dec_outs[2] = self.decoder[2](dec_outs[1], sizes[2], skips[2])  # 165
        dec_outs[3] = self.decoder[3](dec_outs[2], sizes[3])  # 330
        dec_outs[4] = self.decoder[4](dec_outs[3], original_size)

        return dec_outs

    def pre_forward(self, x):
        original_size = x.shape[-2:]
        # Encoder
        middle, skips, sizes = self.encode(x)
        # Decoder
        decoder_outs = self.decode(middle, skips, sizes, original_size)[1:]

        for i, out in enumerate(decoder_outs):
            decoder_outs[i] = F.interpolate(out, size=original_size, mode="bilinear", align_corners=None)

        return torch.cat(decoder_outs, dim=1)

    def forward(self, x):
        x = self.pre_forward(x)
        # Predicts
        x = self.final_conv(x)
        return x

    def inference(self, x):
        x = self.forward(x)
        return nn.Sigmoid()(x)



