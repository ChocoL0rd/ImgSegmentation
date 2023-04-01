import torch
import torch.nn as nn
import torchvision.models as models

from .model_parts.upsample_block import UpsampleBlock

from collections import OrderedDict


class ERU(nn.Module):
    """ Easy Resnet101 Unet"""
    def __init__(self, cfg):
        super(ERU, self).__init__()

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

        self.final_conv = nn.Conv2d(cfg.up_chnls[4], 1, kernel_size=1)

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

        dec_x = self.decoder[0](middle, sizes[0], skips[0])  # 41
        dec_x = self.decoder[1](dec_x, sizes[1], skips[1])  # 82
        dec_x = self.decoder[2](dec_x, sizes[2], skips[2])  # 165
        dec_x = self.decoder[3](dec_x, sizes[3])  # 330
        dec_x = self.decoder[4](dec_x, original_size)

        return dec_x

    def pre_forward(self, x):
        original_size = x.shape[-2:]
        # Encoder
        middle, skips, sizes = self.encode(x)
        # Decoder
        x = self.decode(middle, skips, sizes, original_size)
        return x

    def forward(self, x):
        x = self.pre_forward(x)
        # Predicts
        x = self.final_conv(x)
        return x

    def inference(self, x):
        x = self.forward(x)
        return nn.Sigmoid()(x)



