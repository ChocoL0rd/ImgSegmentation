import torch
from torch import nn
import torchvision.models as models
import torch.utils.checkpoint as cp

from .model_parts.upsample_block import UpsampleBlock
from .model_parts.my_parts import *

from collections import OrderedDict


class BRU(nn.Module):
    """ Bridged Resnet101 Unet"""
    def __init__(self, cfg):
        super(BRU, self).__init__()
        self.in_batch_norm = nn.BatchNorm2d(3) if cfg.in_batch_norm else lambda x: x
        if cfg.in_batch_norm and cfg.in_bn_freeze:
            self.in_batch_norm.weight.requires_grad = False
            self.in_batch_norm.bias.requires_grad = False

        if cfg.pretrained:
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

        bridge_in_chnl = cfg.bridge.in_chnl
        bridge_length = cfg.bridge.length
        bridge_use_bn = [i for i in cfg.bridge.use_bn]
        self.bridge = nn.Sequential(OrderedDict([
            ("in_conv", nn.Conv2d(in_channels=64, out_channels=bridge_in_chnl, kernel_size=3, padding=1, stride=1)),
            ("res_sequence", NeckSequence(bridge_in_chnl, bridge_length, bridge_use_bn))
        ]))

        if cfg.bridge.freeze_in_conv:
            for param in self.bridge["in_conv"].parameters():
                param.requires_grad = False

        for i, freeze in enumerate(cfg.bridge.freeze):
            if freeze:
                for param in self.bridge["res_sequence"][i].parameters():
                    param.requires_grad = False

        self.upsample_blocks = nn.ModuleList(
            [
                UpsampleBlock(2048, cfg.up_chnls[0], 1024, cfg.modes[0],
                              use_bn=cfg.use_bn[0], drop_value=cfg.drop_values[0]),
                UpsampleBlock(cfg.up_chnls[0], cfg.up_chnls[1], 512, cfg.modes[1],
                              use_bn=cfg.use_bn[1], drop_value=cfg.drop_values[1]),
                UpsampleBlock(cfg.up_chnls[1], cfg.up_chnls[2], 256, cfg.modes[2],
                              use_bn=cfg.use_bn[2], drop_value=cfg.drop_values[2]),
                UpsampleBlock(cfg.up_chnls[2], cfg.up_chnls[3], bridge_in_chnl, cfg.modes[3],
                              use_bn=cfg.use_bn[3], drop_value=cfg.drop_values[3]),
                UpsampleBlock(cfg.up_chnls[3], cfg.up_chnls[4], 0, cfg.modes[4],
                              use_bn=cfg.use_bn[4], drop_value=cfg.drop_values[4])
            ]
        )

        self.checkpoints = cfg.checkpoints

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

        x0 = self.in_batch_norm(x0)

        x0 = self.encoder["layer0"](x0)
        x1 = self.encoder["layer1"](x0)
        x2 = self.encoder["layer2"](x1)
        x3 = self.encoder["layer3"](x2)

        middle = self.encoder["layer4"](x3)

        # sizes to up properly
        size0 = x0.shape[-2:]
        size1 = x1.shape[-2:]
        size2 = x2.shape[-2:]
        size3 = x3.shape[-2:]

        return middle, [x3, x2, x1, x0], [size3, size2, size1, size0]

    def decode(self, middle, skips, skip_sizes, original_size):
        """
        :param middle: fmap after encoder
        :param skips: skip fmaps
        :param skip_sizes: sizes of skip fmaps
        :param original_size: size of input image
        :return: last feature map, that going to be conv-ed in 1 channel feature map to make predict
        """
        dec_x = middle
        # for i in range(len(skips)):
        #     dec_x = self.upsample_blocks[i](dec_x, skip_sizes[i], skips[i])

        # dec_x = self.upsample_blocks[0](dec_x, skip_sizes[0], skips[0])
        # dec_x = self.upsample_blocks[1](dec_x, skip_sizes[1], skips[1])
        # dec_x = self.upsample_blocks[2](dec_x, skip_sizes[2], skips[2])
        # dec_x = self.upsample_blocks[3](dec_x, skip_sizes[3], self.bridge.forward(skips[3]))
        # dec_x = self.upsample_blocks[4](dec_x, original_size)

        if self.checkpoints[0]:
            dec_x = cp.checkpoint(self.upsample_blocks[0], *(dec_x, skip_sizes[0], skips[0]))
        else:
            dec_x = self.upsample_blocks[0](dec_x, skip_sizes[0], skips[0])

        if self.checkpoints[1]:
            dec_x = cp.checkpoint(self.upsample_blocks[1], *(dec_x, skip_sizes[1], skips[1]))
        else:
            dec_x = self.upsample_blocks[1](dec_x, skip_sizes[1], skips[1])

        if self.checkpoints[2]:
            dec_x = cp.checkpoint(self.upsample_blocks[2], *(dec_x, skip_sizes[2], skips[2]))
        else:
            dec_x = self.upsample_blocks[2](dec_x, skip_sizes[2], skips[2])

        if self.checkpoints[3]:
            dec_x = cp.checkpoint(self.upsample_blocks[3], *(dec_x, skip_sizes[3], self.bridge.forward(skips[3])))
        else:
            dec_x = self.upsample_blocks[3](dec_x, skip_sizes[3], self.bridge.forward(skips[3]))

        if self.checkpoints[4]:
            dec_x = cp.checkpoint(self.upsample_blocks[4], *(dec_x, original_size))
        else:
            dec_x = self.upsample_blocks[4](dec_x, original_size)

        return dec_x

    def pre_forward(self, x):
        original_size = x.shape[-2:]
        # Encoder
        middle, skips, skip_sizes = self.encode(x)
        # Decoder
        x = self.decode(middle, skips, skip_sizes, original_size)
        return x

    def forward(self, x):
        x = self.pre_forward(x)
        # Predicts
        x = self.final_conv(x)
        return x

    def inference(self, x):
        x = self.forward(x)
        return nn.Sigmoid()(x)


