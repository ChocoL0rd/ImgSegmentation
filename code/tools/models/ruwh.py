import torch
from torch import nn
import torchvision.models as models
import torch.utils.checkpoint as cp

import os

from .model_parts.upsample_block import UpsampleBlock
from .model_parts.my_parts import *
from .model_parts.dense_block import _DenseBlock

from collections import OrderedDict


class RUwH(nn.Module):
    """ Resnet101 Unet with Head"""
    def __init__(self, cfg):
        super(RUwH, self).__init__()

        # body
        body_cfg = cfg.body
        from ..build_model import model_cfg2model
        model = model_cfg2model(body_cfg)
        if body_cfg.pretrained_model:
            model.load_state_dict(torch.load(os.path.join(body_cfg.pretrained_path, "model.pt")))

        self.in_batch_norm = model.in_batch_norm

        self.encoder = nn.ModuleDict({
            "layer0": nn.Sequential(OrderedDict([
                ("conv1", model.backbone.conv1),
                ("bn1", model.backbone.bn1),
                ("relu", model.backbone.relu),
            ])),
            "layer1": nn.Sequential(OrderedDict([
                ("maxpool", model.backbone.maxpool),
                ("backbone_layer1", model.backbone.layer1)
            ])),
            "layer2": model.backbone.layer2,
            "layer3": model.backbone.layer3,
            "layer4": model.backbone.layer4
        })

        self.decoder = model.upsample_blocks

        # head
        if cfg.head.name == "res":
            self.neck = NeckSequence(body_cfg.up_chnls[4], cfg.head.length, cfg.head.use_bn)
            out_layers = body_cfg.up_chnls[4]
        elif cfg.head.name == "dense":
            self.neck = _DenseBlock(cfg.head.length, body_cfg.up_chnls[4], cfg.head.bn_size, cfg.head.growth_rate,
                                    cfg.head.drop_rate, cfg.head.efficient)
            out_layers = body_cfg.up_chnls[4] + cfg.head.length * cfg.head.growth_rate
        else:
            raise Exception("wrong neck")

        self.final_conv = nn.Conv2d(in_channels=out_layers, out_channels=1, kernel_size=1)

        if cfg.head.freeze_neck:
            for param in self.neck.parameters():
                param.requires_grad = False

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

        dec_x = self.decoder[0](dec_x, skip_sizes[0], skips[0])
        dec_x = self.decoder[1](dec_x, skip_sizes[1], skips[1])
        dec_x = self.decoder[2](dec_x, skip_sizes[2], skips[2])
        dec_x = self.decoder[3](dec_x, skip_sizes[3], skips[3])
        dec_x = self.decoder[4](dec_x, original_size)

        return dec_x

    def pre_forward(self, x):
        original_size = x.shape[-2:]
        # Encoder
        middle, skips, skip_sizes = self.encode(x)
        # Decoder
        x = self.decode(middle, skips, skip_sizes, original_size)
        del middle, skips, skip_sizes
        x = self.neck(x)
        return x

    def forward(self, x):
        x = self.pre_forward(x)
        # Predicts
        x = self.final_conv(x)
        return x

    def inference(self, x):
        x = self.forward(x)
        return nn.Sigmoid()(x)


