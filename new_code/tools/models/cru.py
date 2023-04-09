import torch
from torch import nn
import torchvision.models as models
from collections import OrderedDict


class CRU(nn.Module):
    """ Concatenated Resnet101 Unet"""
    def __init__(self, cfg):
        super(CRU, self).__init__()
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

        from ..model_tools import cfg2model
        self.decoders = nn.ModuleList([])
        self.decoder_params = []
        dec_out_chnls = 0
        for decoder_cfg in cfg.decoders:
            model = cfg2model(decoder_cfg)
            dec_out_chnls += decoder_cfg.up_chnls[-1]
            self.decoder_params = self.decoder_params + model.decoder_params
            self.decoders.append(model.decoder)

        self.final_conv = nn.Conv2d(in_channels=dec_out_chnls, out_channels=1,
                                    kernel_size=1, stride=1, padding=0)

        self.final_params = []
        if cfg.freeze_final_conv:
            for param in self.final_conv.parameters():
                param.requires_grad = False
        else:
            self.final_params = [self.final_conv.parameters()]

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

        return middle, [x3, x2, x1, x0]

    def decode(self, middle, skips, original_size):
        res = []
        for decoder in self.decoders:
            dec_x = middle
            for i in range(len(skips)):
                dec_x = decoder[f"upsample{i}"](dec_x, skips[i].shape[-2:], skips[i])
            dec_x = decoder["upsample4"](dec_x, original_size)
            res.append(dec_x)
        return torch.cat(res, dim=1)

    def pre_forward(self, x):
        original_size = x.shape[-2:]
        # Encoder
        middle, skips = self.encode(x)
        # Decoder
        x = self.decode(middle, skips, original_size)
        return x

    def forward(self, x):
        x = self.pre_forward(x)
        x = self.final_conv(x)
        return x

    def inference(self, x):
        x = self.forward(x)
        return nn.Sigmoid()(x)

    def get_params(self):
        return self.encoder_params + self.decoder_params + self.final_params