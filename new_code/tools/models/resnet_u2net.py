import torch
from torch import nn
import torchvision.models as models
import torch.nn.functional as F

from .model_parts.upsample_block import UpsampleBlock
from .model_parts.u2net_parts import RSU7, RSU6, RSU5, RSU4, RSU4F

from collections import OrderedDict


class ResNet101U2Net(nn.Module):
    def __init__(self, cfg):
        super(ResNet101U2Net, self).__init__()

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

        # 2048, 1024, 512, 256, 64
        self.decoder = nn.ModuleDict(
            {
                "stage0": RSU4F(2048, 512, 1024),
                "stage1": RSU4F(2048, 256, 512),
                "stage2": RSU4(1024, 128, 256),
                "stage3": RSU5(512, 64, 64),
                "stage4": RSU6(128, 32, 32)
            }
        )

        self.decoder_params = []
        for i in range(5):
            if cfg.freeze_decoder[i]:
                for param in self.decoder[f"stage{i}"].parameters():
                    param.requires_grad = False
            else:
                self.decoder_params.append(
                    self.decoder[f"stage{i}"].parameters()
                )

        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)
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
        """
        :param middle: fmap after encoder
        :param skips: skip fmaps
        :param original_size: size of input image
        :return: last feature map, that going to be conv-ed in 1 channel feature map to make predict
        """

        # x0 = self.decoder["stage0"](middle)
        # x0 = F.interpolate(x0, skips[0].shape[2:], mode="bilinear")
        #
        # x1 = self.decoder["stage1"](torch.cat([x0, skips[0]]))
        # x1 = F.interpolate(x1, skips[1].shape[2:], mode="bilinear")
        #
        # x2 = self.decoder["stage2"](torch.cat([x1, skips[1]]))
        # x2 = F.interpolate(x2, skips[2].shape[2:], mode="bilinear")
        #
        # x3 = self.decoder["stage3"](torch.cat([x2, skips[2]]))
        # x3 = F.interpolate(x3, skips[3].shape[2:], mode="bilinear")
        #
        # x4 = self.decoder["stage4"](torch.cat([x3, skips[3]]))
        # x4 = F.interpolate(x4, original_size, mode="bilinear")

        x = self.decoder["stage0"](middle)
        x = F.interpolate(x, skips[0].shape[2:], mode="bilinear")

        x = self.decoder["stage1"](torch.cat([x, skips[0]], dim=1))
        x = F.interpolate(x, skips[1].shape[2:], mode="bilinear")

        x = self.decoder["stage2"](torch.cat([x, skips[1]], dim=1))
        x = F.interpolate(x, skips[2].shape[2:], mode="bilinear")

        x = self.decoder["stage3"](torch.cat([x, skips[2]], dim=1))
        x = F.interpolate(x, skips[3].shape[2:], mode="bilinear")

        x = self.decoder["stage4"](torch.cat([x, skips[3]], dim=1))
        x = F.interpolate(x, original_size, mode="bilinear")

        return x

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
        param_list = self.encoder_params + self.decoder_params + self.final_params
        return param_list


class ResNet101U2Net2(nn.Module):
    def __init__(self, cfg):
        super(ResNet101U2Net2, self).__init__()

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

        # 2048, 1024, 512, 256, 64
        self.decoder = nn.ModuleDict(
            {
                "stage0": RSU4F(2048, 512, 1024),
                "stage1": RSU4F(2048, 256, 512),
                "stage2": RSU4(1024, 128, 256),
                "stage3": RSU5(512, 64, 64),
                "stage4": RSU6(128, 32, 32)
            }
        )

        self.decoder_params = []
        for i in range(5):
            if cfg.freeze_decoder[i]:
                for param in self.decoder[f"stage{i}"].parameters():
                    param.requires_grad = False
            else:
                self.decoder_params.append(
                    self.decoder[f"stage{i}"].parameters()
                )

        self.final_conv = nn.Sequential(
            RSU7(35, 8, 16),
            nn.Conv2d(16, 1, kernel_size=1)
        )
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
        """
        :param middle: fmap after encoder
        :param skips: skip fmaps
        :param original_size: size of input image
        :return: last feature map, that going to be conv-ed in 1 channel feature map to make predict
        """

        # x0 = self.decoder["stage0"](middle)
        # x0 = F.interpolate(x0, skips[0].shape[2:], mode="bilinear")
        #
        # x1 = self.decoder["stage1"](torch.cat([x0, skips[0]]))
        # x1 = F.interpolate(x1, skips[1].shape[2:], mode="bilinear")
        #
        # x2 = self.decoder["stage2"](torch.cat([x1, skips[1]]))
        # x2 = F.interpolate(x2, skips[2].shape[2:], mode="bilinear")
        #
        # x3 = self.decoder["stage3"](torch.cat([x2, skips[2]]))
        # x3 = F.interpolate(x3, skips[3].shape[2:], mode="bilinear")
        #
        # x4 = self.decoder["stage4"](torch.cat([x3, skips[3]]))
        # x4 = F.interpolate(x4, original_size, mode="bilinear")

        x = self.decoder["stage0"](middle)
        x = F.interpolate(x, skips[0].shape[2:], mode="bilinear")

        x = self.decoder["stage1"](torch.cat([x, skips[0]], dim=1))
        x = F.interpolate(x, skips[1].shape[2:], mode="bilinear")

        x = self.decoder["stage2"](torch.cat([x, skips[1]], dim=1))
        x = F.interpolate(x, skips[2].shape[2:], mode="bilinear")

        x = self.decoder["stage3"](torch.cat([x, skips[2]], dim=1))
        x = F.interpolate(x, skips[3].shape[2:], mode="bilinear")

        x = self.decoder["stage4"](torch.cat([x, skips[3]], dim=1))
        x = F.interpolate(x, original_size, mode="bilinear")

        return x

    def pre_forward(self, x):
        original_size = x.shape[-2:]
        # Encoder
        middle, skips = self.encode(x)
        # Decoder
        x = self.decode(middle, skips, original_size)
        return x

    def forward(self, orig_x):
        x = self.pre_forward(orig_x)
        x = self.final_conv(torch.cat([orig_x, x], dim=1))
        return x

    def inference(self, x):
        x = self.forward(x)
        return nn.Sigmoid()(x)

    def get_params(self):
        param_list = self.encoder_params + self.decoder_params + self.final_params
        return param_list


class ResNet101U2Net3(nn.Module):
    def __init__(self, cfg):
        super(ResNet101U2Net3, self).__init__()

        # resnet101 backboned encoder
        if cfg.pretrained_backbone:
            backbone = models.resnet101(weights="ResNet101_Weights.DEFAULT")
        else:
            backbone = models.resnet101()

        self.resnet_encoder = nn.ModuleDict({
            "layer0": nn.Sequential(OrderedDict([
                ("conv1", backbone.conv1),
                ("bn1", backbone.bn1),
                ("relu", backbone.relu),
            ])),  # 64, 165, 165
            "layer1": nn.Sequential(OrderedDict([
                ("maxpool", backbone.maxpool),
                ("backbone_layer1", backbone.layer1)
            ])),  # 256, 83, 83
            "layer2": backbone.layer2,  # 512, 42, 42
            "layer3": backbone.layer3,  # 1024, 21, 21
            "layer4": backbone.layer4  # 2048, 11, 11
        })

        self.resnet_encoder_params = []
        for i in range(5):
            if cfg.freeze_resnet_encoder[i]:
                for param in self.resnet_encoder[f"layer{i}"].parameters():
                    param.requires_grad = False
            else:
                self.resnet_encoder_params.append(
                    self.resnet_encoder[f"layer{i}"].parameters()
                )

        # U2-Net encoder
        self.u2net_encoder = nn.ModuleDict({
            "stage0": nn.Sequential(
                RSU7(3, 32, 64),
                nn.MaxPool2d(2, stride=2, ceil_mode=True)
            ),  # 64, 165, 165
            "stage1": nn.Sequential(
                RSU6(64, 32, 128),
                nn.MaxPool2d(2, stride=2, ceil_mode=True)
            ),  # 128, 83, 83
            "stage2": nn.Sequential(
                RSU5(128, 64, 256),
                nn.MaxPool2d(2, stride=2, ceil_mode=True)
            ),  # 256, 42, 42
            "stage3": nn.Sequential(
                RSU4(256, 128, 512),
                nn.MaxPool2d(2, stride=2, ceil_mode=True)
            ),  # 512, 21, 21
            "stage4": nn.Sequential(
                RSU4F(512, 256, 512),
                nn.MaxPool2d(2, stride=2, ceil_mode=True)
            ),  # 512, 11, 11
            "stage5": RSU4F(512, 256, 512)  # 512, 11, 11
        })

        self.u2net_encoder_params = []
        for i in range(5):
            if cfg.freeze_u2net_encoder[i]:
                for param in self.u2net_encoder[f"stage{i}"].parameters():
                    param.requires_grad = False
            else:
                self.u2net_encoder_params.append(
                    self.u2net_encoder[f"stage{i}"].parameters()
                )

        # 512,  512,  256, 128, 64
        # 2048, 1024, 512, 256, 64
        self.decoder = nn.ModuleDict(
            {
                "stage0": RSU4F(2048 + 512, 512, 1024),
                "stage1": RSU4F(1024 + 1024 + 512, 256, 512),
                "stage2": RSU4(512 + 512 + 256, 128, 256),
                "stage3": RSU5(256 + 256 + 128, 64, 64),
                "stage4": RSU6(64 + 64 + 64, 32, 32)
            }
        )

        self.decoder_params = []
        for i in range(5):
            if cfg.freeze_decoder[i]:
                for param in self.decoder[f"stage{i}"].parameters():
                    param.requires_grad = False
            else:
                self.decoder_params.append(
                    self.decoder[f"stage{i}"].parameters()
                )

        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)
        self.final_params = []
        if cfg.freeze_final_conv:
            for param in self.final_conv.parameters():
                param.requires_grad = False
        else:
            self.final_params = [self.final_conv.parameters()]

    def u2net_encode(self, x0):
        x0 = self.u2net_encoder["stage0"](x0)
        x1 = self.u2net_encoder["stage1"](x0)
        x2 = self.u2net_encoder["stage2"](x1)
        x3 = self.u2net_encoder["stage3"](x2)
        middle = self.u2net_encoder["stage4"](x3)

        return middle, [x3, x2, x1, x0]

    def resnet_encode(self, x0):
        """
        :param x0: original image
        :return: last feature map in encoder (middle), skip feature maps and their sizes
        """

        x0 = self.resnet_encoder["layer0"](x0)  # 330
        x1 = self.resnet_encoder["layer1"](x0)  # 165
        x2 = self.resnet_encoder["layer2"](x1)  # 82
        x3 = self.resnet_encoder["layer3"](x2)  # 41
        middle = self.resnet_encoder["layer4"](x3)  # 20

        return middle, [x3, x2, x1, x0]

    def decode(self, middle, skips, original_size):
        """
        :param middle: fmap after encoder
        :param skips: skip fmaps
        :param original_size: size of input image
        :return: last feature map, that going to be conv-ed in 1 channel feature map to make predict
        """

        # x0 = self.decoder["stage0"](middle)
        # x0 = F.interpolate(x0, skips[0].shape[2:], mode="bilinear")
        #
        # x1 = self.decoder["stage1"](torch.cat([x0, skips[0]]))
        # x1 = F.interpolate(x1, skips[1].shape[2:], mode="bilinear")
        #
        # x2 = self.decoder["stage2"](torch.cat([x1, skips[1]]))
        # x2 = F.interpolate(x2, skips[2].shape[2:], mode="bilinear")
        #
        # x3 = self.decoder["stage3"](torch.cat([x2, skips[2]]))
        # x3 = F.interpolate(x3, skips[3].shape[2:], mode="bilinear")
        #
        # x4 = self.decoder["stage4"](torch.cat([x3, skips[3]]))
        # x4 = F.interpolate(x4, original_size, mode="bilinear")

        x = self.decoder["stage0"](middle)
        x = F.interpolate(x, skips[0].shape[2:], mode="bilinear")

        x = self.decoder["stage1"](torch.cat([x, skips[0]], dim=1))
        x = F.interpolate(x, skips[1].shape[2:], mode="bilinear")

        x = self.decoder["stage2"](torch.cat([x, skips[1]], dim=1))
        x = F.interpolate(x, skips[2].shape[2:], mode="bilinear")

        x = self.decoder["stage3"](torch.cat([x, skips[2]], dim=1))
        x = F.interpolate(x, skips[3].shape[2:], mode="bilinear")

        x = self.decoder["stage4"](torch.cat([x, skips[3]], dim=1))
        x = F.interpolate(x, original_size, mode="bilinear")

        return x

    def pre_forward(self, x):
        original_size = x.shape[-2:]
        # Encoder
        resnet_middle, resnet_skips = self.resnet_encode(x)
        u2net_middle, u2net_skips = self.u2net_encode(x)
        middle = torch.cat([resnet_middle, u2net_middle], dim=1)
        skips = [torch.cat([resnet_skips[i], u2net_skips[i]], dim=1) for i in range(4)]
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
        param_list = self.resnet_encoder_params + self.u2net_encoder_params + self.decoder_params + self.final_params
        return param_list


