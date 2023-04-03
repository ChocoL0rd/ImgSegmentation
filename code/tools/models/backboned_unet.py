import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.nn import functional as F


def get_backbone(name, pretrained=True):

    """ Loading backbone, defining names for skip-connections and encoder output. """

    # loading backbone model
    if name == 'resnet18':
        if pretrained:
            backbone = models.resnet18(weights="ResNet18_Weights.DEFAULT")
        else:
            backbone = models.resnet18()
    elif name == 'resnet34':
        if pretrained:
            backbone = models.resnet34(weights="ResNet34_Weights.DEFAULT")
        else:
            backbone = models.resnet34()
    elif name == 'resnet50':
        if pretrained:
            backbone = models.resnet50(weights="ResNet50_Weights.DEFAULT")
        else:
            backbone = models.resnet50()
    elif name == 'resnet101':
        if pretrained:
            backbone = models.resnet101(weights="ResNet101_Weights.DEFAULT")
        else:
            backbone = models.resnet101()
    elif name == 'resnet152':
        if pretrained:
            backbone = models.resnet152(weights="ResNet152_Weights.DEFAULT")
        else:
            backbone = models.resnet152()
    elif name == 'vgg16':
        if pretrained:
            backbone = models.vgg16_bn(weights="VGG16_BN_Weights.DEFAULT").features
        else:
            backbone = models.vgg16_bn().features
    elif name == 'vgg19':
        if pretrained:
            backbone = models.vgg19_bn(weights="VGG19_BN_Weights.DEFAULT").features
        else:
            backbone = models.vgg19_bn().features
    elif name == 'densenet121':
        if pretrained:
            backbone = models.densenet121(weights="DenseNet121_Weights.DEFAULT").features
        else:
            backbone = models.densenet121().features
    elif name == 'densenet161':
        if pretrained:
            backbone = models.densenet161(weights="DenseNet161_Weights.DEFAULT").features
        else:
            backbone = models.densenet161().features
    elif name == 'densenet169':
        if pretrained:
            backbone = models.densenet169(weights="DenseNet169_Weights.DEFAULT").features
        else:
            backbone = models.densenet169().features
    elif name == 'densenet201':
        if pretrained:
            backbone = models.densenet201(weights="DenseNet201_Weights.DEFAULT").features
        else:
            backbone = models.densenet201().features
    elif name == 'unet_encoder':
        from .model_parts.unet_backbone import UnetEncoder
        backbone = UnetEncoder(3)
    else:
        raise NotImplemented('{} backbone model is not implemented so far.'.format(name))

    # specifying skip feature and output names
    if name.startswith('resnet'):
        feature_names = [None, 'relu', 'layer1', 'layer2', 'layer3']
        backbone_output = 'layer4'
    elif name == 'vgg16':
        feature_names = ['5', '12', '22', '32', '42']
        backbone_output = '43'
    elif name == 'vgg19':
        feature_names = ['5', '12', '25', '38', '51']
        backbone_output = '52'
    elif name.startswith('densenet'):
        feature_names = [None, 'relu0', 'denseblock1', 'denseblock2', 'denseblock3']
        backbone_output = 'denseblock4'
    elif name == 'unet_encoder':
        feature_names = ['module1', 'module2', 'module3', 'module4']
        backbone_output = 'module5'
    else:
        raise NotImplemented('{} backbone model is not implemented so far.'.format(name))

    return backbone, feature_names, backbone_output


class UpsampleBlock(nn.Module):

    def __init__(self, ch_in, size, ch_out=None, skip_in=0, use_bn=True, parametric=False):
        super(UpsampleBlock, self).__init__()

        self.parametric = parametric
        ch_out = ch_in/2 if ch_out is None else ch_out

        # first convolution: either transposed conv, or conv following the skip connection
        if parametric:

            # versions: kernel=4 padding=1, kernel=2 padding=0
            self.up = nn.ConvTranspose2d(in_channels=ch_in, out_channels=ch_out, kernel_size=(4, 4),
                                         stride=2, padding=1, output_padding=0, bias=(not use_bn))
            self.bn1 = nn.BatchNorm2d(ch_out) if use_bn else None
        else:
            self.up = None
            ch_in = ch_in + skip_in
            self.conv1 = nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=(3, 3),
                                   stride=1, padding=1, bias=(not use_bn))
            self.bn1 = nn.BatchNorm2d(ch_out) if use_bn else None

        self.relu = nn.ReLU(inplace=True)

        # second convolution
        conv2_in = ch_out if not parametric else ch_out + skip_in
        self.conv2 = nn.Conv2d(in_channels=conv2_in, out_channels=ch_out, kernel_size=(3, 3),
                               stride=1, padding=1, bias=(not use_bn))
        self.bn2 = nn.BatchNorm2d(ch_out) if use_bn else None

        self.size = size

    def forward(self, x, skip_connection=None):
        # print(f"upsample in size {x.shape}")
        # print(f"upsample expected size {self.size}")
        x = self.up(x) if self.parametric else F.interpolate(x, size=self.size, mode='bilinear',
                                                             align_corners=None)
        # print(f"upsample in size after up or interpolate {x.shape}")
        if self.parametric:
            x = self.bn1(x) if self.bn1 is not None else x
            x = self.relu(x)

        if skip_connection is not None:
            # print(f"upsample skip_connections {skip_connection.shape}")
            x = torch.cat([x, skip_connection], dim=1)

        if not self.parametric:
            x = self.conv1(x)
            x = self.bn1(x) if self.bn1 is not None else x
            x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x) if self.bn2 is not None else x
        x = self.relu(x)

        return x


class BackbonedUnet(nn.Module):

    """ U-Net (https://arxiv.org/pdf/1505.04597.pdf) implementation with pre-trained torchvision backbones."""

    def __init__(self, cfg):
        super(BackbonedUnet, self).__init__()

        backbone_name = cfg.backbone_name
        pretrained = cfg.pretrained
        encoder_freeze = cfg.encoder_freeze
        classes = cfg.classes
        decoder_filters = cfg.decoder_filters
        parametric_upsampling = cfg.parametric_upsampling
        shortcut_features = cfg.shortcut_features
        decoder_use_batchnorm = cfg.decoder_use_batchnorm

        self.backbone_name = backbone_name

        self.backbone, self.shortcut_features, self.bb_out_name = get_backbone(backbone_name, pretrained=pretrained)
        shortcut_chs, bb_out_chs, size = self.infer_skip_channels()
        if cfg.backbone_name.startswith('resnet'):
            size = [[330, 330], [165, 165], [83, 83], [42, 42], [21, 21]]

        if shortcut_features != 'default':
            self.shortcut_features = shortcut_features

        # build decoder part
        self.upsample_blocks = nn.ModuleList()

        decoder_filters = decoder_filters[:len(self.shortcut_features)]  # avoiding having more blocks than skip connections
        decoder_filters_in = [bb_out_chs] + list(decoder_filters[:-1])
        num_blocks = len(self.shortcut_features)
        for i, [filters_in, filters_out] in enumerate(zip(decoder_filters_in, decoder_filters)):
            # print(f'unet upsample_blocks[{i}] in: {filters_in}   out: {filters_out}')
            self.upsample_blocks.append(UpsampleBlock(filters_in,
                                                      ch_out=filters_out,
                                                      skip_in=shortcut_chs[num_blocks-i-1],
                                                      parametric=parametric_upsampling,
                                                      use_bn=decoder_use_batchnorm,
                                                      size=size[num_blocks-i-1])
                                        )

        self.final_conv = nn.Conv2d(decoder_filters[-1], classes, kernel_size=(1, 1))

        if encoder_freeze:
            self.freeze_encoder()

        self.replaced_conv1 = False  # for accommodating  inputs with different number of channels later

    def freeze_encoder(self):

        """ Freezing encoder parameters, the newly initialized decoder parameters are remaining trainable. """

        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, *input):

        """ Forward propagation in U-Net. """
        # применяет backbone
        x, features = self.forward_backbone(*input)

        for skip_name, upsample_block in zip(self.shortcut_features[::-1], self.upsample_blocks):
            skip_features = features[skip_name]
            x = upsample_block(x, skip_features)

        x = self.final_conv(x)
        return x

    def inference(self, *input):
        x = self.forward(*input)
        return nn.Sigmoid()(x)

    def forward_backbone(self, x):
        """ Forward propagation in backbone encoder network.  """
        # shortcut_features - список названий модулей в бэкбоуне
        features = {None: None} if None in self.shortcut_features else dict()

        for name, child in self.backbone.named_children():
            x = child(x)
            if name in self.shortcut_features:
                print(f"{name}: {x.shape[-2:]}")  # ================================
                features[name] = x
            if name == self.bb_out_name:
                print(f"{name}: {x.shape[-2:]}")
                break

        return x, features

    def infer_skip_channels(self):
        """ Getting the number of channels at skip connections and at the output of the encoder. """

        x = torch.zeros(1, 3, 330, 330)
        has_fullres_features = self.backbone_name.startswith('vgg') or self.backbone_name == 'unet_encoder'
        channels = [] if has_fullres_features else [0]  # only VGG has features at full resolution
        size = []
        # forward run in backbone to count channels (dirty solution but works for *any* Module)
        for name, child in self.backbone.named_children():
            tmp_size = [x.shape[-2], x.shape[-1]]
            x = child(x)
            if name in self.shortcut_features:
                channels.append(x.shape[1])
                size.append(tmp_size)
                # print(f" infere size {size}")
                # print(f" infere channels {channels}")
            if name == self.bb_out_name:
                out_channels = x.shape[1]
                size.append(tmp_size)
                # print(f" infere size {size}")
                # print(f" infere channels {channels}")
                break

        return channels, out_channels, size

    def get_pretrained_parameters(self):
        for name, param in self.backbone.named_parameters():
            if not (self.replaced_conv1 and name == 'conv1.weight'):
                yield param

    def get_random_initialized_parameters(self):
        pretrained_param_names = set()
        for name, param in self.backbone.named_parameters():
            if not (self.replaced_conv1 and name == 'conv1.weight'):
                pretrained_param_names.add('backbone.{}'.format(name))

        for name, param in self.named_parameters():
            if name not in pretrained_param_names:
                yield param
