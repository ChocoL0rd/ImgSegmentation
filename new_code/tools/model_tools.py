# Here all models and their names are contained
from .models.resnet101_unet import ResNet101UNet
from .models.cru import CRU
from .models.ruf import RUF
from .models.u2net import MiniU2Net, U2NET
from .models.resnet_u2net import ResNet101U2Net, ResNet101U2Net2, ResNet101U2Net3

import torch
import os

__all__ = [
    "cfg2model"
]

# model_name: model
name2model_class = {
    "resnet101_unet": ResNet101UNet,
    "cru": CRU,
    "ruf": RUF,
    "mini_u2net": MiniU2Net,
    "u2net": U2NET,
    "resnet_u2net": ResNet101U2Net,
    "resnet_u2net2": ResNet101U2Net2,
    "resnet_u2net3": ResNet101U2Net3
}


def cfg2model(cfg):
    """ Returns model according to it config """
    model = name2model_class[cfg.name](cfg).to(cfg.device)
    if cfg.load_pretrained:
        model.load_state_dict(torch.load(cfg.pretrained_path))

    return model



