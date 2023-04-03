from .models.base_unet import BaseUNet
from .models.res_unet import ResUNet
from .models.backboned_unet import BackbonedUnet
from .models.resnet101_unet import ResNet101UNet
from .models.multi_depth_unet import MultiDepthUnet
from .models.ww_net import WWNet
from .models.inception_block import InceptionBlock
from .models.vgg19_unet import VGG19Unet
from .models.nothingnet import NothingNet
from .models.remover import Remover
from .models.resnet101_deeplab import Resnet101DeepLab
from .models.sequence import Seq
from .models.bru import BRU
from .models.ruwh import RUwH
from .models.eru import ERU
from .models.peru import PERU
from .models.ruf import RUF

__all__ = [
    "cfg2model",
    "model_cfg2model"
]

name2model_class = {
    "base_unet": BaseUNet,
    "res_unet": ResUNet,
    "backboned_unet": BackbonedUnet,
    "resnet101_unet": ResNet101UNet,
    "multi_depth_unet": MultiDepthUnet,
    "ww_net": WWNet,
    "inception_block": InceptionBlock,
    "vgg19_unet": VGG19Unet,
    "nothing_net": NothingNet,
    "remover": Remover,
    "resnet101_deeplab": Resnet101DeepLab,
    "sequence": Seq,
    "bru": BRU,
    "ruwh": RUwH,
    "eru": ERU,
    "peru": PERU,
    "ruf": RUF,
}


def cfg2model(cfg):
    model_cfg = cfg.model_conf
    return name2model_class[model_cfg.name](model_cfg).to(model_cfg.device)


def model_cfg2model(model_cfg, to_device=False):
    if to_device:
        return name2model_class[model_cfg.name](model_cfg).to(model_cfg.device)
    else:
        return name2model_class[model_cfg.name](model_cfg)

