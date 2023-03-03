from .models.base_unet import BaseUNet
from .models.res_unet import ResUNet

__all__ = [
    "cfg2model"
]

name2model_class = {
    "base_unet": BaseUNet,
    "res_unet": ResUNet
}


def cfg2model(cfg):
    model_cfg = cfg.model_conf
    return name2model_class[model_cfg.name](model_cfg).to(model_cfg.device)