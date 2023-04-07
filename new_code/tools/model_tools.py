# Here all models and their names are contained
from .models.resnet101_unet import ResNet101UNet

import os

__all__ = [
    "cfg2model"
]

# model_name: model
name2model_class = {
    "resnet101_unet": ResNet101UNet
}


def cfg2model(cfg):
    """ Returns model according to it config """
    model = name2model_class[cfg.name](cfg).to(cfg.device)
    if cfg.load_pretrained:
        model.load_state_dict(cfg.load(os.path.join(cfg.pretrained_path, "model.pt")))

    return model


