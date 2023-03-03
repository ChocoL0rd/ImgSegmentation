import albumentations as A

from albumentations.augmentations.dropout.coarse_dropout import CoarseDropout
from albumentations.augmentations.transforms import PixelDropout
from albumentations.augmentations.dropout.channel_dropout import ChannelDropout
from albumentations.augmentations.transforms import ChannelShuffle
from albumentations.augmentations.crops.transforms import RandomCrop
from albumentations.augmentations.geometric.transforms import HorizontalFlip
from albumentations.augmentations.geometric.transforms import VerticalFlip
from albumentations.augmentations.geometric.resize import Resize
from albumentations.augmentations.transforms import ColorJitter
from albumentations.augmentations.blur.transforms import Blur
from albumentations.augmentations.transforms import ToGray
from albumentations.augmentations.transforms import RandomShadow

import logging

log = logging.getLogger(__name__)


__all__ = [
    "cfg2transform",
    "cfg2augmentation"
]


def cfg2transform(cfg):
    """
    Convert config to transformation
    """
    name = cfg.name
    if name == "coarse_dropout":
        trfm = CoarseDropout(max_holes=cfg.max_holes,
                             max_height=cfg.max_height,
                             max_width=cfg.max_width,
                             fill_value=cfg.fill_value,
                             p=cfg.p)
    elif name == "pixel_dropout":
        trfm = PixelDropout(dropout_prob=cfg.dropout_prob,
                            per_channel=cfg.per_channel,
                            drop_value=cfg.drop_value,
                            p=cfg.p)
    elif name == "channel_dropout":
        trfm = ChannelDropout(p=cfg.p)
    elif name == "channel_shuffle":
        trfm = ChannelShuffle(p=cfg.p)
    elif name == "random_crop":
        trfm = RandomCrop(height=cfg.height,
                          width=cfg.width,
                          always_apply=cfg.always_apply,
                          p=cfg.p)
    elif name == "vertical_flip":
        trfm = VerticalFlip(p=cfg.p)
    elif name == "horizontal_flip":
        trfm = HorizontalFlip(p=cfg.p)
    elif name == "resize":
        trfm = Resize(height=cfg.height,
                      width=cfg.width,
                      p=cfg.p)
    elif name == "blur":
        trfm = Blur(p=cfg.p)
    elif name == "color_jitter":
        trfm = ColorJitter(brightness=cfg.brightness,
                           contrast=cfg.contrast,
                           saturation=cfg.saturation,
                           p=cfg.p)
    elif name == "to_gray":
        trfm = ToGray(p=cfg.p)

    elif name == "random_shadow":
        trfm = RandomShadow(shadow_roi=cfg.shadow_roi,
                            num_shadows_lower=cfg.num_shadows_lower,
                            num_shadows_upper=cfg.num_shadows_upper,
                            shadow_dimension=cfg.shadow_dimension,
                            p=cfg.p)
    else:
        msg = f"Transform {name} is wrong."
        log.critical(msg)
        raise Exception(msg)

    log.debug(f"Transform {trfm} is created.")
    return trfm


def cfg2augmentation(cfg):
    """
    Convert config to augmentation as Composition of transformations.
    """
    trfm_list = []
    for trfm_cfg in cfg:
        trfm = cfg2transform(trfm_cfg)
        if not trfm is None:
            trfm_list.append(trfm)

    return A.Compose(trfm_list)