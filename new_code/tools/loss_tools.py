import torch
from torch import nn
import logging

log = logging.getLogger(__name__)


def soft_dice(x, y, eps=0, reduction="mean"):
    if reduction == "mean":
        return 2 * (((x * y).sum([-1, -2]) + eps)/(x.sum([-1, -2]) + y.sum([-1, -2])) + eps).mean()
    elif reduction == "none":
        return 2 * (((x * y).sum([-1, -2]) + eps)/(x.sum([-1, -2]) + y.sum([-1, -2])) + eps)


def negative_ln_dice(x, y, eps=0.0000001):
    return -torch.log(2 * (((x * y).sum([-1, -2]) + eps) / (x.sum([-1, -2]) + y.sum([-1, -2])) + eps)).mean()


def soft_jaccard(x, y, eps=0.0000001):
    return 1 - (((x * y).sum([-1, -2]) + eps)/(x.sum([-1, -2]) + y.sum([-1, -2]) - (x * y).sum([-1, -2]) + eps)).mean()


def negative_ln_jaccard(x, y, eps=0.0000001):
    return -torch.log(((x * y).sum([-1, -2]) + eps)/(x.sum([-1, -2]) + y.sum([-1, -2]) - (x * y).sum([-1, -2]) + eps)).mean()


def cfg2mini_loss(cfg):
    name = cfg.name
    if name == "bce":
        loss = nn.BCELoss()
    elif name == "soft_dice":
        def loss(x, y):
            return 1 - soft_dice(x, y)
    elif name == "soft_jaccard":
        loss = soft_jaccard
    elif name == "inversed_soft_jaccard":
        def loss(x, y):
            x = 1 - x
            y = 1 - y
            return soft_jaccard(x, y)
    elif name == "attentive_jaccard":
        def loss(x, y):
            field = abs(x - y) > cfg.receptive_field
            new_x = x * field
            new_y = y * field
            return soft_jaccard(new_x, new_y)

    elif name == "negative_ln_dice":
        def loss(x, y):
            return -torch.log(soft_dice(x, y))
    elif name == "inversed_negative_ln_dice":  # inv_neg_ln_dice
        def loss(x, y):
            x = 1 - x
            y = 1 - y
            return -torch.log(soft_dice(x, y))
    elif name == "attentive_dice":
        def loss(x, y):
            field = abs(x - y) > cfg.receptive_field
            new_x = x * field
            new_y = y * field
            return 1 - soft_dice(new_x, new_y)
    elif name == "negative_ln_attentive_dice":
        def loss(x, y):
            field = abs(x - y) > cfg.receptive_field
            new_x = x * field
            new_y = y * field
            return negative_ln_dice(new_x, new_y)

    elif name == "symmetric_dice":
        def loss(x, y):
            return 2 - soft_dice(x, y) - soft_dice(1-x, 1-y)
    elif name == "negative_ln_jaccard":
        loss = negative_ln_jaccard
    elif name == "tversky_loss":
        bce = nn.BCELoss()

        def loss(x, y):
            d00 = (x*y).sum([-1, -2])
            d01 = (x*(1-y)).sum([-1, -2])
            d10 = ((1-x)*y).sum([-1, -2])

            return 1 - (d00/(d00 + cfg.alpha * d01 + cfg.beta * d10)).mean()
    else:
        log.critical(f"Loss {name} is wrong.")
        raise Exception(f"Loss {name} is wrong.")

    return loss


def cfg2loss(cfg):
    def loss(x, y):
        s = 0
        for mini_cfg in cfg:
            s += mini_cfg.weight * cfg2mini_loss(mini_cfg)(x, y)
        return s

    return loss
