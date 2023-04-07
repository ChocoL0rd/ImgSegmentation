import torch
from torch import nn


def soft_dice(x, y):
    eps = 0.0000001
    return 2 * (((x * y).sum([-1, -2]) + eps) / (x.sum([-1, -2]) + y.sum([-1, -2])) + eps)


def neg_ln_dice(x, y):
    return -torch.log(soft_dice(x, y))


def inv_soft_dice(x, y):
    return soft_dice(1 - x, 1 - y)


def inv_neg_ln_dice(x, y):
    return -torch.log(inv_soft_dice(x, y))


def soft_jaccard(x, y):
    eps = 0.0000001
    return ((x * y).sum([-1, -2]) + eps) / (x.sum([-1, -2]) + y.sum([-1, -2]) - (x * y).sum([-1, -2]) + eps)


def neg_ln_jaccard(x, y):
    return -torch.log(soft_jaccard(x, y))


def inv_soft_jaccard(x, y):
    return soft_jaccard(1 - x, 1 - y)


def inv_neg_ln_jaccard(x, y):
    return -torch.log(inv_soft_jaccard(x, y))


name2func = {
    "soft_dice": lambda x, y: 1 - soft_dice(x, y),
    "neg_ln_dice": neg_ln_dice,
    "inv_soft_dice": inv_soft_dice,
    "inv_neg_ln_dice": inv_neg_ln_dice,

    "soft_jaccard": lambda x, y: 1 - soft_jaccard(x, y),
    "neg_ln_jaccard": neg_ln_jaccard,
    "inv_soft_jaccard": inv_soft_jaccard,
    "inv_neg_ln_jaccard": inv_neg_ln_jaccard,

    "bce": nn.BCELoss(reduction="none")
}
