import logging
import torch
from torch import nn


log = logging.getLogger(__name__)


__all__ = [
    "cfg2metric",
    "cfg2metric_list",
    "cfg2stat_list",
    "num2threshold",
    "name2metric",
    "name2stat"
]


def cfg2metric(cfg):
    name = cfg.name
    threshold = cfg.threshold
    if threshold == 0:
        threshold_adding = "with no threshold"

        def set_threshold(x):
            return x

    elif type(threshold) in [int, float] and 0 < threshold < 1:
        threshold_adding = f"with threshold {threshold}"

        def set_threshold(x):
            new_x = x.clone()
            new_x[new_x >= 1-threshold] = 1
            new_x[new_x <= threshold] = 0
            return new_x

        # def set_threshold(x):
        #     new_x = x.clone()
        #     new_x[new_x > threshold] = 1
        #     new_x[new_x <= threshold] = 0
        #     return new_x
    else:
        msg = f"In metric {name} threshold {threshold} is wrong. " \
              f"threshold is a number between 0 and 0.5 or 0 if no threshold."
        # msg = f"In metric {name} threshold {threshold} is wrong. " \
        #       f"threshold is a number between 0 and 1 or 0 if no threshold."
        log.critical(msg)
        raise Exception(msg)

    if name == "mae":
        def new_metric(x, y):
            new_x = set_threshold(x)
            return abs(new_x-y).mean([-1, -2])

    elif name == "dice":
        def new_metric(x, y):
            new_x = set_threshold(x)
            return 1 - 2 * (new_x * y).sum([-1, -2]) / (new_x.sum([-1, -2]) + y.sum([-1, -2]))
    elif name == "jaccard":
        def new_metric(x, y):
            new_x = set_threshold(x)
            return 1 - (new_x*y).sum([-1, -2])/(new_x.sum([-1, -2])+y.sum([-1, -2]) - (new_x*y).sum([-1, -2]))
    else:
        msg = f"Metric {name} is wrong."
        log.critical(msg)
        raise Exception(msg)

    return [f"{name} {threshold_adding}", new_metric]


def name2metric(name):
    if name == "mae":
        def new_metric(x, y):
            return abs(x-y).mean([-1, -2])
    elif name == "bce":
        bce = nn.BCELoss(reduction="none")

        def new_metric(x, y):
            return bce(x, y).mean([-1, -2])
    elif name == "negative_ln_dice":
        def new_metric(x, y):
            return -torch.log(2 * (((x * y).sum([-1, -2]) + 0.000001) / (x.sum([-1, -2]) + y.sum([-1, -2])) + 0.000001))
    elif name == "dice":
        def new_metric(x, y):
            return 1 - 2 * (x * y).sum([-1, -2]) / (x.sum([-1, -2]) + y.sum([-1, -2]))
    elif name == "jaccard":
        def new_metric(x, y):
            return 1 - (x*y).sum([-1, -2])/(x.sum([-1, -2])+y.sum([-1, -2]) - (x*y).sum([-1, -2]))
    else:
        msg = f"Metric {name} is wrong."
        log.critical(msg)
        raise Exception(msg)

    return new_metric


def num2threshold(num):
    if num == 0:
        def set_threshold(x):
            return x

    elif type(num) in [int, float] and 0 < num < 1:
        def set_threshold(x):
            new_x = x.clone()
            new_x[new_x >= 1-num] = 1
            new_x[new_x <= num] = 0
            return new_x

        # def set_threshold(x):
        #     new_x = x.clone()
        #     new_x[new_x > num] = 1
        #     new_x[new_x <= num] = 0
        #     return new_x
    else:
        msg = f"Threshold {num} is wrong. " \
              f"threshold is a number between 0 and 0.5 or 0 if no threshold."
        # msg = f"Threshold {num} is wrong. " \
        #       f"threshold is a number between 0 and 1 or 0 if no threshold."
        log.critical(msg)
        raise Exception(msg)

    return set_threshold


def name2stat(name):
    if name == "mean":
        stat = lambda x: x.mean()
    elif name == "max":
        stat = lambda x: x.max()
    elif name == "min":
        stat = lambda x: x.min()
    elif name == "std":
        stat = lambda x: x.std()
    else:
        msg = f"Statistic name {name} is wrong."
        log.critical(msg)
        raise Exception(msg)

    return stat


def cfg2stats(cfg):
    """
    returns statistics list of dictionaries of format:
    {
        "metric": "...",
        "threshold": *some number*,
        "stat": "...",
        "metric_func": *callable*
        "threshold_func": *callable*
        "stat_func": *callable*
    }
    :param cfg:
    :return: list of dicts
    """
    stat_list = []
    metric_name = cfg.metric
    metric_func = name2metric(metric_name)
    for threshold in cfg.thresholds:
        thr_func = num2threshold(threshold)
        for stat in cfg.stats:
            stat_func = name2stat(stat)
            stat_dict = {
                "metric": metric_name,
                "threshold": threshold,
                "stat": stat,
                "metric_func": metric_func,
                "threshold_func": thr_func,
                "stat_func": stat_func
            }
            stat_list.append(stat_dict)

    return stat_list


def cfg2stat_list(cfg):
    stat_list = []
    for metric_cfg in cfg:
        stat_list = stat_list + cfg2stats(metric_cfg)
    return stat_list


def cfg2metric_list(cfg):
    """Return list [[metric_name, metric], ...]"""
    metric_list = []
    for metric_cfg in cfg:
        metric_list.append(cfg2metric(metric_cfg))
    return metric_list
