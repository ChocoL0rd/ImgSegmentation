import logging

log = logging.getLogger(__name__)


__all__ = [
    "cfg2metric",
    "cfg2metric_list"
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
            new_x[new_x > threshold] = 1
            new_x[new_x < threshold] = 0
            return new_x
    else:
        msg = f"In metric {name} threshold {threshold} is wrong. " \
              f"threshold is a number between 0 and 1 or 0 if no threshold."
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


def cfg2metric_list(cfg):
    """Return list [[metric_name, metric], ...]"""
    metric_list = []
    for metric_cfg in cfg:
        metric_list.append(cfg2metric(metric_cfg))
    return metric_list
