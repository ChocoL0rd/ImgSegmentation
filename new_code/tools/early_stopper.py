import logging
import torch
import os

__all__ = [
    "EarlyStopper"
]

log = logging.getLogger(__name__)


class EarlyStopper:
    def __init__(self, cfg):
        self.patience = cfg.patience
        self.delta = cfg.delta
        self.metric_name = cfg.target_metric
        self.save_path = cfg.save_path

        self.counter = 0
        self.best_metric = None
        self.early_stop = False

    def __call__(self, metrics, model):
        val_metric = metrics[self.metric_name]

        if self.best_metric is None or val_metric < self.best_metric - self.delta:
            # if new loss better than best
            log.info(f"Validation metric changed from {self.best_metric} to {val_metric}. Saving best model.")
            self.best_metric = val_metric
            self.counter = 0
            torch.save(model.state_dict(), os.path.join(self.save_path, "model.pt"))
        else:
            # if not better
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                log.info("Early stopping.")
