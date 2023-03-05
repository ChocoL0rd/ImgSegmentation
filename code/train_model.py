import os
import hydra
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import OmegaConf, open_dict
import logging
import hydra.core.hydra_config
import numpy as np

import torch

from tools.train_tools import *
from tools.build_model import *
from tools.test_tools import *

import random

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg):
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg.runtime.output_dir
    OmegaConf.resolve(cfg)
    model = cfg2model(cfg)
    if cfg.load_pretrained:
        model.load_state_dict(torch.load(os.path.join(cfg.pretrained_path, "model.pt")))
    train_dataset, val_dataset = cfg2datasets(cfg)
    cfg2fit(model, train_dataset, val_dataset, cfg)
    test(model, train_dataset, output_dir, cfg)
    test(model, val_dataset, output_dir, cfg)
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))


if __name__ == "__main__":
    my_app()
