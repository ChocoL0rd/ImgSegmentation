import hydra
from hydra.utils import get_original_cwd, to_absolute_path
import hydra.core.hydra_config
from omegaconf import OmegaConf

import torch
import numpy as np

import os
import logging


from tools.train_tools import cfg2fit
from tools.model_tools import cfg2model
from tools.test_tools import cfg2test
from tools.dataset_tools import cfg2datasets, datasets2json_file

import random

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="test_config")
def my_app(cfg):
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    save_path = hydra_cfg.runtime.output_dir
    OmegaConf.resolve(cfg)

    # creating model
    model = cfg2model(cfg.model_cfg)

    # creating datasets
    datasets = cfg2datasets(cfg.dataset_cfg)

    # saving imgs of each dataset in dir made by hydra
    datasets2json_file(datasets, save_path)

    for name, dataset in datasets.items():
        cfg2test(cfg.test_cfg, model, dataset)


if __name__ == "__main__":
    my_app()


