import os
import hydra
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import OmegaConf, open_dict
import logging
import hydra.core.hydra_config

import torch

from tools.train_tools import *
from tools.build_model import *
from tools.test_tools import *

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg):
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg.runtime.output_dir
    OmegaConf.resolve(cfg)
    model = cfg2model(cfg)
    val_loader = fit(model, cfg)
    test(model, val_loader, os.path.join(output_dir, "val"), cfg)
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))


if __name__ == "__main__":
    my_app()
