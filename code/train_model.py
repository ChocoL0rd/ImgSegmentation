import os
import hydra
import pandas as pd
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
import json

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg):
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg.runtime.output_dir
    OmegaConf.resolve(cfg)

    if cfg.load_pretrained:
        if cfg.get_model_conf:
            new_cfg = OmegaConf.load(os.path.join(cfg.pretrained_path, ".hydra/config.yaml"))
            cfg.model_conf = new_cfg.model_conf
            OmegaConf.resolve(cfg)

    model = cfg2model(cfg)

    sublists = None
    if cfg.load_pretrained:
        model.load_state_dict(torch.load(os.path.join(cfg.pretrained_path, "model.pt")))
        if cfg.load_sublists:
            sublists = {}
            with open(os.path.join(cfg.pretrained_path, "train_dataset.json")) as f:
                sublists["train"] = json.load(f)["imgs_list"]

            with open(os.path.join(cfg.pretrained_path, "validation_dataset.json")) as f:
                sublists["val"] = json.load(f)["imgs_list"]

            val_top = pd.read_excel(os.path.join(cfg.pretrained_path, "validation", "full_results.xlsx"),
                                    header=[0, 1], index_col=[0])
            # here some function that should change train and val according to some params
            sublists = cfg2sublists(cfg.sublist_conf, sublists, val_top)

    train_dataset, val_dataset = cfg2datasets(cfg, data_sublists=sublists)
    for dataset in [train_dataset, val_dataset]:
        dataset.img_list2file(output_dir)

    cfg2fit(model, train_dataset, val_dataset, cfg)

    torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))

    test(model, train_dataset, output_dir, cfg)
    test(model, val_dataset, output_dir, cfg)


if __name__ == "__main__":
    my_app()
