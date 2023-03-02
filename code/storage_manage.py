from tools.storage_tools import *
import logging
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg):
    img2mask(in_dir="../data/segmented_imgs", out_dir="../data/lk", clean=True)


if __name__ == "__main__":
    my_app()
