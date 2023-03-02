import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
import logging
from torchvision.utils import save_image

from tools.train_tools import *

from torch.utils.data import Dataset, DataLoader


# cs = ConfigStore.instance()
# cs.store(name="group_config", node=GroupConfig)

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg):
    # OmegaConf.resolve(cfg)
    # print(OmegaConf.to_yaml(cfg.train_conf.dataset))
    # train_set, val_set = cfg2datasets(cfg)
    # mask_folder = "tmp_mask"
    # img_folder = "tmp_img"
    #
    # loader = DataLoader(train_set, batch_size=1)
    #
    # for img, mask in loader:
    #     print(mask)
    #     print(mask.shape)
    #     break

    print(cfg.something)
    print(type(cfg.something))
    # k= 0
    # for i in train_set:
    #     save_image(i[0], img_folder+f"/{k}.jpeg")
    #     save_image(i[1], mask_folder+f"/{k}.jpeg")
    #     k += 1

    # augment = cfg2augmentation(cfg.train_conf.dataset)
    #




if __name__ == "__main__":
    my_app()


