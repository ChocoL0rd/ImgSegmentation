import torch
from torch import nn
import torch.optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.transforms import ToTensor

import numpy as np
from sklearn.model_selection import train_test_split

from PIL import Image
import cv2 as cv

from typing import Optional, Union, List
import os

import logging

from .metric_tools import *
from .augmentation_tools import *


__all__ = [
    "cfg2datasets",
    "cfg2loss",
    "cfg2optimizer",
    "fit",
    "ImgMaskSet",
]

# Here implemented 'fit' that trains model.

log = logging.getLogger(__name__)
img2tensor = ToTensor()


class ImgMaskSet(Dataset):
    """
    It's dataset that returns images and their masks (with the same file names).
    """
    def __init__(self, log_name: str, img_dir_path: str, mask_dir_path: str, img_list: Optional[List[str]],
                 transforms,  # add type of augmentation
                 device: torch.device):
        """
        :param log_name:  name that is used in log
        :param img_dir_path: path to directory where images are contained. in this directory all images are .jpeg
        :param mask_dir_path: path to directory where masks (images with deleted background) are contained.
        in this directory all images are .png
        :param img_list: specifies image names in image directory that should be used
        :param transforms: transformations to augment dataset
        :param device: device of images and masks
        """
        self.log_name = log_name

        self.local_log = logging.getLogger(__name__)
        self.local_log.setLevel(log.getEffectiveLevel())
        self.local_log.debug(f"Local log for {self.log_name} dataset is created.")

        self.img_dir_path = img_dir_path
        self.mask_dir_path = mask_dir_path
        self.device = device

        log.info(f"Creating {self.log_name} dataset on device {self.device}: \n"
                     f"Path to image dir: {self.img_dir_path} \n"
                     f"Path to mask dir: {self.mask_dir_path}")

        if img_list is None:
            log.info(f"List of images isn't specified.\n"
                     f"Get all images from the image directory: {img_dir_path}\n"
                     f"and mask directory {mask_dir_path}")
    #         suppose all files are in both directories
            self.img_list = [i[:-5] for i in os.listdir(img_dir_path)]  # deleted extension .jpeg
        else:
            self.img_list = [i[:-5] for i in img_list]  # deleted extension .jpeg

        self.transforms = transforms
        self.size = len(self.img_list)

        log.info(f"{self.log_name} dataset is created. Size: {self.size}")

    def __len__(self):
        return self.size

    def __getitem__(self, idx):

        img_path = os.path.join(self.img_dir_path, self.img_list[idx] + ".jpeg")
        mask_path = os.path.join(self.mask_dir_path, self.img_list[idx] + ".png")

        self.local_log.debug(f"Dataset {self.log_name} returns item with idx:{idx}\n"
                             f"img_path: {img_path}\n"
                             f"mask_path: {mask_path}")

        img = cv.imread(img_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # convert to RGB format
        # mask = cv.imread(mask_path)
        with Image.open(mask_path) as mask_im:
            mask = np.array(mask_im.split()[-1])  # retrieve transparent mask

        if self.transforms is None:
            img_tensor = img2tensor(img).to(torch.float32)
            mask_tensor = img2tensor(mask).to(torch.float32)
            return img_tensor.to(self.device), mask_tensor.to(self.device)

        # apply transformations
        transformed = self.transforms(image=img, mask=mask)
        self.local_log.debug("Transform is applied.")

        img_tensor = img2tensor(transformed["image"]).to(torch.float32)
        mask_tensor = img2tensor(transformed["mask"]).to(torch.float32)

        return img_tensor.to(self.device), mask_tensor.to(self.device)


def cfg2datasets(cfg):
    """
    Convert config to train and val datasets.
    """
    log_level = log.getEffectiveLevel()
    local_cfg = cfg.train_conf.dataset
    local_debug = local_cfg.debug
    if local_debug:
        log.setLevel(10)
        log.info("While creating datasets, debug level - is activated.")

    augment = cfg2augmentation(local_cfg.augmentation)
    device = torch.device(local_cfg.device)

    train_img_dir_name = cfg.train_img_dir_name
    train_mask_dir_name = cfg.train_mask_dir_name

    val_img_dir_name = cfg.val_img_dir_name
    val_mask_dir_name = cfg.val_mask_dir_name

    if train_img_dir_name == val_img_dir_name:
        if train_mask_dir_name != val_mask_dir_name:
            msg = f"Image dirs for train and val are the same, but mask dirs are different.\n"\
                         f"train image directory name: {train_img_dir_name}\n"\
                         f"train mask directory name: {train_mask_dir_name}\n"\
                         f"val image directory name: {val_img_dir_name}\n"\
                         f"val mask directory name: {val_mask_dir_name}"

            log.critical(msg)
            raise Exception(msg)

        img_path = local_cfg.train_img_dir
        mask_path = local_cfg.train_mask_dir
        train_proportion = local_cfg.train_proportion

        log.info(f"Get train and val data from the same directory:\n"
                 f"train directory: {img_path}\n"
                 f"validation directory: {mask_path}")

        train_list, val_list = train_test_split(os.listdir(img_path),
                                                train_size=train_proportion,
                                                shuffle=True)
        train_dataset = ImgMaskSet(log_name="train",
                                   img_dir_path=img_path, mask_dir_path=mask_path,
                                   img_list=train_list,
                                   transforms=augment, device=device)
        val_dataset = ImgMaskSet(log_name="validation",
                                 img_dir_path=img_path, mask_dir_path=mask_path,
                                 img_list=val_list,
                                 transforms=None, device=device)

    else:
        log.info(f"Get train and val data from different directories:\n"
                 f"train image directory name: {train_img_dir_name}\n"
                 f"train mask directory name: {train_mask_dir_name}\n"
                 f"val image directory name: {val_img_dir_name}\n"
                 f"val mask directory name: {val_mask_dir_name}")

        train_dataset = ImgMaskSet(log_name="train",
                                   img_dir_path=local_cfg.train_img_dir, mask_dir_path=local_cfg.train_mask_dir,
                                   img_list=None,
                                   transforms=augment, device=device)
        val_dataset = ImgMaskSet(log_name="validation",
                                 img_dir_path=local_cfg.val_img_dir, mask_dir_path=local_cfg.val_mask_dir,
                                 img_list=None,
                                 transforms=None, device=device)

    if local_debug:
        log.setLevel(log_level)
        log.info(f"After creating datasets, log level changed to {log_level}.")

    return train_dataset, val_dataset


def cfg2mini_loss(cfg):
    name = cfg.name
    if name == "bce":
        return nn.BCELoss()
    elif name == "soft_dice":
        def loss(x, y):
            return (1 - 2*(x*y).sum([-1, -2])/(x.sum([-1, -2])+y.sum([-1, -2]))).mean()
        return loss
    elif name == "soft_jaccard":
        def loss(x, y):
            return (1 - (x*y).sum([-1, -2])/(x.sum([-1, -2])+y.sum([-1, -2]) - (x*y).sum([-1, -2]))).mean()
        return loss
    else:
        log.critical(f"Loss {name} is wrong.")
        raise Exception(f"Loss {name} is wrong.")


def cfg2loss(cfg):
    def loss(x, y):
        s = 0
        for mini_cfg in cfg:
            s += mini_cfg.weight * cfg2mini_loss(mini_cfg)(x, y)
        return s
    return loss


def cfg2optimizer(model, cfg):
    name = cfg.name
    lr = cfg.lr
    weight_decay = cfg.weight_decay
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        msg = f"Optimizer {name} is wrong."
        log.critical(msg)
        raise Exception(msg)


def fit(model, cfg):
    """
    trains model
    """
    train_cfg = cfg.train_conf
    train_dataset, val_dataset = cfg2datasets(cfg)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=train_cfg.batch_size,
                              drop_last=True,
                              shuffle=True)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=train_cfg.batch_size)

    val_period = train_cfg.val_period
    epochs = train_cfg.epochs

    loss = cfg2loss(train_cfg.loss)
    optimizer = cfg2optimizer(model, train_cfg.optimizer)
    metrics = cfg2metric_list(train_cfg.metrics)

    for epoch in range(epochs):
        log.info(f"===== EPOCH: {epoch} =====")
        model.train()
        for img_batch, mask_batch in train_loader:
            log.info("===== TRAIN BATCH =====")
            optimizer.zero_grad()
            predicted_batch = model.inference(img_batch)
            loss_value = loss(predicted_batch, mask_batch)
            log.info(f"Loss: {loss_value.data}.")
            loss_value.backward()
            optimizer.step()

        if epoch % val_period == 0 or epoch == epochs-1:
            model.eval()
            with torch.no_grad():
                for img_batch, mask_batch in val_loader:
                    log.info("===== VALIDATION BATCH =====")
                    predicted_batch = model.inference(img_batch)
                    loss_value = loss(predicted_batch, mask_batch)
                    log.info(f"Loss: {loss_value.data}.")
                    for metric_name, metric in metrics:
                        log.info(f"Metric {metric_name}: {metric(predicted_batch, mask_batch).data}")

    return val_loader
