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
import json

import logging

from .metric_tools import *
from .augmentation_tools import *
from .loss_tools import *

__all__ = [
    "cfg2datasets",
    "cfg2optimizer",
    "cfg2fit",
    "ImgMaskSet",
    "cfg2sublists"
]

# Here implemented 'fit' that trains model.

log = logging.getLogger(__name__)
img2tensor = ToTensor()

#
# class ImgMaskSet(Dataset):
#     """
#     It's dataset that returns images and their masks (with the same file names).
#     """
#     def __init__(self, log_name: str, img_dir_path: str, mask_dir_path: str, img_list: Optional[List[str]],
#                  transforms,  # add type of augmentation
#                  device: torch.device,):
#         """
#         :param log_name:  name that is used in log
#         :param img_dir_path: path to directory where images are contained. in this directory all images are .jpeg
#         :param mask_dir_path: path to directory where masks (images with deleted background) are contained.
#         in this directory all images are .png
#         :param img_list: specifies image names in image directory that should be used
#         :param transforms: transformations to augment dataset
#         :param device: device of images and masks
#         """
#         self.log_name = log_name
#
#         self.local_log = logging.getLogger(__name__)
#         self.local_log.setLevel(log.getEffectiveLevel())
#         self.local_log.debug(f"Local log for {self.log_name} dataset is created.")
#
#         self.img_dir_path = img_dir_path
#         self.mask_dir_path = mask_dir_path
#         self.device = device
#
#         log.info(f"Creating {self.log_name} dataset on device {self.device}: \n"
#                      f"Path to image dir: {self.img_dir_path} \n"
#                      f"Path to mask dir: {self.mask_dir_path}")
#         self.apply_trfms = True
#         self.return_not_transformed = False
#
#         if img_list is None:
#             log.info(f"List of images isn't specified.\n"
#                      f"Get all images from the image directory: {img_dir_path}\n"
#                      f"and mask directory {mask_dir_path}")
#     #         suppose all files are in both directories
#             self.img_list = [i[:-5] for i in os.listdir(img_dir_path)]  # deleted extension .jpeg
#         else:
#             self.img_list = [i[:-5] for i in img_list]  # deleted extension .jpeg
#
#         self.trfms = transforms
#
#         self.size = len(self.img_list)
#
#         log.info(f"{self.log_name} dataset is created. Size: {self.size}")
#
#     def __len__(self):
#         return self.size
#
#     def __getitem__(self, idx):
#
#         img_name = os.path.basename(self.img_list[idx])
#         img_path = os.path.join(self.img_dir_path, self.img_list[idx] + ".jpeg")
#         mask_path = os.path.join(self.mask_dir_path, self.img_list[idx] + ".png")
#
#         self.local_log.debug(f"Dataset {self.log_name} returns item with idx:{idx}\n"
#                              f"img_path: {img_path}\n"
#                              f"mask_path: {mask_path}")
#
#         img = cv.imread(img_path)
#         img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # convert to RGB format
#
#         # mask = cv.imread(mask_path)
#         with Image.open(mask_path) as mask_im:
#             mask = np.array(mask_im.split()[-1])  # retrieve transparent mask
#
#         if self.trfms is None or self.apply_trfms is False:
#             img_tensor = img2tensor(img).to(torch.float32)
#             mask_tensor = img2tensor(mask).to(torch.float32)
#         else:
#             # apply transformations
#             transformed = self.trfms(image=img, mask=mask)
#             self.local_log.debug("Transform is applied.")
#
#             img_tensor = img2tensor(transformed["image"]).to(torch.float32)
#             mask_tensor = img2tensor(transformed["mask"]).to(torch.float32)
#
#         if self.return_not_transformed:
#             img = img2tensor(img)
#             return img_name, img_tensor.to(self.device), mask_tensor.to(self.device), img.to(self.device)
#         else:
#             return img_name, img_tensor.to(self.device), mask_tensor.to(self.device)
#
#     def get_img_list(self):
#         return [f"{i}.jpeg" for i in self.img_list]
#
#     def img_list2file(self, path):
#         with open(os.path.join(path, f"{self.log_name}_dataset.json"), "w") as fp:
#             json.dump(
#                 {
#                     "img_dir": self.img_dir_path,
#                     "mask_dir": self.mask_dir_path,
#                     "imgs_list":  self.get_img_list() # names saved with jpeg format
#                 },
#                 fp, indent=2)


class ImgMaskSet(Dataset):
    """
    It's dataset that returns images and their masks (with the same file names).
    """
    def __init__(self, log_name: str, img_dir_path: str, mask_dir_path: str, img_list: Optional[List[str]],
                 transforms, bgr_trfm,  # add type of augmentation
                 device: torch.device,):
        """
        :param log_name:  name that is used in log
        :param img_dir_path: path to directory where images are contained. in this directory all images are .jpeg
        :param mask_dir_path: path to directory where masks (images with deleted background) are contained.
        in this directory all images are .png
        :param img_list: specifies image names in image directory that should be used
        :param transforms: transformations to augment dataset
        :param bgr_trfm: transformation of background
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

        self.trfms = transforms
        self.bgr_trfm = bgr_trfm

        self.apply_trfms = True
        self.apply_bgr_trfms = True
        self.return_not_transformed = False

        self.size = len(self.img_list)

        log.info(f"{self.log_name} dataset is created. Size: {self.size}")

    def __len__(self):
        return self.size

    def __getitem__(self, idx):

        img_name = os.path.basename(self.img_list[idx])
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

        if self.trfms is None or self.apply_trfms is False:
            img_tensor = img2tensor(img).to(torch.float32)
            mask_tensor = img2tensor(mask).to(torch.float32)
        else:
            # new_img - image with changed background
            if self.bgr_trfm is None or self.apply_bgr_trfms is False:
                new_img = img
            else:
                trfmd_bgr = self.bgr_trfm(image=img, mask=mask)["image"]
                new_img = img * mask.reshape([330, 330, 1]) + trfmd_bgr * (1 - mask.reshape([330, 330, 1]))

            # apply transformations
            transformed = self.trfms(image=new_img, mask=mask)
            self.local_log.debug("Transform is applied.")

            img_tensor = img2tensor(transformed["image"]).to(torch.float32)
            mask_tensor = img2tensor(transformed["mask"]).to(torch.float32)

        if self.return_not_transformed:
            img = img2tensor(img)
            return img_name, img_tensor.to(self.device), mask_tensor.to(self.device), img.to(self.device)
        else:
            return img_name, img_tensor.to(self.device), mask_tensor.to(self.device)

    def get_img_list(self):
        return [f"{i}.jpeg" for i in self.img_list]

    def img_list2file(self, path):
        with open(os.path.join(path, f"{self.log_name}_dataset.json"), "w") as fp:
            json.dump(
                {
                    "img_dir": self.img_dir_path,
                    "mask_dir": self.mask_dir_path,
                    "imgs_list":  self.get_img_list()  # names saved with jpeg format
                },
                fp, indent=2)


def cfg2sublists(cfg, sublists, top):
    """
    :param cfg:
    :param sublists: dictionary
    :param top: dataframe
    :return:
    """
    df = top[[cfg.metric, "img_name"]]
    if cfg.name == "threshold":
        satisfying_imgs = df[df[cfg.metric][0] <= cfg.threshold]["img_name"][0].tolist()
        log.info(f"Number of satisfying images: {len(satisfying_imgs)}")
        sublists = {
            "train": sublists["train"] + satisfying_imgs,
            "val": [img for img in sublists["val"] if img not in satisfying_imgs]
        }
    else:
        msg = f"Name of sublist rule {cfg.name} is wrong"
        log.critical(msg)
        raise Exception(msg)

    return sublists


def cfg2datasets(cfg, data_sublists=None):
    """
    Convert config to train and val datasets.
    """
    log_level = log.getEffectiveLevel()
    local_cfg = cfg.train_conf.dataset
    local_debug = local_cfg.debug
    if local_debug:
        log.setLevel(10)
        log.info("While creating datasets, debug level is activated.")

    augment = cfg2augmentation(local_cfg.augmentation)
    bgr_augment = cfg2augmentation(local_cfg.bgr_augmentation)
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

        if data_sublists is None:
            train_list, val_list = train_test_split(os.listdir(img_path),
                                                    train_size=train_proportion,
                                                    shuffle=True)
        else:
            train_list, val_list = data_sublists["train"], data_sublists["val"]

        train_dataset = ImgMaskSet(log_name="train",
                                   img_dir_path=img_path, mask_dir_path=mask_path,
                                   img_list=train_list,
                                   transforms=augment, bgr_trfm=bgr_augment,
                                   device=device)
        # val should have a preproc transform
        val_dataset = ImgMaskSet(log_name="validation",
                                 img_dir_path=img_path, mask_dir_path=mask_path,
                                 img_list=val_list,
                                 transforms=None, bgr_trfm=bgr_augment,
                                 device=device)

    else:
        log.info(f"Get train and val data from different directories:\n"
                 f"train image directory name: {train_img_dir_name}\n"
                 f"train mask directory name: {train_mask_dir_name}\n"
                 f"val image directory name: {val_img_dir_name}\n"
                 f"val mask directory name: {val_mask_dir_name}")

        # here should be preproc transforms
        train_dataset = ImgMaskSet(log_name="train",
                                   img_dir_path=local_cfg.train_img_dir, mask_dir_path=local_cfg.train_mask_dir,
                                   img_list=None,
                                   transforms=augment, bgr_trfm=None,
                                   device=device)
        val_dataset = ImgMaskSet(log_name="validation",
                                 img_dir_path=local_cfg.val_img_dir, mask_dir_path=local_cfg.val_mask_dir,
                                 img_list=None,
                                 transforms=None,  bgr_trfm=None,
                                 device=device)

    if local_debug:
        log.setLevel(log_level)
        log.info(f"After creating datasets, log level changed to {log_level}.")

    return train_dataset, val_dataset


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


def fit(epochs, val_period,
        model, loss, metrics, optimizer,
        train_loader, val_loader):

    for epoch in range(epochs):
        log.info(f"===== EPOCH: {epoch} =====")
        model.train()

        for img_names, img_batch, mask_batch in train_loader:
            log.info("===== TRAIN BATCH =====")
            optimizer.zero_grad()
            predicted_batch = model.inference(img_batch)
            loss_value = loss(predicted_batch, mask_batch)
            log.info(f"Loss: {loss_value.data}.")
            loss_value.backward()
            optimizer.step()

        if epoch != 0 and (epoch % val_period == 0 or epoch == epochs-1):
            model.eval()
            with torch.no_grad():
                for img_names, img_batch, mask_batch in val_loader:
                    log.info("===== VALIDATION BATCH =====")
                    predicted_batch = model.inference(img_batch)
                    loss_value = loss(predicted_batch, mask_batch)
                    log.info(f"Loss: {loss_value.data}.")
                    for metric_name, metric in metrics:
                        log.info(f"Metric {metric_name}:\n"
                                 f"{metric(predicted_batch, mask_batch).data.reshape([-1]).tolist()}")


def cfg2fit(model, train_dataset: ImgMaskSet, val_dataset: ImgMaskSet, cfg):
    """
    trains model
    """
    train_cfg = cfg.train_conf
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

    fit(epochs=epochs, val_period=val_period,
        model=model, loss=loss, metrics=metrics, optimizer=optimizer,
        train_loader=train_loader, val_loader=val_loader)

