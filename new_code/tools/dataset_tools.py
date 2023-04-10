from torch.utils.data import Dataset
from torchvision.transforms.transforms import ToTensor
import torch

import albumentations as A

import numpy as np

from PIL import Image
import cv2 as cv

from typing import Dict, Optional, Union, List
import os
import json

import logging

from .aug_tools import *

__all__ = [
    "cfg2datasets",
    "ImgMaskSet",
    "datasets2json_file"
]

# Here implemented 'fit' that trains model.

log = logging.getLogger(__name__)
img2tensor = ToTensor()


class ImgMaskSet(Dataset):
    """
    It's dataset that returns images, their masks and names. As well, it can return not transformed img.
    In img and mask dirs imgs and corresponding masks should be named the same.
    fgr and bgr trfms have not to change mask (it's can not be a flip, for instance)
    """
    def __init__(self, log_name: str, img_dir_path: str, mask_dir_path: str, img_list: List[str],
                 bgr_trfm, fgr_trfm, trfm, preproc,  # add type of augmentation
                 device: torch.device,):
        """
        :param log_name:  name that is used in log

        :param img_dir_path: path to directory where images are contained. in this directory all images are .jpeg
        :param mask_dir_path: path to directory where masks (images with deleted background) are contained.
        in this directory all images are .png
        :param img_list: specifies image names in image directory that should be used

        :param bgr_trfm: transformation of background, is not used during the test
        :param fgr_trfm: foreground augmentations, is not used during the test
        :param trfm: transformations to augment dataset, is not used during the test
        :param preproc: transformations that used during the test, it is applied after all other transformations

        :param device: device of images and masks
        """

        self.log_name = log_name

        self.img_dir_path = img_dir_path
        self.mask_dir_path = mask_dir_path
        self.device = device

        self.img_list = [i[:-5] for i in img_list]  # deleted extension .jpeg
        self.size = len(self.img_list)

        # augmentation
        self.bgr_trfm = bgr_trfm
        self.fgr_trfm = fgr_trfm
        self.trfm = trfm
        self.aug_flag = True  # applying of augmentation depends on it

        # preprocessing
        self.preproc = preproc
        self.preproc_flag = True  # applying of preprocessing depends on it

        # needed to return img in it original form, without preproc and augmentation
        self.return_original_img = False

        log.info(f"Created {self.log_name} dataset: \n"
                 f"Size: {self.size} \n"
                 f"Device: {self.device} \n"
                 f"Path to image dir: {self.img_dir_path} \n"
                 f"Path to mask dir: {self.mask_dir_path}")

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int):

        img_name = self.img_list[idx]
        img_path = os.path.join(self.img_dir_path, img_name + ".jpeg")
        mask_path = os.path.join(self.mask_dir_path, img_name + ".png")

        # read img and mask
        original_img = cv.imread(img_path)
        if original_img is None:
            msg = f"Wrong reading image {img_path}"
            log.critical(msg)
            raise Exception(msg)

        original_img = cv.cvtColor(original_img, cv.COLOR_BGR2RGB)  # convert to RGB format
        img = original_img.copy()

        with Image.open(mask_path) as mask_im:
            mask = np.array(mask_im.split()[-1])  # retrieve transparent mask

        # apply transformations
        if self.aug_flag:
            img, mask = self.apply_aug(img, mask)
        if self.preproc_flag:
            img, mask = self.apply_preproc(img, mask)

        # convert to tensor and transfer to device
        img_tensor = img2tensor(img).to(torch.float32).to(self.device)
        mask_tensor = img2tensor(mask).to(torch.float32).to(self.device)

        # return original image if it's needed
        if self.return_original_img:
            original_img = img2tensor(original_img).to(self.device)
            return img_name, img_tensor, mask_tensor, original_img
        else:
            return img_name, img_tensor, mask_tensor

    def apply_aug(self, img, mask):
        trfmd_bgr = self.bgr_trfm(image=img, mask=mask)["image"]
        trfmd_fgr = self.fgr_trfm(image=img, mask=mask)["image"]
        mask = mask.reshape([330, 330, 1])
        img = (trfmd_fgr * mask + trfmd_bgr * (1 - mask)).astype("uint8")

        augmented = self.trfm(image=img, mask=mask)
        return augmented["image"], augmented["mask"]

    def apply_preproc(self, img, mask):
        preprocessed = self.preproc(image=img, mask=mask)
        return preprocessed["image"], preprocessed["mask"]

    def get_img_list(self):
        return [f"{i}.jpeg" for i in self.img_list]

    def img_list2file(self, path: str):
        with open(os.path.join(path, f"{self.log_name}_dataset.json"), "w") as fp:
            json.dump(
                {
                    "img_dir": self.img_dir_path,
                    "mask_dir": self.mask_dir_path,
                    "imgs_list":  self.get_img_list()  # names saved with jpeg format
                },
                fp, indent=2)


def cfg2filter(cfg, ds_lists):
    """
    Changes lists of datasets according to the filter
    :param cfg: consists of name, and private settings for each filter
    :param ds_lists: (datasets_lists) dictionary where key is the name
    of dataset and value is a list of imgs
    :return: changed ds_lists
    """
    if cfg.name == "pass":
        pass
    else:
        msg = f"Wrong filter \"{cfg.name}\""
        log.critical(msg)
        raise Exception(msg)

    return ds_lists


def cfg2datasets(cfg):
    """
    :param cfg: dataset_cfg from main config
    consist of:
        1) device - where to contain returned images
        2) path - path to the file to read to get list of images for each dataset
        and path to the folder where it contains.
        3) filter - manipulations with datasets to get new
        4) bgr_trfm, fgr_trfm, trfm, preproc

    :return: dictionary: {dataset_name1: dataset1, ...}
    """
    with open(cfg.path, "r") as f:
        file_dict = json.load(f)

    img_path = file_dict["img_path"]
    mask_path = file_dict["mask_path"]
    ds_lists = cfg2filter(cfg.filter, file_dict["dataset_lists"])

    # converting configs to transformations
    bgr_trfm = cfg2trfm(cfg.bgr_trfm)
    fgr_trfm = cfg2trfm(cfg.fgr_trfm)
    trfm = cfg2trfm(cfg.trfm)
    preproc = cfg2trfm(cfg.preproc)

    # creating
    datasets = {}
    for ds_name in ds_lists:
        if ds_name == "validation":
            datasets[ds_name] = ImgMaskSet(
                log_name=ds_name,
                img_dir_path=img_path, mask_dir_path=mask_path,
                img_list=ds_lists[ds_name],
                bgr_trfm=A.Compose([]), fgr_trfm=A.Compose([]), trfm=A.Compose([]), preproc=preproc,
                device=torch.device(cfg.device)
            )
        else:
            datasets[ds_name] = ImgMaskSet(
                log_name=ds_name,
                img_dir_path=img_path, mask_dir_path=mask_path,
                img_list=ds_lists[ds_name],
                bgr_trfm=bgr_trfm, fgr_trfm=fgr_trfm, trfm=trfm, preproc=preproc,
                device=torch.device(cfg.device)
            )

    return datasets


def datasets2json_file(datasets: Dict[str, ImgMaskSet], save_path: str):
    key = [i for i in datasets.keys()][0]
    ds_dict = {
        "img_path": datasets[key].img_dir_path,
        "mask_path": datasets[key].mask_dir_path
    }

    for dataset_name, dataset in datasets.items():
        ds_dict[dataset_name] = dataset.get_img_list()

    with open(os.path.join(save_path, "datasets.json"), "w") as fp:
        json.dump(ds_dict, fp, indent=4)

