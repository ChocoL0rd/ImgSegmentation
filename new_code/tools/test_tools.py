import torch
from torchvision.transforms import ToPILImage
from torch.utils.data import DataLoader
import pandas as pd

import os
import logging

from .metric_tools import cfg2metric_dict
from .train_tools import ImgMaskSet

__all__ = [
    "cfg2test"
]


log = logging.getLogger(__name__)

tensor2pil = ToPILImage()


def cfg2test(cfg, model, dataset: ImgMaskSet):
    dataset.aug_flag = False
    dataset.return_original_img = True

    data_save_path = os.path.join(cfg.save_path, dataset.log_name)
    img_save_path = os.path.join(data_save_path, "seg_imgs")
    os.mkdir(data_save_path)
    os.mkdir(img_save_path)

    dataloader = DataLoader(dataset=dataset,
                            batch_size=cfg.batch_size,
                            drop_last=False,
                            shuffle=False)

    metrics = cfg2metric_dict(cfg.metrics)
    metric_history = {}
    for metric_name in metrics:
        metric_history[metric_name] = []

    model.eval()
    with torch.no_grad():
        for img_names, img_batch, mask_batch, orig_img_batch in dataloader:
            predicted_batch = model.inference(img_batch)
            masked_orig_img_batch = predicted_batch * orig_img_batch

            # saving segmented imgs
            for i in range(masked_orig_img_batch.shape[0]):
                pil_image = tensor2pil(masked_orig_img_batch[i])
                pil_image.save(os.path.join(img_save_path, f"{img_names[i]}.jpeg"))

            # saving metric history
            for metric_name, metric in metrics.items():
                metric_values = metrics[metric_name](predicted_batch, mask_batch).cpu().reshape([-1]).tolist()
                metric_history[metric_name] = metric_history[metric_name] + metric_values

    # saving all metric results for each img
    results = metric_history
    results["img"] = dataset.get_img_list()
    res_df = pd.DataFrame(results)
    res_df.to_excel(os.path.join(data_save_path, "full_results.xlsx"), index=False)
    res_df.describe().to_excel(os.path.join(data_save_path, "descr_results.xlsx"), index=False)

