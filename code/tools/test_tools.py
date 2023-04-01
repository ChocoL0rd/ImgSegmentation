import logging
import torch
from torchvision.transforms import ToPILImage
import os
from torch.utils.data import DataLoader
import pandas as pd
from omegaconf import OmegaConf, open_dict
import numpy as np

from .metric_tools import *
from .train_tools import ImgMaskSet
from .augmentation_tools import cfg2augmentation


__all__ = [
    "test"
]

log = logging.getLogger(__name__)

tensor2pil = ToPILImage()


def test(model, dataset: ImgMaskSet, path, cfg):
    log.info(f"===== TEST ({dataset.log_name} dataset) =====")
    test_cfg = cfg.test_conf
    data_save_path = os.path.join(path, dataset.log_name)
    os.mkdir(data_save_path)

    batch_size = test_cfg.batch_size

    preproc = cfg2augmentation(test_cfg.preproc)
    # dataset.apply_trfms = False
    dataset.trfms = preproc
    dataset.bgr_trfm = None
    dataset.return_not_transformed = True

    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            drop_last=False,
                            shuffle=False)

    stat_list = cfg2stat_list(test_cfg.metrics)
    num_stats = len(stat_list)
    for i in range(num_stats):
        stat_list[i]["results"] = torch.tensor([]).to(cfg.device)

    # creating thresholds to illustrate results
    thresholds = {}
    for threshold_value in test_cfg.thresholds:
        set_threshold = num2threshold(threshold_value)
        thresholds[threshold_value] = set_threshold
        os.mkdir(os.path.join(data_save_path, f"thr_{threshold_value}"))

    model.eval()
    with torch.no_grad():
        for img_names, preprocessed_img_batch, mask_batch, img_batch in dataloader:
            predicted_batch = model.inference(preprocessed_img_batch)

            for i in range(img_batch.shape[0]):
                for threshold_value in thresholds:
                    # set threshold for batch and save contained images in corresponding directory
                    set_threshold = thresholds[threshold_value]
                    thresholded_batch = set_threshold(predicted_batch)
                    predicted_mask = thresholded_batch[i]
                    img = img_batch[i]
                    pil_image = tensor2pil(predicted_mask * img)
                    pil_image.save(os.path.join(data_save_path, f"thr_{threshold_value}/{img_names[i]}.jpeg"))

            for i in range(num_stats):
                thresholded_batch = stat_list[i]["threshold_func"](predicted_batch)
                res = stat_list[i]["metric_func"](thresholded_batch, mask_batch).reshape([-1])
                stat_list[i]["results"] = torch.cat([stat_list[i]["results"], res])

        multi_columns = pd.MultiIndex.from_tuples(
            [(stat_list[i]["metric"], stat_list[i]["threshold"]) for i in range(num_stats)] + [["img_name"]]
        )
        full_res_ndarray = np.array([stat_list[i]["results"].cpu().numpy() for i in range(num_stats)] + [dataset.get_img_list()])
        full_res_df = pd.DataFrame(
            full_res_ndarray.T,
            columns=multi_columns,
        )
        full_res_df.to_excel(os.path.join(data_save_path, "full_results.xlsx"))

        res_df = pd.DataFrame()
        for i in range(num_stats):
            res = float(stat_list[i]["stat_func"](stat_list[i]["results"]))
            log.info(f"Metric {stat_list[i]['metric']}; Threshold {stat_list[i]['threshold']}; Stat {stat_list[i]['stat']}: {res}")

            res_df = pd.concat([res_df, pd.DataFrame({
                "stat": [stat_list[i]["stat"]],
                "metric": [stat_list[i]["metric"]],
                "threshold": [stat_list[i]["threshold"]],
                "result": [res]
            })])

        res_df.to_excel(os.path.join(data_save_path, "results.xlsx"), index=False)

        top_df = pd.DataFrame()
        top_df = pd.concat([top_df, res_df[res_df.threshold == 0]])
        top_df.insert(4, "top", "no_thr")

        res_df = res_df[res_df.threshold != 0]

        for metric in res_df.metric.unique():
            metric_df = res_df[res_df.metric == metric]
            for stat in res_df.stat.unique():
                for threshold in metric_df[metric_df.stat == stat].query("result==result.min()").threshold.unique():
                    tmp = metric_df[metric_df.threshold == threshold]
                    tmp.insert(4, 'top', stat)
                    top_df = pd.concat([top_df, tmp])

        top_df.sort_values(["stat", "metric"])
        top_df.to_excel(os.path.join(data_save_path, "top_results.xlsx"), index=False)

