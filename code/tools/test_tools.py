import logging
import torch
from torchvision.transforms import ToPILImage
import os
from torch.utils.data import DataLoader
import pandas as pd
from omegaconf import OmegaConf, open_dict

from .metric_tools import *
from .train_tools import ImgMaskSet

__all__ = [
    "test"
]

log = logging.getLogger(__name__)

tensor2pil = ToPILImage()


# def test(model, dataloader, path, cfg):
#     os.mkdir(path)
#     test_cfg = cfg.test_conf
#     metrics = cfg2metric_list(test_cfg.metrics)  # a list [[metric_name, metric],...]
#     num_metrics = len(metrics)
#     results = [torch.tensor([]).to(cfg.device) for i in range(num_metrics)]
#
#     img_num = 0
#     model.eval()
#     with torch.no_grad():
#         for img_batch, mask_batch in dataloader:
#             predicted = model.inference(img_batch)
#
#             if test_cfg.illustrate:
#                 for i in range(img_batch.shape[0]):
#                     predicted_mask = predicted[i]
#                     img = img_batch[i]
#                     pil_image = tensor2pil(predicted_mask*img)
#                     pil_image.save(os.path.join(path, f"img{img_num}.jpeg"))
#                     img_num += 1
#
#             for i in range(num_metrics):
#                 res = metrics[i][1](predicted, mask_batch).reshape([-1])
#                 results[i] = torch.cat([results[i], res])
#
#         for i in range(num_metrics):
#             metric_name = metrics[i][0]
#             res = float(results[i].mean())
#             log.info(f"Metric {metric_name}: {res}")
#             with open_dict(test_cfg):
#                 test_cfg.metrics[i].result = res
#
#             OmegaConf.save(config=test_cfg.metrics, f=os.path.join(path, "results.yaml"))


# def test(model, dataset: ImgMaskSet, path, cfg):
#     log.info("===== TEST =====")
#     test_cfg = cfg.test_conf
#     data_save_path = os.path.join(path, dataset.log_name)
#     os.mkdir(data_save_path)
#
#     batch_size = test_cfg.batch_size
#     dataset.apply_trfms = False
#     dataloader = DataLoader(dataset=dataset,
#                             batch_size=batch_size,
#                             drop_last=False,
#                             shuffle=False)
#
#     metrics = cfg2metric_list(test_cfg.metrics)
#     num_metrics = len(metrics)
#     results = [torch.tensor([]).to(cfg.device) for i in range(num_metrics)]
#
#     thresholds = {}
#
#     for threshold_value in test_cfg.thresholds:
#         if threshold_value == 0:
#             def set_threshold(x):
#                 return x
#         elif type(threshold_value) in [int, float] and 0 < threshold_value < 1:
#             def set_threshold(x):
#                 new_x = x.clone()
#                 new_x[new_x > threshold_value] = 1
#                 new_x[new_x <= threshold_value] = 0
#                 return new_x
#         else:
#             msg = f"Threshold {threshold_value} is wrong. " \
#                   f"Threshold is a number between 0 and 1 or 0 if no threshold."
#             log.critical(msg)
#             raise Exception(msg)
#         thresholds[threshold_value] = set_threshold
#         os.mkdir(os.path.join(data_save_path, f"thr_{threshold_value}"))
#
#     img_num = 0
#     model.eval()
#     with torch.no_grad():
#         for img_batch, mask_batch in dataloader:
#             predicted_batch = model.inference(img_batch)
#
#             for i in range(img_batch.shape[0]):
#                 for threshold_value in thresholds:
#                     set_threshold = thresholds[threshold_value]
#                     thresholded_batch = set_threshold(predicted_batch)
#
#                     predicted_mask = thresholded_batch[i]
#                     img = img_batch[i]
#                     pil_image = tensor2pil(predicted_mask * img)
#                     pil_image.save(os.path.join(data_save_path, f"thr_{threshold_value}/img{img_num}.jpeg"))
#
#                 img_num += 1
#
#             for i in range(num_metrics):
#                 res = metrics[i][1](predicted_batch, mask_batch).reshape([-1])
#                 results[i] = torch.cat([results[i], res])
#
#         for i in range(num_metrics):
#             metric_name = metrics[i][0]
#             res = float(results[i].mean())
#             log.info(f"Metric {metric_name}: {res}")
#             with open_dict(test_cfg):
#                 test_cfg.metrics[i].result = res
#
#             OmegaConf.save(config=test_cfg.metrics, f=os.path.join(data_save_path, "results.yaml"))


def test(model, dataset: ImgMaskSet, path, cfg):
    log.info(f"===== TEST ({dataset.log_name} dataset) =====")
    test_cfg = cfg.test_conf
    data_save_path = os.path.join(path, dataset.log_name)
    os.mkdir(data_save_path)

    batch_size = test_cfg.batch_size
    dataset.apply_trfms = False
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

    img_num = 0
    model.eval()
    with torch.no_grad():
        for img_batch, mask_batch in dataloader:
            predicted_batch = model.inference(img_batch)

            for i in range(img_batch.shape[0]):
                for threshold_value in thresholds:
                    # set threshold for batch and save contained images in corresponding directory
                    set_threshold = thresholds[threshold_value]
                    thresholded_batch = set_threshold(predicted_batch)
                    predicted_mask = thresholded_batch[i]
                    img = img_batch[i]
                    pil_image = tensor2pil(predicted_mask * img)
                    pil_image.save(os.path.join(data_save_path, f"thr_{threshold_value}/img{img_num}.jpeg"))

                img_num += 1

            for i in range(num_stats):
                thresholded_batch = stat_list[i]["threshold_func"](predicted_batch)
                res = stat_list[i]["metric_func"](thresholded_batch, mask_batch).reshape([-1])
                stat_list[i]["results"] = torch.cat([stat_list[i]["results"], res])

        res_df = pd.DataFrame()
        for i in range(num_stats):
            res = float(stat_list[i]["stat_func"](stat_list[i]["results"]))
            log.info(f"Metric {stat_list[i]['metric']}; Threshold {stat_list[i]['threshold']}; Stat {stat_list[i]['stat']}: {res}")

            res_df = pd.concat([res_df, pd.DataFrame({
                "metric": [stat_list[i]["metric"]],
                "threshold": [stat_list[i]["threshold"]],
                "stat": [stat_list[i]["stat"]],
                "result": [res]
            })])

        res_df.to_excel(os.path.join(data_save_path, "results.xlsx"), index=False)

