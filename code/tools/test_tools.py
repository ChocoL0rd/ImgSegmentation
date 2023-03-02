import logging
import torch
from torchvision.transforms import ToPILImage
import os

from omegaconf import OmegaConf, open_dict

from .metric_tools import *

__all__ = [
    "test"
]

log = logging.getLogger(__name__)

tensor2pil = ToPILImage()


def test(model, dataloader, path, cfg):
    os.mkdir(path)
    test_cfg = cfg.test_conf
    metrics = cfg2metric_list(test_cfg.metrics)  # a list [[metric_name, metric],...]
    num_metrics = len(metrics)
    results = [torch.tensor([]).to(cfg.device) for i in range(num_metrics)]

    img_num = 0
    model.eval()
    with torch.no_grad():
        for img_batch, mask_batch in dataloader:
            predicted = model.inference(img_batch)

            if test_cfg.illustrate:
                for i in range(img_batch.shape[0]):
                    predicted_mask = predicted[i]
                    img = img_batch[i]
                    pil_image = tensor2pil(predicted_mask*img)
                    pil_image.save(os.path.join(path, f"img{img_num}.jpeg"))
                    img_num += 1

            for i in range(num_metrics):
                res = metrics[i][1](predicted, mask_batch).reshape([-1])
                results[i] = torch.cat([results[i], res])

        for i in range(num_metrics):
            metric_name = metrics[i][0]
            res = float(results[i].mean())
            log.info(f"Metric {metric_name}: {res}")
            with open_dict(test_cfg):
                test_cfg.metrics[i].result = res

            OmegaConf.save(config=test_cfg.metrics, f=os.path.join(path, "results.yaml"))
