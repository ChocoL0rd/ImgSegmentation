import torch
import torch.optim
from torch.utils.data import DataLoader

import logging

from .dataset_tools import ImgMaskSet
from .optim_tools import CustomOptimizer, CustomScheduler
from .early_stopper import EarlyStopper
from .metric_tools import cfg2metric_dict
from .loss_tools import cfg2loss

log = logging.getLogger(__name__)


def fit(epochs: int, val_period: int,
        model, loss, metrics: dict,
        optimizer: CustomOptimizer, scheduler: CustomScheduler, early_stopper: EarlyStopper,
        train_loader: DataLoader, val_loader: DataLoader):

    for epoch in range(epochs):
        model.train()

        for img_names, img_batch, mask_batch in train_loader:
            log.info(f"===== TRAIN BATCH (Epoch: {epoch}) =====")
            optimizer.zero_grad()
            predicted_batch = model.inference(img_batch)
            loss_value = loss(predicted_batch, mask_batch)
            log.info(f"Loss: {loss_value.data}.")
            loss_value.backward()
            optimizer.step()

        # on first and last epochs validation happens too
        if epoch % val_period == 0 or epoch == epochs - 1:
            model.eval()
            loss_sum = 0
            n = 0
            metric_values = {}
            for metric_name in metrics:
                metric_values[metric_name] = torch.tensor([])
            with torch.no_grad():
                for img_names, img_batch, mask_batch in val_loader:
                    log.info(f"===== VALIDATION BATCH (Epoch: {epoch} =====")
                    predicted_batch = model.inference(img_batch)
                    loss_value = loss(predicted_batch, mask_batch)
                    log.info(f"Loss: {loss_value.data}.")
                    loss_sum += loss_value
                    n += 1
                    for metric_name, metric in metrics.items():
                        metric_value = metric(predicted_batch, mask_batch).cpu().data.reshape([-1])
                        log.info(f"Metric {metric_name}: {metric_value.tolist()}")
                        metric_values[metric_name] = torch.cat([metric_values[metric_name], metric_value])
                        # if metric is main then add to metric history

            early_stopper(metric_values, model)
            scheduler.step(loss_sum / n)

            if early_stopper.early_stop:
                break

    # saving best params in model
    model.load_state_dict(early_stopper.load_last_params())


def cfg2fit(cfg, model, train_dataset: ImgMaskSet, val_dataset: ImgMaskSet):
    """
    trains model
    """
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=cfg.batch_size,
                              drop_last=True,
                              shuffle=True)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=cfg.batch_size)

    val_period = cfg.val_period
    epochs = cfg.epochs

    loss = cfg2loss(cfg.loss)
    metrics = cfg2metric_dict(cfg.metrics)

    # check if target_metric is in metrics
    if cfg.early_stopper.target_metric not in metrics.keys():
        msg = f"Main metric {cfg.target_metric} not in metrics."
        log.critical(msg)
        raise Exception(msg)

    optimizer = CustomOptimizer(cfg.optimizer, model.get_params())
    scheduler = CustomScheduler(cfg.scheduler, optimizer.optimizer)
    early_stopper = EarlyStopper(cfg.early_stopper)

    fit(epochs=epochs, val_period=val_period,
        model=model, loss=loss, metrics=metrics,
        optimizer=optimizer, scheduler=scheduler, early_stopper=early_stopper,
        train_loader=train_loader, val_loader=val_loader)

    # save model here or in fit getting the best of early_stopper

