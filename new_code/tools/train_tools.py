import torch
import torch.optim
from torch.utils.data import DataLoader

import logging

from dataset_tools import *


log = logging.getLogger(__name__)


def fit(epochs, val_period, checkpoint_period,
        model, loss,
        metrics, optimizer,
        train_loader, val_loader):

    for epoch in range(epochs):
        model.train()

        for img_names, img_batch, mask_batch in train_loader:
            log.info(f"===== TRAIN BATCH (Epoch: {epoch} =====")
            optimizer.zero_grad()
            predicted_batch = model.inference(img_batch)
            loss_value = loss(predicted_batch, mask_batch)
            log.info(f"Loss: {loss_value.data}.")
            loss_value.backward()
            optimizer.step()

        if epoch != 0 and epoch % val_period == 0:
            model.eval()
            with torch.no_grad():
                for img_names, img_batch, mask_batch in val_loader:
                    log.info(f"===== VALIDATION BATCH (Epoch: {epoch} =====")
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

