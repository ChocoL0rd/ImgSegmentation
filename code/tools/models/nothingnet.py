import torch
from torch import nn


class NothingNet(nn.Module):
    def __init__(self, cfg):
        super(NothingNet, self).__init__()
        self.conv = nn.Conv2d(1, 1, 1)
        self.res_device = torch.device(cfg.device)

    def inference(self, x):
        return torch.ones(x.shape).to(self.res_device)
