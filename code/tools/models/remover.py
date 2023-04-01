import torch
from torch import nn


class Remover(nn.Module):
    def __init__(self, cfg):
        super(Remover, self).__init__()
        self.conv = nn.Conv2d(1, 1, 1)
        self.d = cfg.device

    def inference(self, x):
        angle_pixels = x[:, :, 0, 0]
        fulled = torch.zeros(x.shape)
        for x in range(fulled.shape[-1]):
            for y in range(fulled.shape[-2]):
                fulled[:, :, y, x] = angle_pixels

        fulled = fulled.to(self.d)
        return (x != fulled).to(torch.float32)
