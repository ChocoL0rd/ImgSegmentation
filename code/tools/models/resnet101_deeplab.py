import torch
from torch import nn


class Resnet101DeepLab(nn.Module):
    def __init__(self, cfg):
        super(Resnet101DeepLab, self).__init__()
        self.deeplab = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101',
                                 weights="DeepLabV3_ResNet101_Weights.DEFAULT")

    def forward(self, x):
        return 1 - self.deeplab.forward(x)["out"][:, [0]]

    def inference(self, x):
        x = self.forward(x)
        return nn.Sigmoid()(x)

