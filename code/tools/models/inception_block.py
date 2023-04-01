import torch
from torch import nn

import os


class InceptionBlock(nn.Module):
    def __init__(self, cfg):
        super(InceptionBlock, self).__init__()
        models_cfg = cfg.models
        self.models = nn.ModuleList()
        from ..build_model import model_cfg2model

        for model_cfg in models_cfg:
            model = model_cfg2model(model_cfg)
            if model_cfg.pretrained_model:
                model.load_state_dict(torch.load(os.path.join(model_cfg.pretrained_path, "model.pt")))

            self.models.append(model)
            if model_cfg.full_freeze:
                for param in model.parameters():
                    param.requires_grad = False

        self.final_conv = nn.Conv2d(in_channels=cfg.final_channels, out_channels=1, kernel_size=1)

    def pre_forward(self, x):
        res_list = []
        for model in self.models:
            res_list.append(model.pre_forward(x))

        x = torch.cat(res_list, dim=1)
        return x

    def forward(self, x):
        x = self.pre_forward(x)
        x = self.final_conv(x)
        return x

    def inference(self, x):
        x = self.forward(x)
        return nn.Sigmoid()(x)




