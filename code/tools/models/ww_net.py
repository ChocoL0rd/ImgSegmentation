import torch
from torch import nn

import os

# WW - look like a sequence of tops and bottoms.


class WWNet(nn.Module):
    """
    Concatenates models that should contain pre_forward function.
    """
    def __init__(self, cfg):
        super(WWNet, self).__init__()
        models_cfg = cfg.models
        self.models = nn.ModuleList()
        from ..build_model import model_cfg2model

        for model_cfg in models_cfg:
            model = model_cfg2model(model_cfg)
            if model_cfg.pretrained_model:
                model.load_state_dict(torch.load(os.path.join(model_cfg.pretrained_path, "model.pt")))

            if model_cfg.full_freeze:
                for param in model.parameters():
                    param.requires_grad = False
            self.models.append(model)

    def forward(self, x):
        accumulator = x
        for model in self.models:
            x = model.pre_forward(accumulator)
            accumulator = torch.cat([accumulator, x], dim=1)

        x = self.models[-1].forward(x, computed=True)
        return x

    def inference(self, x):
        x = self.forward(x)
        x = self.models[-1].inference(x, computed=True)
        return x

