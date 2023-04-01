import torch
from torch import nn
import os


class Seq(nn.Module):
    def __init__(self, cfg):
        super(Seq, self).__init__()
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
        for model in self.models[:-1]:
            x = model.forward(x)
        x = self.models[-1].inference(x)

        return x
