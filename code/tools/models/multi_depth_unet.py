import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Sequential):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, drop_value=0):
        super(DoubleConv, self).__init__()
        if drop_value > 0:
            self.dropout1 = nn.Dropout2d(drop_value)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        if drop_value > 0:
            self.dropout2 = nn.Dropout2d(drop_value)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)


class Down(nn.Sequential):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, drop_value=0):
        super(Down, self).__init__()
        self.max_pool = nn.MaxPool2d(2)
        self.double_conv = DoubleConv(in_channels, out_channels, drop_value)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_chnls, out_chnls, skip_chnls, mode, drop_value=0):
        super(Up, self).__init__()
        self.double_conv = DoubleConv(in_chnls + skip_chnls, out_chnls, drop_value)
        self.mode = mode

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode=self.mode, align_corners=True)
        x = torch.cat([x, skip], dim=1)
        x = self.double_conv(x)
        return x


class MultiDepthUnet(nn.Module):
    def __init__(self, cfg):
        super(MultiDepthUnet, self).__init__()
        if len(cfg.down_ch) != len(cfg.up_ch):
            raise Exception(f"Size of down_ch {len(cfg.down_ch)} and up_ch {len(cfg.up_ch)} have to be the same.")

        down_chnls = [cfg.first_ch] + cfg.down_ch + [cfg.middle_ch]
        up_chnls = [cfg.middle_ch] + cfg.up_ch + [cfg.last_ch]
        skip_chnls = cfg.down_ch[::-1] + [cfg.first_ch]

        self.in_double_conv = DoubleConv(cfg.in_ch, cfg.first_ch, cfg.in_drop_value)
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        depth = len(down_chnls) - 1

        for i in range(depth):
            self.down_blocks.append(
                Down(down_chnls[i], down_chnls[i+1], cfg.down_drop_values[i])
            )
            self.up_blocks.append(
                Up(up_chnls[i], up_chnls[i+1], skip_chnls[i], cfg.mode[i], cfg.up_drop_values[i])
            )

        self.final_conv = nn.Conv2d(in_channels=cfg.last_ch, out_channels=1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        """
        :param x: original image
        :return: middle feature map
        """
        skip_list = []  # contains all features except middle and original image
        x = self.in_double_conv(x)
        for down_block in self.down_blocks:
            skip_list.append(x)
            x = down_block(x)
        return x, skip_list[::-1]

    def decode(self, x, skip_list):
        """
        :param skip_list: list of skips before middle feature map
        :param x:  middle feature map
        :return: last feature map in encoder (middle), skip feature maps and their sizes
        """
        for i in range(len(self.up_blocks)):
            x = self.up_blocks[i](x, skip_list[i])
        return x

    def pre_forward(self, x, skip_list=None, computed=False):
        """
        :param x: depends on predicted
        :param skip_list: needed if predicted is True
        :param computed: if False then x is original image,
        else it's implied all before is computed
        """
        if not computed:
            x, skip_list = self.encode(x)
        x = self.decode(x, skip_list)
        return x

    def forward(self, x, computed=False):
        """
        :param x: depends on predicted
        :param computed: if False then x is original image,
        else it's implied all before is computed
        """
        if not computed:
            x = self.pre_forward(x)
        x = self.final_conv(x)
        return x

    def inference(self, x, computed=False):
        """
        :param x: depends on predicted
        :param computed: if False then x is original image,
        else it's implied all before is computed
        """
        if not computed:
            x = self.forward(x)
        x = self.sigmoid(x)
        return x
