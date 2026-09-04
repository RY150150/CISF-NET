"""Segmented Information Fusion Enhancement module."""

import torch.nn as nn
from model.BasicConv import BasicConv2d
from model.SpatialChannelSelection import SpatialChannelSelection


class SIEM(nn.Module):
    def __init__(self,out_channel=32):
        super(SIEM, self).__init__()

        self.uper =  nn.Sequential(nn.ConvTranspose2d(in_channels=out_channel, out_channels=out_channel, kernel_size=4, stride=2, padding=1),
                                   BasicConv2d(in_planes=out_channel, out_planes=out_channel,kernel_size=3,stride=1,padding=1))
        self.ksb = SpatialChannelSelection(out_channel)

    @property
    def spatial_channel_selection(self):
        """Return the dual-branch spatial-channel selection block."""
        return self.ksb

    def forward(self, deep_feature, middle_feature, shallow_feature):
        middle_excitation = self.uper(deep_feature) * middle_feature + middle_feature
        shallow_excitation = self.uper(middle_feature) * shallow_feature + shallow_feature
        fused_feature = self.spatial_channel_selection(
            shallow_excitation,
            middle_excitation,
        )
        return fused_feature
