"""Dual-branch spatial-channel selection used within SIEM."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialChannelSelectionCore(nn.Module):
    """Compute competitive spatial and channel weights for two feature branches."""

    def __init__(self, dim, r=16, L=32):
        super().__init__()
        d = max(dim // r, L)
        self.dim = dim

        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)

        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(2, 2, 1)
        self.conv_proj = nn.Conv2d(dim // 2, dim, 1)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_maxpool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Conv2d(d, dim, 1, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def _align_like(self, tensor, target_channels, target_hw=None):
        """Align a feature tensor to the requested channels and spatial size."""
        if target_hw is not None and tensor.shape[-2:] != target_hw:
            tensor = F.interpolate(
                tensor, size=target_hw, mode='bilinear', align_corners=False
            )
        channels = tensor.shape[1]
        if channels != target_channels:
            if channels > target_channels:
                tensor = tensor[:, :target_channels, :, :]
            else:
                pad_ch = target_channels - channels
                pad = tensor.new_zeros((tensor.size(0), pad_ch, tensor.size(2), tensor.size(3)))
                tensor = torch.cat([tensor, pad], dim=1)
        return tensor

    def forward(self, main_feature, auxiliary_feature):
        main_feature = self._align_like(main_feature, self.dim)
        target_hw = main_feature.shape[-2:]
        auxiliary_feature = self._align_like(
            auxiliary_feature,
            self.dim,
            target_hw,
        )
        batch_size = main_feature.size(0)
        dim = self.dim

        projected_main = self.conv1(main_feature)
        projected_auxiliary = self.conv2(auxiliary_feature)

        joint_feature = torch.cat([projected_main, projected_auxiliary], dim=1)
        average_map = torch.mean(joint_feature, dim=1, keepdim=True)
        maximum_map, _ = torch.max(joint_feature, dim=1, keepdim=True)
        spatial_descriptor = torch.cat([average_map, maximum_map], dim=1)
        spatial_weights = self.conv(spatial_descriptor).sigmoid()

        channel_descriptor = self.global_pool(joint_feature)
        channel_embedding = self.fc1(channel_descriptor)
        channel_weights = self.fc2(channel_embedding)
        channel_weights = channel_weights.reshape(batch_size, 2, dim // 2, -1)
        channel_weights = self.softmax(channel_weights)

        main_channel_weight, auxiliary_channel_weight = channel_weights.chunk(2, dim=1)
        main_channel_weight = main_channel_weight.reshape(batch_size, dim // 2, 1, 1)
        auxiliary_channel_weight = auxiliary_channel_weight.reshape(batch_size, dim // 2, 1, 1)

        main_weight = main_channel_weight * spatial_weights[:, 0, :, :].unsqueeze(1)
        auxiliary_weight = auxiliary_channel_weight * spatial_weights[:, 1, :, :].unsqueeze(1)

        selected_feature = (
            projected_main * main_weight
            + projected_auxiliary * auxiliary_weight
        )
        modulation = self.conv_proj(selected_feature).sigmoid()

        return main_feature * modulation


class SpatialChannelSelection(nn.Module):
    """Apply SIEM spatial-channel modulation with a residual output path."""

    def __init__(self, dim):
        super().__init__()
        self.proj_1 = nn.Conv2d(dim, dim, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = SpatialChannelSelectionCore(dim)
        self.proj_2 = nn.Conv2d(dim, dim, 1)

    def forward(self, main_feature, auxiliary_feature):
        shortcut = main_feature
        main_feature = self.proj_1(main_feature)
        main_feature = self.activation(main_feature)
        main_feature = self.spatial_gating_unit(
            main_feature,
            auxiliary_feature,
        )
        main_feature = self.proj_2(main_feature)
        main_feature = main_feature + shortcut
        return main_feature
