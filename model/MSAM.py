"""Multi-Scale Self-Attention module."""

import torch
import torch.nn as nn
from model.BasicConv import BasicConv2d
from utils.utils import cus_sample
from model.LDConv import LDConv
import math


class MSAM(nn.Module):
    def __init__(self,out_channel = 32, n_ld = 6):
        super(MSAM, self).__init__()
        self.h2l_pool = nn.AvgPool2d((2, 2), stride=2)
        self.l2h_up = cus_sample

        self.h2l_0 = nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        self.h2h_0 = nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        self.bnl_0 = nn.BatchNorm2d(out_channel)
        self.bnh_0 = nn.BatchNorm2d(out_channel)

        self.h2h_1 = nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        self.h2l_1 = LDConv(out_channel, out_channel, num_param=n_ld, stride=1, bias=False)
        self.l2h_1 = LDConv(out_channel, out_channel, num_param=n_ld, stride=1, bias=False)
        self.l2l_1 = GroupedChannelQKVGating(out_channel)
        self.bnl_1 = nn.BatchNorm2d(out_channel)
        self.bnh_1 = nn.BatchNorm2d(out_channel)

        self.h2h_2 = nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        self.l2h_2 = LDConv(out_channel, out_channel, num_param=n_ld, stride=1, bias=False)
        self.bnh_2 = nn.BatchNorm2d(out_channel)

        self.relu = nn.ReLU(True)

    def forward(self, x):
        h, w = x.shape[2:]

        # first conv
        x_h = self.relu(self.bnh_0(self.h2h_0(x)))
        x_l = self.relu(self.bnl_0(self.h2l_0(self.h2l_pool(x))))

        # mid conv
        x_h2h = self.h2h_1(x_h)
        x_h2l = self.h2l_1(self.h2l_pool(x_h))
        x_l2l = self.l2l_1(x_l)
        x_l2h = self.l2h_1(self.l2h_up(x_l, size=(h, w)))
        x_h = self.relu(self.bnh_1(x_h2h + x_l2h))
        x_l = self.relu(self.bnl_1(x_l2l + x_h2l))

        # last conv
        x_h2h = self.h2h_2(x_h)
        x_l2h = self.l2h_2(self.l2h_up(x_l, size=(h, w)))
        x_h = self.relu(self.bnh_2(x_h2h + x_l2h))

        return x_h + x

class GroupedChannelQKVGating(nn.Module):
    """Grouped channel-wise QKV gating used by the low-resolution branch."""

    def __init__(self, channels: int):
        super().__init__()
        assert channels >= 3, "channels must be at least 3"
        c1 = channels // 3
        c2 = (channels - c1) // 2
        c3 = channels - c1 - c2
        self.splits = (c1, c2, c3)

        def make_branch(c):
            if c == 0:
                return nn.Identity(), nn.Identity(), nn.Identity(), 1.0
            q = nn.Conv2d(c, c, kernel_size=1, bias=False)
            k = nn.Conv2d(c, c, kernel_size=1, bias=False)
            v = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False)
            scale = 1.0 / math.sqrt(c)
            return q, k, v, scale

        self.q1, self.k1, self.v1, self.s1 = make_branch(c1)
        self.q2, self.k2, self.v2, self.s2 = make_branch(c2)
        self.q3, self.k3, self.v3, self.s3 = make_branch(c3)

    @staticmethod
    def _branch(x, q, k, v, scale):
        if isinstance(q, nn.Identity):
            return x
        qx = q(x)
        kx = k(x)
        vx = v(x)
        attn = torch.softmax((qx * kx) * scale, dim=1)
        return attn * vx

    def forward(self, x):
        x1, x2, x3 = torch.split(x, self.splits, dim=1)
        y1 = self._branch(x1, self.q1, self.k1, self.v1, self.s1)
        y2 = self._branch(x2, self.q2, self.k2, self.v2, self.s2)
        y3 = self._branch(x3, self.q3, self.k3, self.v3, self.s3)
        return torch.cat([y1, y2, y3], dim=1)


if __name__ == '__main__':
    model = MSAM().cuda()
    input_tensor = torch.randn(1, 32, 352, 352).cuda()

    prediction = model(input_tensor)
    print(prediction.size())
