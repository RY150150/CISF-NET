"""CISF-NET: the final segmentation network architecture."""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.PMiT import mit_b3
from model.SIEM import SIEM
from model.BasicConv import BasicConv2d
from model.MSAM import MSAM
from model.Contrastive_head import ProjectionHead
from model.APConv import PConv

class CISFNet(nn.Module):
    """CISF-NET segmentation model."""

    def __init__(self, channel=32, pretrained_path=None):
        super().__init__()
        self.backbone = mit_b3()  # [64, 128, 320, 512]
        if pretrained_path is not None:
            if not os.path.isfile(pretrained_path):
                raise FileNotFoundError(
                    'Backbone checkpoint not found: {}'.format(pretrained_path)
                )
            save_model = torch.load(pretrained_path, map_location='cpu')
            if isinstance(save_model, dict) and 'state_dict' in save_model:
                save_model = save_model['state_dict']
            model_dict = self.backbone.state_dict()
            state_dict = {
                key.replace('module.', '', 1): value
                for key, value in save_model.items()
                if key.replace('module.', '', 1) in model_dict
            }
            model_dict.update(state_dict)
            self.backbone.load_state_dict(model_dict)


        self.Translayer1 = BasicConv2d(64, channel, 1)
        self.Translayer2 = BasicConv2d(128, channel, 1)
        self.Translayer3 = BasicConv2d(320, channel, 1)
        self.Translayer4 = BasicConv2d(512, channel, 1)
        self.contrasthead = ProjectionHead(dim=32)

        self.Smim = MSAM(out_channel=channel)
        self.cim = SIEM(out_channel=channel)

        self.uper =  nn.Sequential(nn.ConvTranspose2d(in_channels=channel, out_channels=channel, kernel_size=4, stride=2, padding=1),
                                   BasicConv2d(in_planes=channel, out_planes=channel,kernel_size=3,stride=1,padding=1))

        self.prediction = nn.Sequential(BasicConv2d(in_planes=channel,out_planes=channel,kernel_size=3, stride=1,padding=1),
                                        nn.Conv2d(channel, 1, 1))

        self.uper_x4 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)

    @property
    def msam(self):
        """Return the shared Multi-Scale Self-Attention module."""
        return self.Smim

    @property
    def siem(self):
        """Return the Segmented Information Fusion Enhancement module."""
        return self.cim

    def forward(self, x):
        backbone_features = self.backbone(x)

        stage1_feature = self.Translayer1(backbone_features[0])
        stage2_feature = self.Translayer2(backbone_features[1])
        stage3_feature = self.Translayer3(backbone_features[2])
        stage4_feature = self.Translayer4(backbone_features[3])
        stage4_shortcut = stage4_feature

        embedding = self.contrasthead(stage4_feature)

        stage4_feature = self.msam(stage4_feature)
        stage3_feature = self.msam(stage3_feature)
        stage2_feature = self.msam(stage2_feature)

        siem_output = self.siem(stage4_feature, stage3_feature, stage2_feature)
        fused_feature = (
            stage1_feature
            + self.uper(siem_output)
            + self.uper_x4(stage4_shortcut)
        )
        prediction_logits = self.prediction(fused_feature)
        prediction = F.interpolate(
            prediction_logits,
            scale_factor=4,
            mode='bilinear',
            align_corners=False,
        )
        return prediction, embedding


if __name__ == '__main__':
    model = CISFNet().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    prediction, embedding = model(input_tensor)

    print(prediction.size())
    print(embedding.size())
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print(num_params)
