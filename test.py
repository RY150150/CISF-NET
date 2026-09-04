"""Testing and prediction-map generation for CISF-NET."""

import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from model.CISF_NET import CISFNet
from utils.dataloader import test_dataset


DEFAULT_DATASETS = [
    'CVC-300',
    'CVC-ClinicDB',
    'Kvasir',
    'CVC-ColonDB',
    'ETIS-LaribPolypDB',
]


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    checkpoint = {
        key.replace('module.', '', 1): value
        for key, value in checkpoint.items()
    }
    model.load_state_dict(checkpoint, strict=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pth_path', type=str, required=True,
                        help='path to the trained CISF-NET checkpoint')
    parser.add_argument('--test_path', type=str, required=True,
                        help='root containing the test dataset folders')
    parser.add_argument('--save_path', type=str,
                        default='./result_map/CISF-NET/',
                        help='directory used to save prediction maps')
    parser.add_argument('--testsize', type=int, default=352,
                        help='testing image size')
    parser.add_argument('--datasets', nargs='+', default=DEFAULT_DATASETS,
                        help='dataset folder names below test_path')
    parser.add_argument('--device', type=str, default=None,
                        help='device such as cuda, cuda:0, or cpu')
    opt = parser.parse_args()

    device_name = opt.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device_name)

    # A full trained checkpoint contains all backbone parameters, so the
    # separate MiT-B3 pretrained checkpoint is not needed for testing.
    model = CISFNet(pretrained_path=None).to(device)
    load_checkpoint(model, opt.pth_path, device)
    model.eval()

    for data_name in opt.datasets:
        data_path = os.path.join(opt.test_path, data_name)
        image_root = os.path.join(data_path, 'images')
        gt_root = os.path.join(data_path, 'masks')
        if not os.path.isdir(image_root) or not os.path.isdir(gt_root):
            raise FileNotFoundError(
                'Expected images/ and masks/ under {}'.format(data_path)
            )

        save_path = os.path.join(opt.save_path, data_name)
        os.makedirs(save_path, exist_ok=True)

        test_loader = test_dataset(
            image_root + os.sep,
            gt_root + os.sep,
            opt.testsize,
        )

        with torch.no_grad():
            for _ in range(len(test_loader.images)):
                image, gt, name = test_loader.load_data()
                gt = np.asarray(gt, np.float32)
                image = image.to(device)

                prediction, _ = model(image)
                prediction = F.interpolate(
                    prediction,
                    size=gt.shape,
                    mode='bilinear',
                    align_corners=False,
                )
                prediction = torch.sigmoid(prediction).cpu().numpy().squeeze()
                prediction = (
                    (prediction - prediction.min())
                    / (prediction.max() - prediction.min() + 1e-8)
                )
                cv2.imwrite(
                    os.path.join(save_path, name),
                    (prediction * 255).astype(np.uint8),
                )

        print('CISF-NET: {} finished.'.format(data_name))


if __name__ == '__main__':
    main()
