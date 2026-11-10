# -*- coding: utf-8 -*-
"""Training entry point for CISF-NET."""

import argparse
import logging
import os
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from model.CISF_NET import CISFNet
from utils.dataloader import get_loader
from utils.loss import structure_loss
from utils.pixel_contrast_cross_entropy_loss import ContrastCELoss
from utils.utils import AvgMeter, adjust_lr, clip_gradient


def train(train_loader, model, optimizer, epoch, total_step, opt):
    model.train()
    size_rates = [0.75, 1, 1.25]
    loss_record = AvgMeter()
    contrast_loss = ContrastCELoss()

    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()

            # ---- data prepare ----
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()

            # ---- rescale ----
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.interpolate(
                    images,
                    size=(trainsize, trainsize),
                    mode='bilinear',
                    align_corners=True,
                )
                gts = F.interpolate(
                    gts,
                    size=(trainsize, trainsize),
                    mode='bilinear',
                    align_corners=True,
                )

            # ---- forward ----
            prediction, embedding = model(images)
            loss_contrast = contrast_loss(prediction, embedding, gts)
            loss_prediction = structure_loss(prediction, gts)
            loss = loss_prediction + 0.1 * loss_contrast

            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            if rate == 1:
                loss_record.update(loss.data, images.size(0))

        if i % 20 == 0 or i == total_step:
            print(
                '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], loss: {:0.4f}'.format(
                    datetime.now(),
                    epoch,
                    opt.epoch,
                    i,
                    total_step,
                    loss_record.show(),
                )
            )

    return loss_record.show()


if __name__ == '__main__':
    model_name = 'CISF-NET'
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int, default=100, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--optimizer', type=str, default='AdamW',
                        help='choosing optimizer AdamW or SGD')
    parser.add_argument('--augmentation', default=False,
                        help='choose to do random flip rotation')
    parser.add_argument('--batchsize', type=int, default=16,
                        help='training batch size')
    parser.add_argument('--trainsize', type=int, default=352,
                        help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5,
                        help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1,
                        help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50,
                        help='every n epochs decay learning rate')
    parser.add_argument('--train_path', type=str,
                        default='data/Kvasir/training/',
                        help='path to train dataset')
    parser.add_argument('--train_save', type=str,
                        default='./model_pth/' + model_name + '/')
    parser.add_argument('--pretrained_path', type=str,
                        default='pretrained_pth/mit_b3.pth',
                        help='path to the MiT-B3 backbone checkpoint')

    opt = parser.parse_args()

    logging.basicConfig(
        filename='train_log.log',
        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
        level=logging.INFO,
        filemode='a',
        datefmt='%Y-%m-%d %I:%M:%S %p',
    )

    os.makedirs(opt.train_save, exist_ok=True)

    model = CISFNet(pretrained_path=opt.pretrained_path).cuda()
    params = model.parameters()

    if opt.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(
            params,
            opt.lr,
            weight_decay=1e-4,
            momentum=0.9,
        )

    print(optimizer)

    image_root = '{}/images/'.format(opt.train_path.rstrip('/'))
    gt_root = '{}/masks/'.format(opt.train_path.rstrip('/'))
    train_loader = get_loader(
        image_root,
        gt_root,
        batchsize=opt.batchsize,
        trainsize=opt.trainsize,
        augmentation=opt.augmentation,
    )
    total_step = len(train_loader)

    print('#' * 20, 'Start CISF-NET Training', '#' * 20)

    for epoch in range(1, opt.epoch + 1):
        adjust_lr(optimizer, opt.lr, epoch, 0.1, 200)
        current_lr = optimizer.param_groups[0]['lr']
        print('lr:', current_lr)

        train_loss = train(
            train_loader,
            model,
            optimizer,
            epoch,
            total_step,
            opt,
        )
        logging.info(
            'Epoch [%03d/%03d] | lr: %.8f | loss: %.4f',
            epoch,
            opt.epoch,
            current_lr,
            train_loss,
        )

    final_path = os.path.join(opt.train_save, 'final.pth')
    torch.save(model.state_dict(), final_path)
    print('CISF-NET training finished. Final model saved to {}'.format(final_path))
    logging.info('CISF-NET training finished. Final model saved to %s', final_path)
