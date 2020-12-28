# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import argparse
import os
import pprint
import shutil
import sys

import logging
import time
import timeit
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import _init_paths
import models
import datasets
from config import config
from config import update_config
from core.function import testval, test
from utils.modelsummary import get_model_summary
from utils.utils import create_logger, FullModel

# rlaunch --cpu=16 --gpu=1 --memory=$((120*1024)) --max-wait-time 10h --preemptible=no --charged-group v_tracking -- python3 tools/test.py --cfg experiments/daodixian_seg/seg_hrnet_ocr_w48_epoch48_1209.yaml

# rlaunch --cpu=16 --gpu=1 --memory=$((120*1024)) --max-wait-time 10h --preemptible=no --charged-group v_tracking -- python3 tools/test.py --cfg experiments/daodixian_seg/seg_hrnet_ocr_w48_epoch48_1209.yaml TEST.MODEL_FILE outputs/daodixian_seg_1209_bs_8_and_validate/daodixian_seg/seg_hrnet_ocr_w48_epoch48_1209/best.pth  TEST.SCALE_LIST 0.5,0.75,1.0,1.25,1.5,1.75 TEST.FLIP_TEST True

# rlaunch --cpu=16 --gpu=1 --memory=$((120*1024)) --max-wait-time 10h --preemptible=no --charged-group v_tracking -- python3 tools/test.py --cfg experiments/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_8_epoch48_1118.yaml TEST.MODEL_FILE outputs/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_8_epoch48_1118/best.pth TEST.SCALE_LIST 0.5,0.75,1.0,1.25,1.5,1.75 TEST.FLIP_TEST True

# val时候  调用testval 函数（默认多尺度推理）
# test 无标注时候 调用test函数，同时指定保存路径，保存预测图片（默认多尺度推理，多尺度推理时batchsize必须为1）
# 注意：训练时候，如果不开启multi_scale 的话，会使用原图训练，但是导地线原图太大，会溢出显存，所以必须开启
# 所以模型一边训练，一边验证时候，这个验证的test_dataset也要开multi_scale（multi_scale的时候会把原图进行crop resize）
# 训练完单独测试时候（调用test.py），不用开启multi_scale，这时候会自动调用test_dataset.multi_scale_inference（以base_size为基础进行resize）
# yaml里面的参数不能随便乱加，default里面有才能加
# 分割网络对输入大小好像没有要求 任何尺寸都行  训练和测试的尺寸不要求保持一致？


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args

def main():
    args = parse_args()

    logger, final_output_dir, _ = create_logger(
        config, args.cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # build model
    if torch.__version__.startswith('1'):
        module = eval('models.'+config.MODEL.NAME)
        module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d
    model = eval('models.'+config.MODEL.NAME +
                 '.get_seg_model')(config)

    dump_input = torch.rand(
        (1, 3, config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    )
    logger.info(get_model_summary(model.cuda(), dump_input.cuda()))

    if config.TEST.MODEL_FILE:
        model_state_file = config.TEST.MODEL_FILE
    else:
        model_state_file = os.path.join(final_output_dir, 'final_state.pth')        
    logger.info('=> loading model from {}'.format(model_state_file))
        
    pretrained_dict = torch.load(model_state_file)
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                        if k[6:] in model_dict.keys()}
    for k, _ in pretrained_dict.items():
        logger.info(
            '=> loading {} from pretrained model'.format(k))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    gpus = list(config.GPUS)
    model = nn.DataParallel(model, device_ids=gpus).cuda()

    # prepare data
    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    test_dataset = eval('datasets.'+config.DATASET.DATASET)(
                        root=config.DATASET.ROOT,
                        list_path=config.DATASET.TEST_SET,
                        num_samples=None,
                        num_classes=config.DATASET.NUM_CLASSES,
                        # 源代码 默认 false
                        multi_scale=False,
                        flip=False,
                        # multi_scale=config.TEST.MULTI_SCALE,
                        # flip=config.TEST.FLIP,
                        ignore_label=config.TRAIN.IGNORE_LABEL,
                        base_size=config.TEST.BASE_SIZE,
                        crop_size=test_size,
                        downsample_rate=1)

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True)
    
    start = timeit.default_timer()
    if 'val' in config.DATASET.TEST_SET:
        mean_IoU, IoU_array, pixel_acc, mean_acc = testval(config, 
                                                           test_dataset, 
                                                           testloader, 
                                                           model,
                                                           sv_dir=final_output_dir)
    
        msg = 'MeanIOU: {: 4.4f}, Pixel_Acc: {: 4.4f}, \
            Mean_Acc: {: 4.4f}, Class IoU: '.format(mean_IoU, 
            pixel_acc, mean_acc)
        logging.info(msg)
        logging.info(IoU_array)
    elif 'test' in config.DATASET.TEST_SET:
        test(config, 
             test_dataset, 
             testloader, 
             model,
             sv_dir=final_output_dir)

    end = timeit.default_timer()
    # logger.info('Mins: %d' % np.int((end-start)/60))
    logger.info('seconds: %.4f' % np.float((end-start)))
    logger.info('Done')


if __name__ == '__main__':
    main()
