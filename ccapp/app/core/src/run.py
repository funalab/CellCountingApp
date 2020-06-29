# -*- coding: utf-8 -*-

import chainer
from chainer import cuda
import chainer.functions as F

import csv
import sys
import time
import random
import copy
import os
import numpy as np
from datetime import datetime
import pytz
import skimage.io as io

sys.path.append(os.getcwd())
from ..lib.dataset import min_max_normalize_one_image, crop_pair_2d
from ..lib.model import CCNet
from ..lib.wrapper import Regressor

# 追加　菊原
from django.conf import settings

def run(image_path_a, image_path_b, gpu):
    start_time = time.time()
    crop_size = '(320,320)'
    coordinate = '(2100,950)'
    print(image_path_a)
    print(image_path_b)
    print(os.getcwd())
    init_model = settings.BASE_DIR + '/app/core/models/reg.npz'

    model = Regressor(
        CCNet(
            n_class=1
            ), lossfun=F.mean_squared_error
        )

    chainer.serializers.load_npz(init_model, model)
    if gpu >= 0:
        chainer.cuda.get_device_from_id(gpu).use()
        model.to_gpu()

    if io.imread(image_path_a).shape[0] >= io.imread(image_path_a).shape[1]:
        img_a = min_max_normalize_one_image(crop_pair_2d(
                io.imread(image_path_a).transpose(2, 0, 1)
                , crop_size=eval(crop_size), coordinate=eval(coordinate), aug_flag=False))
        img_b = min_max_normalize_one_image(crop_pair_2d(
                io.imread(image_path_b).transpose(2, 0, 1)
                , crop_size=eval(crop_size), coordinate=eval(coordinate), aug_flag=False))
    elif io.imread(image_path_a).shape[0] < io.imread(image_path_a).shape[1]:
        img_a = min_max_normalize_one_image(crop_pair_2d(
                np.rot90(io.imread(image_path_a)).transpose(2, 0, 1)
                , crop_size=eval(crop_size), coordinate=eval(coordinate), aug_flag=False))
        img_b = min_max_normalize_one_image(crop_pair_2d(
                np.rot90(io.imread(image_path_b)).transpose(2, 0, 1)
                , crop_size=eval(crop_size), coordinate=eval(coordinate), aug_flag=False))
    else:
        return 'image invalid!'

    input = np.concatenate([img_a, img_b]).astype(np.float32)

    x = np.expand_dims(input, axis=0)
    if gpu >= 0:
        x = chainer.cuda.to_gpu(x)
    y = model.predict(x)
    if gpu >= 0:
        y = chainer.cuda.to_cpu(y.data)
    else:
        y = y.data
    pre = y[0][0] * 10
    # print('{:.3g} x 10^6'.format(np.round(pre.data, 3)))
    return np.round(pre, 3)
    # end_time = time.time()
    # etime = end_time - start_time
    # print('Elapsed time is (sec) {}'.format(etime))
    # print('CCN Completed Process!')

# if __name__ == '__main__':
#     run('datasets/test/a.jpeg', 'datasets/test/b.jpeg', -1)
