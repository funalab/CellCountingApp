# -*- coding: utf-8 -*-

import chainer
from chainer import cuda
import chainer.functions as F
from chainer import training
from chainer.training import extensions, triggers
from chainer.functions.activation import softmax

import csv
import sys
import time
import random
import copy
import os
import numpy as np
from datetime import datetime
import pytz
from argparse import ArgumentParser
import matplotlib as mpl

sys.path.append(os.getcwd())
mpl.use('Agg')

from lib.dataset import PreprocessedRegressionDataset
from lib.model import CCNet
from lib.wrapper import Regressor

def main():

    start_time = time.time()
    ap = ArgumentParser(description='python test_cc.py')
    ap.add_argument('--indir', '-i', nargs='?', default='datasets/test', help='Specify input files directory for learning data')
    ap.add_argument('--outdir', '-o', nargs='?', default='results/result_test', help='Specify output files directory for create save model files')
    ap.add_argument('--test_list', nargs='?', default='datasets/split_list/test.list', help='Specify split test list')
    ap.add_argument('--init_model', help='Specify Loading File Path of Learned Cell Classification Model')
    ap.add_argument('--gpu', '-g', type=int, default=-1, help='Specify GPU ID (negative value indicates CPU)')
    ap.add_argument('--crop_size', nargs='?', default='(640, 640)', help='Specify crop size (default (y,x) = (640,640))')
    ap.add_argument('--coordinate', nargs='?', default='(780, 1480)', help='Specify initial coordinate (default (y,x) = (1840,700))')
    ap.add_argument('--nclass', type=int, default=10, help='Specify classification class')

    args = ap.parse_args()
    argvs = sys.argv
    current_datetime = datetime.now(pytz.timezone('Asia/Tokyo')).strftime('%Y%m%d_%H%M%S')
    opbase = args.outdir + '_' + str(current_datetime)
    os.makedirs(opbase, exist_ok=True)


    print('init dataset...')
    test_dataset = PreprocessedRegressionDataset(
        path=args.indir,
        split_list=args.test_list,
        crop_size=args.crop_size,
        coordinate=args.coordinate,
        train=False
    )

    print('init model construction')
    model = Regressor(
        CCNet(
            n_class=1
            ), lossfun=F.mean_squared_error
        )

    if args.init_model is not None:
        print('Load model from', args.init_model)
        chainer.serializers.load_npz(args.init_model, model)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    with open(os.path.join(opbase, 'result.csv'), 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['file name', 'prediction', 'label'])

    tp_cnt = 0
    for num in range(test_dataset.__len__()):
        input, _ = test_dataset.get_example(num)
        label = float(test_dataset.split_list[num][:test_dataset.split_list[num].find('_')])
        x = np.expand_dims(input, axis=0)
        if args.gpu >= 0:
            x = chainer.cuda.to_gpu(x)
        y = model.predict(x)
        if args.gpu >= 0:
            y = chainer.cuda.to_cpu(y.data)
        else:
            y = y.data
        pre = y[0][0] * 10

        #if int(round(label)) == int(round(pre)):
        #    print('True')
        #    tp_cnt += 1
        #else:
        #    print('False')
        print()
        print('inference: {}'.format(pre))
        print('ground truth: {}'.format(label))

        with open(os.path.join(opbase, 'result.csv'), 'a') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow([test_dataset.split_list[num], pre, label])

    #with open(os.path.join(opbase, 'result.txt'), 'w') as f:
    #    f.write('Accuracy: {}%'.format(( tp_cnt / test_dataset.__len__() ) * 100))

    end_time = time.time()
    etime = end_time - start_time
    print()
    print('Elapsed time is (sec) {}'.format(etime))
    print('CCN Completed Process!')

if __name__ == '__main__':
    main()
