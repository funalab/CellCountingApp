# -*- coding: utf-8 -*-

import chainer
from chainer import cuda
import chainer.functions as F
from chainer import training
from chainer.training import extensions, triggers

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
from lib.model import CellClassificationModel as CCM
from lib.wrapper import Regressor

def main():
    start_time = time.time()
    ap = ArgumentParser(description='python train_cc.py')
    ap.add_argument('--indir', '-i', nargs='?', default='datasets/train', help='Specify input files directory for learning data')
    ap.add_argument('--outdir', '-o', nargs='?', default='results/results_training_cc', help='Specify output files directory for create save model files')
    ap.add_argument('--train_list', nargs='?', default='datasets/split_list/train.list', help='Specify split train list')
    ap.add_argument('--validation_list', nargs='?', default='datasets/split_list/validation.list', help='Specify split validation list')
    ap.add_argument('--init_model', help='Specify Loading File Path of Learned Cell Classification Model')
    ap.add_argument('--gpu', '-g', type=int, default=-1, help='Specify GPU ID (negative value indicates CPU)')
    ap.add_argument('--epoch', '-e', type=int, default=10, help='Specify number of sweeps over the dataset to train')
    ap.add_argument('--batchsize', '-b', type=int, default=5, help='Specify Batchsize')
    ap.add_argument('--crop_size', nargs='?', default='(640, 640)', help='Specify crop size (default (y,x) = (640,640))')
    ap.add_argument('--coordinate', nargs='?', default='(780, 1480)', help='Specify initial coordinate (default (y,x) = (1840,700))')
    ap.add_argument('--optimizer', default='SGD', help='Optimizer [SGD, MomentumSGD, Adam]')
    args = ap.parse_args()
    argvs = sys.argv
    psep = '/'

    print('init dataset...')
    train_dataset = PreprocessedRegressionDataset(
        path=args.indir,
        split_list=args.train_list,
        crop_size=args.crop_size,
        coordinate=args.coordinate,
        train=True
    )
    validation_dataset = PreprocessedRegressionDataset(
        path=args.indir,
        split_list=args.validation_list,
        crop_size=args.crop_size,
        coordinate=args.coordinate,
        train=False
    )

    print('init model construction')
    model = Regressor(
        CCM(
            n_class=1
            ), lossfun=F.mean_squared_error
        )

    if args.init_model is not None:
        print('Load model from', args.init_model)
        chainer.serializers.load_npz(args.init_model, model)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    print('init optimizer...')
    if args.optimizer == 'SGD':
        optimizer = chainer.optimizers.SGD(lr=0.0001)
    elif args.optimizer == 'MomentumSGD':
        optimizer = chainer.optimizers.MomentumSGD(lr=0.01)
    elif args.optimizer == 'Adam':
        optimizer = chainer.optimizers.Adam()
    else:
        print('Specify optimizer name')
        sys.exit()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(rate=0.0001))


    ''' Updater '''
    print('init updater')
    train_iter = chainer.iterators.SerialIterator(
        train_dataset, batch_size=args.batchsize)
    validation_iter = chainer.iterators.SerialIterator(
        validation_dataset, batch_size=1, repeat=False, shuffle=False)
    updater = chainer.training.updaters.StandardUpdater(
        train_iter, optimizer, device=args.gpu)


    ''' Trainer '''
    current_datetime = datetime.now(pytz.timezone('Asia/Tokyo')).strftime('%Y%m%d_%H%M%S')
    save_dir = args.outdir + '_' + str(current_datetime)
    os.makedirs(save_dir, exist_ok=True)
    trainer = training.Trainer(updater, stop_trigger=(args.epoch, 'epoch'), out=save_dir)

    '''
    Extensions:
        Evaluator         : Evaluate the segmentor with the validation dataset for each epoch
        ProgressBar       : print a progress bar and recent training status.
        ExponentialShift  : The typical use case is an exponential decay of the learning rate.
        dump_graph        : This extension dumps a computational graph.
        snapshot          : serializes the trainer object and saves it to the output directory
        snapshot_object   : serializes the given object and saves it to the output directory.
        LogReport         : output the accumulated results to a log file.
        PrintReport       : print the accumulated results.
        PlotReport        : output plots.
    '''

    evaluator = extensions.Evaluator(validation_iter, model, device=args.gpu)
    trainer.extend(evaluator, trigger=(1, 'epoch'))

    if args.optimizer == 'SGD':
        trainer.extend(extensions.ExponentialShift('lr', 0.1), trigger=(50, 'epoch'))

    trigger = triggers.MinValueTrigger('validation/main/loss', trigger=(1, 'epoch'))
    trainer.extend(extensions.snapshot_object(model, filename='best_loss_model'), trigger=trigger)

    trainer.extend(chainer.training.extensions.observe_lr(), trigger=(1, 'epoch'))

    # LogReport
    trainer.extend(
        trigger=(1, 'epoch'),
        extension=extensions.LogReport()
    )

    # PrintReport
    trainer.extend(
        extension=extensions.PrintReport([
            'epoch', 'iteration',
            'main/loss', 'validation/main/loss',
            'elapsed_time'])
    )

    # PlotReport
    trainer.extend(
        extension=extensions.PlotReport(
            ['main/loss', 'validation/main/loss'], 'epoch', file_name='loss.png'))
    # trainer.extend(
    #     extension=extensions.PlotReport(
    #         ['main/accuracy', 'validation/main/accuracy'], 'epoch', file_name='accuracy.png'))

    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.run()


if __name__ == '__main__':
    main()
