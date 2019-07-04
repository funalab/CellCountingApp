# -*- coding: utf-8 -*-

'''
Label Annotation
0: 100
1: 500
2: 1000
'''

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
from skimage import io
from argparse import ArgumentParser
sys.path.append(os.getcwd())

from lib.dataset import PreprocessedDataset
from lib.model import CellClassificationModel as CCM
from lib.wrapper import Classifier

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
    ap.add_argument('--coordinate', nargs='?', default='(780, 1480)', help='Specify initial coordinate (default (y,x) = (780,1480))')
    ap.add_argument('--nclass', type=int, default=10, help='Specify classification class')

    args = ap.parse_args()
    argvs = sys.argv
    psep = '/'

    train_dataset = PreprocessedDataset(
        path=args.indir,
        split_list=args.train_list,
        crop_size=args.crop_size,
        coordinate=args.coordinate,
        train=True
    )
    validation_dataset = PreprocessedDataset(
        path=args.indir,
        split_list=args.validation_list,
        crop_size=args.crop_size,
        coordinate=args.coordinate,
        train=False
    )

    c_weight = np.ones((args.nclass)).astype(np.float32)
    if args.gpu >= 0:
        c_weight = cuda.to_gpu(c_weight)
        use_cudnn = True
    else:
        use_cudnn = False

    model = Classifier(
        CCM(
            class_weight=c_weight,
            n_class=args.nclass,
            use_cudnn=use_cudnn
            ), lossfun=F.softmax_cross_entropy
        )

    if args.init_model is not None:
        print('Load model from', args.init_model)
        chainer.serializers.load_npz(args.init_model, model)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(rate=0.0005))


    ''' Updater '''
    train_iter = chainer.iterators.SerialIterator(
        train_dataset, batch_size=args.batchsize)
    validation_iter = chainer.iterators.SerialIterator(
        validation_dataset, batch_size=1, repeat=False, shuffle=False)
    updater = chainer.training.updaters.StandardUpdater(
        train_iter, optimizer, device=args.gpu)


    ''' Trainer '''
    current_datetime = datetime.now(pytz.timezone('Asia/Tokyo')).strftime('%Y%m%d_%H%M%S')
    #save_dir = os.path.join(args.out, current_datetime)
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

    trigger = triggers.MaxValueTrigger('validation/main/mean_accuracy', trigger=(1, 'epoch'))
    trainer.extend(extensions.snapshot_object(model, filename='best_mean_acc_model'), trigger=trigger)

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
            'main/accuracy', 'validation/main/accuracy',
            'elapsed_time'])
    )

    #trainer.extend(extensions.ProgressBar(update_interval=eval(args.report_trigger)[0]))

    # PlotReport
    trainer.extend(
        extension=extensions.PlotReport(
            ['main/loss', 'validation/main/loss'], 'epoch', file_name='loss.png'))
    trainer.extend(
        extension=extensions.PlotReport(
            ['main/accuracy', 'validation/main/accuracy'], 'epoch', file_name='accuracy.png'))

    trainer.extend(extensions.dump_graph('main/loss'))

    trainer.run()


if __name__ == '__main__':
    main()
