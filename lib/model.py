# -*- coding: utf-8 -*-

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

class CellClassificationModel(Chain):
    def __init__(self, class_weight, n_class=2, use_cudnn=False):
        initializer = chainer.initializers.HeNormal()
        self.class_weight = class_weight
        self.use_cudnn = use_cudnn
        super(CellClassificationModel, self).__init__(
            conv1 = L.Convolution2D(None, 32,  11, initialW=initializer),
            bn1 = L.BatchNormalization(32),
            conv2 = L.Convolution2D(32, 32,  11, initialW=initializer),
            bn2 = L.BatchNormalization(32),
            conv3 = L.Convolution2D(32, 64, 9, initialW=initializer),
            bn3 = L.BatchNormalization(64),

            conv4 = L.Convolution2D(64, 64, 9, initialW=initializer),
            bn4 = L.BatchNormalization(64),
            conv5 = L.Convolution2D(64, 64, 9, initialW=initializer),
            bn5 = L.BatchNormalization(64),
            conv6 = L.Convolution2D(64, 128, 7, initialW=initializer),
            bn6 = L.BatchNormalization(128),

            conv7 = L.Convolution2D(128, 128, 7, initialW=initializer),
            bn7 = L.BatchNormalization(128),
            conv8 = L.Convolution2D(128, 256, 5, initialW=initializer),
            bn8 = L.BatchNormalization(256),

            conv9 = L.Convolution2D(256, 256, 5, initialW=initializer),
            bn9 = L.BatchNormalization(256),
            conv10 = L.Convolution2D(256, 256, 5, initialW=initializer),
            bn10 = L.BatchNormalization(256),
            conv11 = L.Convolution2D(256, 256, 3, initialW=initializer),
            bn11 = L.BatchNormalization(256),

            fc12 = L.Linear(None, 10240, initialW=initializer),
            fc13 = L.Linear(None, 1028, initialW=initializer),
            fc14 = L.Linear(None, n_class, initialW=initializer)
        )

    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(self.bn1(h)))
        h = F.relu(self.conv3(self.bn2(h)))
        h = F.max_pooling_2d(self.bn3(h), 3, stride=3)
        h = F.relu(self.conv4(h))
        h = F.relu(self.conv5(self.bn4(h)))
        h = F.relu(self.conv6(self.bn5(h)))
        h = F.max_pooling_2d(self.bn6(h), 3, stride=3)
        h = F.relu(self.conv7(h))
        h = F.relu(self.conv8(self.bn7(h)))
        h = F.max_pooling_2d((self.bn8(h)), 2, stride=2)
        h = F.relu(self.conv9(h))
        h = F.relu(self.conv10(self.bn9(h)))
        h = F.relu(self.conv11(self.bn10(h)))
        h = F.max_pooling_2d((self.bn11(h)), 2, stride=2)
        h = F.dropout(F.relu(self.fc12(h)))
        h = F.dropout(F.relu(self.fc13(h)))
        return self.fc14(h)
