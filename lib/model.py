# -*- coding: utf-8 -*-

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

class CCNet(Chain):
    def __init__(self, n_class=10):
        super(CCNet, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 16, 13, 1, 6)
            self.bn1 = L.BatchNormalization(16)
            self.conv2 = L.Convolution2D(None, 32, 11, 1, 5)
            self.bn2 = L.BatchNormalization(32)

            self.conv3 = L.Convolution2D(None, 32, 9, 1, 4)
            self.bn3 = L.BatchNormalization(32)
            self.conv4 = L.Convolution2D(None, 64, 7, 1, 3)
            self.bn4 = L.BatchNormalization(64)

            self.conv5 = L.Convolution2D(None, 64, 7, 1, 3)
            self.bn5 = L.BatchNormalization(64)
            self.conv6 = L.Convolution2D(None, 128, 5, 1, 2)
            self.bn6 = L.BatchNormalization(128)

            self.conv7 = L.Convolution2D(None, 128, 5, 1, 2)
            self.bn7 = L.BatchNormalization(128)
            self.conv8 = L.Convolution2D(None, 256, 3, 1, 1)
            self.bn8 = L.BatchNormalization(256)

            self.conv9 = L.Convolution2D(None, 256, 3, 1, 1)
            self.bn9 = L.BatchNormalization(256)
            self.conv10 = L.Convolution2D(None, 256, 3, 1, 1)
            self.bn10 = L.BatchNormalization(256)

            self.conv11 = L.Convolution2D(None, 256, 3, 1, 1)
            self.bn11 = L.BatchNormalization(256)
            self.conv12 = L.Convolution2D(None, 256, 3, 1, 1)
            self.bn12 = L.BatchNormalization(256)

            self.fc13 = L.Linear(None, 10240)
            self.fc14 = L.Linear(None, 1028)
            self.fc15 = L.Linear(None, n_class)

    def __call__(self, x):

        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(self.bn1(h)))
        h = F.max_pooling_2d(self.bn2(h), 2, stride=2)

        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(self.bn3(h)))
        h = F.max_pooling_2d(self.bn4(h), 2, stride=2)

        h = F.relu(self.conv5(h))
        h = F.relu(self.conv6(self.bn5(h)))
        h = F.max_pooling_2d((self.bn6(h)), 2, stride=2)

        h = F.relu(self.conv7(h))
        h = F.relu(self.conv8(self.bn7(h)))
        h = F.max_pooling_2d((self.bn8(h)), 2, stride=2)

        h = F.relu(self.conv9(h))
        h = F.relu(self.conv10(self.bn9(h)))
        h = F.max_pooling_2d((self.bn10(h)), 2, stride=2)

        h = F.relu(self.conv11(h))
        h = F.relu(self.conv12(self.bn11(h)))
        h = F.max_pooling_2d((self.bn12(h)), 2, stride=2)

        h = F.dropout(F.relu(self.fc13(h)))
        h = F.dropout(F.relu(self.fc14(h)))
        return self.fc15(h)
