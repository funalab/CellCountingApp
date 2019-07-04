# -*- coding: utf-8 -*-

import csv
import sys
import time
import random
import copy
import math
import os
import os.path as pt
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage import measure
from skimage import morphology
from skimage.morphology import watershed
from scipy import ndimage
from argparse import ArgumentParser
from utils import Utils

plt.style.use('ggplot')
starttime = time.time()


class Evaluator():
    def __init__(self):
        pass

    def classification_evaluator(self, TP, TN, FP, FN):
        evals = {}
        try:
            evals['accuracy'] = (TP + TN) / float(TP + TN + FP + FN)
        except:
            evals['accuracy'] = 0.0
        try:
            evals['recall'] = TP / float(TP + FN)
        except:
            evals['recall'] = 0.0
        try:
            evals['precision'] = TP / float(TP + FP)
        except:
            evals['precision'] = 0.0
        try:
            evals['F-measure'] = 2 * evals['recall'] * evals['precision'] / (evals['recall'] + evals['precision'])
        except:
            evals['F-measure'] = 0.0
        return evals


    def detection_evaluator(self, PR, GT, thr):
        numPR, numGT = len(PR), len(GT)
        pare = []
        for gn in range(numGT):
            tmp = 0
            chaild = []
            for pn in range(numPR):
                if np.sum((GT[gn] - PR[pn])**2) < thr**2:
                    chaild.append(pn)
                    tmp += 1
            if tmp > 0:
                pare.append(chaild)
        used = np.zeros(numPR)
        TP = self._search_list(pare, used, 0)
        
        evals = {}
        FP = numPR - TP
        FN = numGT - TP
        try:
            evals['recall'] = TP / float(TP + FN)
        except:
            evals['recall'] = 0.0
        try:
            evals['precision'] = TP / float(TP + FP)
        except:
            evals['precision'] = 0.0
        try:
            evals['F-measure'] = 2 * evals['recall'] * evals['precision'] / (evals['recall'] + evals['precision'])
        except:
            evals['F-measure'] = 0.0
        try:
            evals['IoU'] = TP / float(TP + FP + FN)
        except:
            evals['IoU'] = 0.0
        return evals

        
    def _search_list(self, node, used, idx):
        if len(node) == idx:
            return 0
        else:
            tmp = []
            for i in range(len(node[idx])):
                if used[node[idx][i]] == 0:
                    used[node[idx][i]] += 1
                    tmp.append(self._search_list(node, used, idx+1) + 1)
                    used[node[idx][i]] -= 1
                else:
                    tmp.append(self._search_list(node, used, idx+1))
            return np.max(tmp)


if __name__ == '__main__':
    ap = ArgumentParser(description='python evaluation.py')
    ap.add_argument('--seg_image', '-i', nargs='?', default='../images/test_bin_images/segimg_1.tif', help='Specify Evaluation CDN Output Center Image')
    ap.add_argument('--ans_image', '-a', nargs='?', default='../images/test_gt_images/1.tif', help='Specify GT Center Image')
    ap.add_argument('--outdir', '-o', nargs='?', default='result_eval_detection', help='Specify output files directory for evaluation result')
    ap.add_argument('--radius', '-r', type=int, default=10, help='Specify GT radius')
    ap.add_argument('--delete', '-dc', type=int, default=1, help='Specify Delete voxel size less than of center estimation area')

    args = ap.parse_args()
    opbase = args.outdir
    argvs = sys.argv
    util = Utils()
    sys.setrecursionlimit(200000) # 10000 is an example, try with different values

    psep = '/'
    opbase = util.createOpbase(args.outdir)    
    thr = args.radius
    num_delv = args.delete  # Center Image の偽陽性を消すパラメータ (num_delv以下のvoxel領域を削除)
    print('GT Radius Size: {}'.format(args.radius))
    print('Delete Pixel Size: {}'.format(args.delete))
    with open(opbase + psep + 'result.txt', 'w') as f:
        f.write('python ' + ' '.join(argvs) + '\n')
        f.write('[Properties of parameter]\n')
        f.write('Output Directory: {}\n'.format(opbase))
        f.write('GT Radius Size: {}\n'.format(thr))
        f.write('Delete Pixel Size: {}\n'.format(num_delv))
    
    images = io.imread(args.seg_image)
    gt_images = io.imread(args.ans_image)

    # Labeling
    markers = morphology.label(images, neighbors=4)
    mask_size = np.unique(markers, return_counts=True)[1] < (num_delv+1)
    remove_voxel = mask_size[markers]
    markers[remove_voxel] = 0
    labels = np.unique(markers)
    images = np.searchsorted(labels, markers)
    numPR = np.max(images)
    gt_images = morphology.label(gt_images, neighbors=4)
    numGT = np.max(gt_images)

    # Make Centroid List
    props = measure.regionprops(images)
    PRcenter = [np.array([p.centroid[0], p.centroid[1]]) for p in measure.regionprops(images)]
    GTcenter = [np.array([p.centroid[0], p.centroid[1]]) for p in measure.regionprops(gt_images)]

    print('Number of Ground Truth: {}'.format(numGT))
    print('Number of Prediction: {}'.format(numPR))
    with open(opbase + psep + 'result.txt', 'a') as f:
        f.write('number of GT: {}\n'.format(numGT))
        f.write('number of Predict: {}\n'.format(numPR))
        f.write('===========================================\n')

    eva = Evaluator()
    evals = eva.detection_evaluator(PRcenter, GTcenter, thr)
            
    print('Recall: {}'.format(evals['recall']))
    print('Precision: {}'.format(evals['precision']))
    print('F-measure: {}'.format(evals['F-measure']))

    with open(opbase + psep + 'result.txt', 'a') as f:
        f.write('Recall: {}\n'.format(evals['recall']))
        f.write('Precision: {}\n'.format(evals['precision']))
        f.write('F-measure: {}\n'.format(evals['F-measure']))
