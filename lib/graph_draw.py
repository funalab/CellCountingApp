# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import sys

plt.style.use('ggplot')

class GraphDraw():
    def __init__(self, epoch, cross, opbase):
        self.epoch = epoch
        self.cross = cross
        self.opbase = opbase

    # Graph Draw of Segmentation
    def graph_draw_segmentation(self, train_eval, test_eval):
        try:
            # Error Rate
            plt.figure(figsize=(8,6))
            plt.plot(range(1, self.epoch + 1, 1), train_eval['error'])
            plt.plot(range(1, self.epoch + 1, 1), test_eval['error'])
            plt.legend(["train_error", "test_error"],loc=1) # upper right
            plt.title("Error Rate of Train and Test")
            plt.xlabel('Epoch')
            plt.ylabel('Error Rate')
            plt.plot()
            figname = 'CNN_ErrorRate_epoch' + str(self.epoch) + '_cross' + str(self.cross) + '.pdf'
            plt.savefig(self.opbase + '/' + figname)
        except:
            pass

        try:
            # Train Evaluation
            plt.figure(figsize=(8,6))
            plt.plot(range(1, self.epoch + 1, 1), train_eval['recall'])
            plt.plot(range(1, self.epoch + 1, 1), train_eval['precision'])
            plt.plot(range(1, self.epoch + 1, 1), train_eval['specificity'])
            plt.plot(range(1, self.epoch + 1, 1), train_eval['F-measure'])
            plt.ylim(np.min([np.min(train_eval['recall']), np.min(train_eval['precision']), np.min(train_eval['specificity']), np.min(train_eval['F-measure'])]) - 0.1, 1.5)
            plt.legend(["Recall of Train", "Precision of Train", "Specificity of Train", "F-measure of Train"],loc=1)
            plt.title("Train Evaluation of Recall, Precision, Specificity and F-measure")
            plt.xlabel('Epoch')
            plt.plot()
            figname = 'CNN_Train_Evaluation_epoch' + str(self.epoch) + '_cross' + str(self.cross) + '.pdf'
            plt.savefig(self.opbase + '/' + figname)
        except:
            pass
        
        try:    
            # Test Evaluation
            plt.figure(figsize=(8,6))
            plt.plot(range(1, self.epoch + 1, 1), test_eval['recall'])
            plt.plot(range(1, self.epoch + 1, 1), test_eval['precision'])
            plt.plot(range(1, self.epoch + 1, 1), test_eval['specificity'])
            plt.plot(range(1, self.epoch + 1, 1), test_eval['F-measure'])
            plt.ylim(np.min([np.min(test_eval['recall']), np.min(test_eval['precision']), np.min(test_eval['specificity']), np.min(test_eval['F-measure'])]) - 0.1, 1.5)
            plt.legend(["Recall of Test", "Precision of Test", "Specificity of Test", "F-measure of Test"],loc=1)
            plt.title("Test Evaluation of Recall, Precision, Specificity and F-measure")
            plt.xlabel('Epoch')
            plt.plot()
            figname = 'CNN_Test_Evaluation_epoch' + str(self.epoch) + '_cross' + str(self.cross) + '.pdf'
            plt.savefig(self.opbase + '/' + figname)
        except:
            pass

        try:
            # IoU
            plt.figure(figsize=(8,6))
            plt.plot(range(1, self.epoch + 1, 1), train_eval['IoU'])
            plt.plot(range(1, self.epoch + 1, 1), test_eval['IoU'])
            plt.ylim(0.0, np.max([np.max(train_eval['IoU']), np.max(test_eval['IoU'])]) + 0.2)
            plt.legend(["train_IoU", "test_IoU"],loc=1) # upper right
            plt.title("Intersection over Union of Train and Test")
            plt.xlabel('Epoch')
            plt.ylabel('IoU')
            plt.plot()
            figname = 'CNN_IoU_epoch' + str(self.epoch) + '_cross' + str(self.cross) + '.pdf'
            plt.savefig(self.opbase + '/' + figname)
        except:
            pass

        try:
            # Simpson coefficient
            plt.figure(figsize=(8,6))
            plt.plot(range(1, self.epoch + 1, 1), train_eval['Simpson'])
            plt.plot(range(1, self.epoch + 1, 1), test_eval['Simpson'])
            plt.ylim(0.0, np.max([np.max(train_eval['Simpson']), np.max(test_eval['Simpson'])]) + 0.2)
            plt.legend(["train_Simpson", "test_Simpson"],loc=1) # upper right
            plt.title("Simpson coefficient of Train and Test")
            plt.xlabel('Epoch')
            plt.ylabel('Simpson coefficient')
            plt.plot()
            figname = 'CNN_Simpson_epoch' + str(self.epoch) + '_cross' + str(self.cross) + '.pdf'
            plt.savefig(self.opbase + '/' + figname)
        except:
            pass

        try:
            # Dice coefficient
            plt.figure(figsize=(8,6))
            plt.plot(range(1, self.epoch + 1, 1), train_eval['Dice'])
            plt.plot(range(1, self.epoch + 1, 1), test_eval['Dice'])
            plt.ylim(0.0, np.max([np.max(train_eval['Dice']), np.max(test_eval['Dice'])]) + 0.2)
            plt.legend(["train_Dice", "test_Dice"],loc=1) # upper right
            plt.title("Dice coefficient of Train and Test")
            plt.xlabel('Epoch')
            plt.ylabel('Dice coefficient')
            plt.plot()
            figname = 'CNN_Dice_epoch' + str(self.epoch) + '_cross' + str(self.cross) + '.pdf'
            plt.savefig(self.opbase + '/' + figname)
        except:
            pass

        try:
            # Loss
            plt.figure(figsize=(8,6))
            plt.plot(range(len(train_eval['loss'])), train_eval['loss'])
            plt.plot(range(len(test_eval['loss'])), test_eval['loss'])
            plt.ylim(0.0, 1.2)
            plt.legend(["Loss of Train", "Loss of Test"],loc=1) # loc='lower right'
            plt.title("Loss of Train and Test")
            plt.xlabel('number of data')
            plt.ylabel('Loss')
            plt.plot()
            figname = 'CNN_Loss_epoch' + str(self.epoch) + '_cross' + str(self.cross) + '.pdf'
            plt.savefig(self.opbase + '/' + figname)
        except:
            pass


    # Graph Draw of Segmentation
    def graph_draw_detection(self, train_eval, test_eval):
        # Loss
        try:
            plt.figure(figsize=(8,6))
            plt.plot(range(len(train_eval['loss'])), train_eval['loss'])
            plt.plot(range(len(test_eval['loss'])), test_eval['loss'])
            plt.ylim(0.0, 1.2)
            plt.legend(["Loss of Train", "Loss of Test"],loc=1) # loc='lower right'
            plt.title("Loss of Train and Test")
            plt.xlabel('number of data')
            plt.ylabel('Loss')
            plt.plot()
            figname = 'CNN_Loss_epoch' + str(self.epoch) + '_cross' + str(self.cross) + '.pdf'
            plt.savefig(self.opbase + '/' + figname)
        except:
            pass

        try:
            # Train Evaluation
            plt.figure(figsize=(8,6))
            plt.plot(range(1, self.epoch + 1, 1), train_eval['recall'])
            plt.plot(range(1, self.epoch + 1, 1), train_eval['precision'])
            plt.plot(range(1, self.epoch + 1, 1), train_eval['specificity'])
            plt.plot(range(1, self.epoch + 1, 1), train_eval['F-measure'])
            plt.ylim(np.min([np.min(train_eval['recall']), np.min(train_eval['precision']), np.min(train_eval['specificity']), np.min(train_eval['F-measure'])]) - 0.1, 1.5)
            plt.legend(["Recall of Train", "Precision of Train", "Specificity of Train", "F-measure of Train"],loc=1)
            plt.title("Train Evaluation of Recall, Precision, Specificity and F-measure")
            plt.xlabel('Epoch')
            plt.plot()
            figname = 'CNN_Train_Evaluation_epoch' + str(self.epoch) + '_cross' + str(self.cross) + '.pdf'
            plt.savefig(self.opbase + '/' + figname)
        except:
            pass

        try:
            # Test Evaluation
            plt.figure(figsize=(8,6))
            plt.plot(range(1, self.epoch + 1, 1), test_eval['recall'])
            plt.plot(range(1, self.epoch + 1, 1), test_eval['precision'])
            plt.plot(range(1, self.epoch + 1, 1), test_eval['specificity'])
            plt.plot(range(1, self.epoch + 1, 1), test_eval['F-measure'])
            plt.ylim(np.min([np.min(test_eval['recall']), np.min(test_eval['precision']), np.min(test_eval['specificity']), np.min(test_eval['F-measure'])]) - 0.1, 1.5)
            plt.legend(["Recall of Test", "Precision of Test", "Specificity of Test", "F-measure of Test"],loc=1)
            plt.title("Test Evaluation of Recall, Precision, Specificity and F-measure")
            plt.xlabel('Epoch')
            plt.plot()
            figname = 'CNN_Test_Evaluation_epoch' + str(self.epoch) + '_cross' + str(self.cross) + '.pdf'
            plt.savefig(self.opbase + '/' + figname)
        except:
            pass
            
        try:
            # IoU
            plt.figure(figsize=(8,6))
            plt.plot(range(1, self.epoch + 1, 1), train_eval['IoU'])
            plt.plot(range(1, self.epoch + 1, 1), test_eval['IoU'])
            plt.ylim(0.0, np.max([np.max(train_eval['IoU']), np.max(test_eval['IoU'])]) + 0.2)
            plt.legend(["train_IoU", "test_IoU"],loc=1) # upper right
            plt.title("Intersection over Union of Train and Test")
            plt.xlabel('Epoch')
            plt.ylabel('IoU')
            plt.plot()
            figname = 'CNN_IoU_epoch' + str(self.epoch) + '_cross' + str(self.cross) + '.pdf'
            plt.savefig(self.opbase + '/' + figname)
        except:
            pass
            
    def graph_draw_classification(self, train_eval, test_eval):
        try:
            # Loss
            plt.figure(figsize=(8,6))
            plt.plot(range(len(train_eval['loss'])), train_eval['loss'])
            plt.plot(range(len(test_eval['loss'])), test_eval['loss'])
            plt.ylim(0.0, 1.2)
            plt.legend(["Loss of Train", "Loss of Test"],loc=1) # loc='lower right'
            plt.title("Loss of Train and Test")
            plt.xlabel('number of data')
            plt.ylabel('Loss')
            plt.plot()
            figname = 'Loss_epoch' + str(self.epoch) + '_cross' + str(self.cross) + '.pdf'
            plt.savefig(self.opbase + '/' + figname)
        except:
            pass

        try:
            # Error Rate
            plt.figure(figsize=(8,6))
            plt.plot(range(1, self.epoch + 1, 1), train_eval['accuracy'])
            plt.plot(range(1, self.epoch + 1, 1), test_eval['accuracy'])
            plt.ylim(0.0, np.max([np.max(train_eval['accuracy']), np.max(test_eval['accuracy'])]) + 0.2)
            plt.legend(["train_accuracy", "test_accuracy"],loc=1) # upper right
            plt.title("Accuracy of Train and Test")
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.plot()
            figname = 'Accuracy_epoch' + str(self.epoch) + '_cross' + str(self.cross) + '.pdf'
            plt.savefig(self.opbase + '/' + figname)
        except:
            pass
        
        try:
            # Train Evaluation
            plt.figure(figsize=(8,6))
            plt.plot(range(1, self.epoch + 1, 1), train_eval['recall'])
            plt.plot(range(1, self.epoch + 1, 1), train_eval['precision'])
            plt.plot(range(1, self.epoch + 1, 1), train_eval['F-measure'])
            plt.ylim(np.min([np.min(train_eval['recall']), np.min(train_eval['precision']), np.min(train_eval['F-measure'])]) - 0.1, 1.5)
            plt.legend(["Recall of Train", "Precision of Train", "F-measure of Train"],loc=1)
            plt.title("Train Evaluation of Recall, Precision and F-measure")
            plt.xlabel('Epoch')
            plt.plot()
            figname = 'Train_Evaluation_epoch' + str(self.epoch) + '_cross' + str(self.cross) + '.pdf'
            plt.savefig(self.opbase + '/' + figname)
        except:
            pass

        try:
            # Test Evaluation
            plt.figure(figsize=(8,6))
            plt.plot(range(1, self.epoch + 1, 1), test_eval['recall'])
            plt.plot(range(1, self.epoch + 1, 1), test_eval['precision'])
            plt.plot(range(1, self.epoch + 1, 1), test_eval['F-measure'])
            plt.ylim(np.min([np.min(test_eval['recall']), np.min(test_eval['precision']), np.min(test_eval['F-measure'])]) - 0.1, 1.5)
            plt.legend(["Recall of Test", "Precision of Test", "F-measure of Test"],loc=1)
            plt.title("Test Evaluation of Recall, Precision and F-measure")
            plt.xlabel('Epoch')
            plt.plot()
            figname = 'Test_Evaluation_epoch' + str(self.epoch) + '_cross' + str(self.cross) + '.pdf'
            plt.savefig(self.opbase + '/' + figname)
        except:
            pass



class GraphDrawCrossVal():
    def __init__(self, cross):
        self.cross = cross
        
    # Graph Draw
    def graph_draw_segmentation(self, acc, rec, pre, fme, iou):

        y = [1, 2, 3, 4, 5]
        acc_mean = np.mean(acc)
        acc_std = np.std(acc)
        rec_mean = np.mean(rec)
        rec_std = np.std(rec)
        pre_mean = np.mean(pre)
        pre_std = np.std(pre)
        fme_mean = np.mean(fme)
        fme_std = np.std(fme)
        iou_mean = np.mean(iou)
        iou_std = np.std(iou)

        mean = [iou_mean, fme_mean, pre_mean, rec_mean, acc_mean]
        std = [iou_std, fme_std, pre_std, rec_std, acc_std]
        print mean
        print std

        plt.figure(figsize=(8,6))
        plt.barh(y, mean, align='center', height=0.6, xerr=std, ecolor='black')
        plt.yticks(y, ['IoU', 'F-measure', 'Precision', 'Recall', 'Accuracy'])
        plt.ylim([0, 6])
        #plt.legend(["Loss of Train", "Loss of Test"],loc=1) # loc='lower right'
        plt.title("Evaluation of Nuclei Segmentation Model Training (6-fold cross validation)")
        plt.xlabel('')
        plt.ylabel('')
        plt.plot()
        #figname = 'CNN_Loss_epoch' + str(self.epoch) + '_cross' + str(self.cross) + '.pdf'
        #plt.savefig(self.opbase + '/' + figname)


    def graph_draw_detection(self, rec, pre, fme):

        y = [1, 2, 3]
        rec_mean = np.mean(rec)
        rec_std = np.std(rec)
        pre_mean = np.mean(pre)
        pre_std = np.std(pre)
        fme_mean = np.mean(fme)
        fme_std = np.std(fme)

        mean = [fme_mean, pre_mean, rec_mean]
        std = [fme_std, pre_std, rec_std]
        print mean
        print std

        plt.figure(figsize=(8,6))
        plt.barh(y, mean, align='center', height=0.4, xerr=std, ecolor='black')
        plt.yticks(y, ['F-measure', 'Precision', 'Recall'])
        plt.xlim([0.0, 1.05])
        plt.ylim([0, 4])
        #plt.legend(["Loss of Train", "Loss of Test"],loc=1) # loc='lower right'                                                  
        plt.title("Evaluation of Cell Detection Network Training (5-fold cross validation)")
        plt.xlabel('')
        plt.ylabel('')
        plt.plot()
        #figname = 'CNN_Loss_epoch' + str(self.epoch) + '_cross' + str(self.cross) + '.pdf'                                       
        #plt.savefig(self.opbase + '/' + figname)

        
    def graph_draw_classification(self, aca_r0, mca_r0, aca_r50, mca_r50, aca_r100, mca_r100,
                                  aca_1d_r50, mca_1d_r50):

        y = [1.4, 2.1, 2.9, 3.6, 5.4, 6.1, 6.9, 7.6]
        aca_r0_mean = np.mean(aca_r0)
        aca_r0_std = np.std(aca_r0)
        mca_r0_mean = np.mean(mca_r0)
        mca_r0_std = np.std(mca_r0)
        aca_r50_mean = np.mean(aca_r50)
        aca_r50_std = np.std(aca_r50)
        mca_r50_mean = np.mean(mca_r50)
        mca_r50_std = np.std(mca_r50)
        aca_r100_mean = np.mean(aca_r100)
        aca_r100_std = np.std(aca_r100)
        mca_r100_mean = np.mean(mca_r100)
        mca_r100_std = np.std(mca_r100)
        aca_1d_r50_mean = np.mean(aca_1d_r50)
        aca_1d_r50_std = np.std(aca_1d_r50)
        mca_1d_r50_mean = np.mean(mca_1d_r50)
        mca_1d_r50_std = np.std(mca_1d_r50)

        mean_r0 = [mca_r0_mean, aca_r0_mean]
        std_r0  = [mca_r0_std, aca_r0_std]
        mean_r50 = [mca_r50_mean, aca_r50_mean]
        std_r50  = [mca_r50_std, aca_r50_std]
        mean_r100 = [mca_r100_mean, aca_r100_mean]
        std_r100  = [mca_r100_std, aca_r100_std]
        mean_1d_r50 = [mca_1d_r50_mean, aca_1d_r50_mean]
        std_1d_r50  = [mca_1d_r50_std, aca_1d_r50_std]
        mean = [mca_1d_r50_mean, mca_r100_mean, mca_r50_mean, mca_r0_mean,
                aca_1d_r50_mean, aca_r100_mean, aca_r50_mean, aca_r0_mean]
        std  = [mca_1d_r50_std, mca_r100_std, mca_r50_std, mca_r0_std,
                aca_1d_r50_std, aca_r100_std, aca_r50_std, aca_r0_std]
        print mean
        print std

        plt.figure(figsize=(8,6))
        y = [4, 9]
        plt.barh(y, mean_r0, align='center', height=1.0, xerr=std_r0, ecolor='black', label='No Augmentation')
        y = [3, 8]
        plt.barh(y, mean_r50, align='center', height=1.0, xerr=std_r50, ecolor='black', label='Augmentation r=50')
        y = [2, 7]
        plt.barh(y, mean_r100, align='center', height=1.0, xerr=std_r100, ecolor='black', label='Augmentation r=100')
        y = [1, 6]
        plt.barh(y, mean_1d_r50, align='center', height=1.0, xerr=std_1d_r50, ecolor='black', label='1D Augmentation r=50')
        plt.yticks([2.5, 7.5], ['MCA', 'ACA'])
        plt.ylim([-1, 13])
        plt.legend(loc=1) # loc='lower right'
        plt.title("Evaluation of Cell Count Network Training (5-fold cross validation)")
        plt.xlabel('')
        plt.ylabel('')
        plt.plot()
        #figname = 'CNN_Loss_epoch' + str(self.epoch) + '_cross' + str(self.cross) + '.pdf'
        #plt.savefig(self.opbase + '/' + figname)


        

if __name__ == '__main__':

    ### Make Fig for Cross Validation ###
    
    aca_r0 = [1.0, 1.0, 1.0, 1.0, 1.0]
    mca_r0 = [1.0, 1.0, 1.0, 1.0, 1.0]
    aca_r50 = [1.0, 1.0, 1.0, 1.0, 1.0]
    mca_r50 = [1.0, 1.0, 1.0, 1.0, 1.0]
    aca_r100 = [0.933333333333, 0.933333333333, 0.966666666667, 0.933333333333, 0.966666666667]
    mca_r100 = [0.933333333333, 0.933333333333, 0.966666666667, 0.933333333333, 0.966666666667]
    aca_1d_r50 = [0.833333333333, 0.8, 0.6, 0.8, 0.7]
    mca_1d_r50 = [0.833333333333, 0.8, 0.6, 0.8, 0.7]
    
    gdcv = GraphDrawCrossVal(5)
    gdcv.graph_draw_classification(aca_r0, mca_r0, aca_r50, mca_r50, aca_r100, mca_r100,
                                   aca_1d_r50, mca_1d_r50)

    sys.exit()

    recall = [0.851041666667, 0.880952380952, 0.830917874396, 0.864948453608, 0.826353421859]
    precision = [0.741379310345, 0.694706994329, 0.69918699187, 0.745777777778, 0.740842490842]
    fmeasure = [0.792434529583, 0.77449947313, 0.759381898455, 0.800954653938, 0.781265089329]
    gdcv.graph_draw_detection(recall, precision, fmeasure)

    
    ### Make Fig for Training Evaluation ###

    import csv
    f = open('../CellDetectionAndClassification/LearnedModel_CellDetetction/CellDet-noAug_Nov16Thu_2017_032925/TestResult.csv')
    csvReader = csv.reader(f)
    train = []
    for i in csvReader:
        train.append(i)
    f.close()
    train.pop(0)
    f = open('../CellDetectionAndClassification/LearnedModel_CellDetetction/CellDet-noAug_Nov16Thu_2017_032925/TrainResult.csv')
    csvReader = csv.reader(f)
    test = []
    for i in csvReader:
        test.append(i)
    f.close()
    test.pop(0)
    
    criteria = ['error', 'recall', 'precision', 'specificity', 'F-measure', 'IoU', 'Simpson', 'Dice']
    train_eval = {}
    test_eval = {}
    for cri in criteria:
        train_eval[cri] = []
        test_eval[cri] = []

    for i in range(len(train) / 5):
        for cri in range(len(criteria)):
            train_eval[criteria[cri]].append(train[i][cri+1])
            test_eval[criteria[cri]].append(test[i][cri+1])
    
    gd = GraphDraw(epoch=50, cross=5, opbase='hoge')
    gd.graph_draw_segmentation(train_eval, test_eval)
