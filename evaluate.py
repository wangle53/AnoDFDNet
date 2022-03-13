#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import print_function
import os
from sklearn.metrics import roc_curve, auc, average_precision_score, f1_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import constants as ct
import numpy as np

def evaluatation(labels, scores, metric, best_auc):
    if metric == 'roc':
        return roc(labels, scores, best_auc)
    elif metric == 'auprc':
        return auprc(labels, scores)
    elif metric == 'f1_score':
        threshold = 0.50
        scores[scores >= threshold] = 1
        scores[scores <  threshold] = 0
        return f1_score(labels.cpu(), scores.cpu())
    else:
        raise NotImplementedError("Check the evaluation metric.")

def roc(labels, scores, best_auc, epoch, saveto='./outputs/valROC', ):
    
    """Compute ROC curve and ROC area for each class"""
    labels = np.asarray(labels).reshape(len(labels)*ct.ISIZE*ct.ISIZE)
    labels[labels >= ct.THRESHOLD] = 1
    labels[labels < ct.THRESHOLD] = 0
    scores = np.asarray(scores).reshape(len(scores)*ct.ISIZE*ct.ISIZE)
    # True/False Positive Rates.
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    data = np.column_stack((fpr, tpr))
    auc_score = auc(fpr, tpr)
    # Equal Error Rate
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    
    if saveto:
        if not os.path.exists(saveto):
            os.makedirs(saveto)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='(AUC = %0.4f, EER = %0.4f)' % (auc_score, eer))
        plt.plot([0, 1], [1, 0], color='navy', lw=1, linestyle=':')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(saveto, "ROC_Curve_Epoch_{}_AUC_{:.4f}.jpg".format(epoch, auc_score)))
        if auc_score>best_auc:
            plt.savefig(os.path.join(saveto, "Best_ROC.jpg"))
        plt.close()
    if os.path.exists(os.path.join(saveto, 'fpr,t_pr.txt')):
        os.remove(os.path.join(saveto, 'fpr_tpr.txt'))
    np.savetxt(os.path.join(saveto, 'fpr_tpr.txt'), data, fmt='%.6f %.6f')

    return auc_score

def auprc(labels, scores):
    ap = average_precision_score(labels.cpu(), scores.cpu())
    return ap

def confuse_matrix(score, lb):
    lb = lb.cpu().numpy()
    score = np.array(score.detach().squeeze(0).cpu())
    threshold = ct.THRESHOLD
    score[score>threshold] = 1.0
    score[score<=threshold] = 0.0 
    lb[lb>threshold] = 1.0
    lb[lb<=threshold] = 0.0 
    lb = lb[0,:,:]
    lb = np.round(lb)
    
    tp = np.sum(lb*score)
    fn = lb-score
    fn[fn<0]=0
    fn = np.sum(fn)
    tn = lb+score
    tn[tn>0]=-1
    tn[tn>=0]=1
    tn[tn<0]=0
    tn = np.sum(tn)
    fp = score - lb
    fp[fp<0] = 0
    fp = np.sum(fp)
    
    return tp, fp, tn, fn
     
def eva_metrics(TP, FP, TN, FN):
    precision = TP/(TP+FP+1e-8)
    oa = (TP+TN)/(TP+FN+TN+FP+1e-8)
    recall = TP/(TP+FN+1e-8)
    f1 = 2*precision*recall/(precision+recall+1e-8)
    iou = TP/(FN+TP+FP+1e-8)
    P = ((TP+FP)*(TP+FN)+(FN+TN)*(FP+TN))/((TP+TN+FP+FN)**2+1e-8)
    results = {'iou':iou,'precision':precision,'oa':oa,'recall':recall,'f1':f1}
    return results
