# -*- coding: utf-8 -*-

import numpy as np
import sklearn 
import tensorflow
import functools
import Bio
import sys
from Bio import SeqIO
#import PipeTasks

sys.path.insert(0, 'feat_extract')

import pipeline




trainingDir='train'
testingDir='test' 
resultsDir='result' 
GetTrainingFeatures=False
GetTestFeatures=True
classType='file'
classifierType='SVCrbf'#'forest'#
outputTrainedModel=False


result=pipeline.res(trainingDir, testingDir, resultsDir, GetTrainingFeatures, GetTestFeatures, classType, classifierType, outputTrainedModel)
pipeline.pipeline(result)

#x=PipeTasks.GetKFeatures('train/trainingSetFeatures.csv')

### SGD LSVC forest SVCrbf SVCpoly

#==============================================================================
# Results
#==============================================================================
"""
SVM rbf kernel
             precision    recall  f1-score   support

       cyto       0.61      0.60      0.61       750
       mito       0.64      0.75      0.69       325
    nucleus       0.72      0.65      0.69       828
   secreted       0.73      0.83      0.78       334

avg / total       0.68      0.68      0.67      2237

0.67545820295
####################################################

SVM poly kernel
             precision    recall  f1-score   support

       cyto       0.42      0.89      0.57       750
       mito       0.58      0.29      0.38       325
    nucleus       0.81      0.29      0.43       828
   secreted       0.91      0.51      0.66       334

avg / total       0.66      0.52      0.50      2237

0.52168082253

####################################################

Random Forest
             precision    recall  f1-score   support

       cyto       0.59      0.56      0.57       750
       mito       0.71      0.56      0.63       325
    nucleus       0.63      0.72      0.67       828
   secreted       0.77      0.74      0.76       334

avg / total       0.65      0.65      0.64      2237

0.64550737595
####################################################

SGD
             precision    recall  f1-score   support

       cyto       0.54      0.54      0.54       750
       mito       0.63      0.57      0.60       325
    nucleus       0.64      0.70      0.67       828
   secreted       0.77      0.66      0.71       334

avg / total       0.62      0.62      0.62      2237

0.621367903442



"""
