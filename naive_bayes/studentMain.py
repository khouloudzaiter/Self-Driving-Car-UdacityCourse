# -*- coding: utf-8 -*-
"""
Created on Mon Oct 09 17:24:34 2017

@author: khouloud
"""

#!/usr/bin/python

""" Complete the code in ClassifyNB.py with the sklearn
    Naive Bayes classifier to classify the terrain data.
    
    The objective of this exercise is to recreate the decision 
    boundary found in the lesson video, and make a plot that
    visually shows the decision boundary """

import sys
sys.path.insert(0, '../')
from common.prep_terrain_data import makeTerrainData
from common.class_vis import prettyPicture, output_image
from ClassifyNB import classify,NBAccuracy


import numpy as np
import pylab as pl


features_train, labels_train, features_test, labels_test = makeTerrainData()

def submitAccuracy():
    accuracy = NBAccuracy(features_train, labels_train, features_test, labels_test)
    return accuracy

### the training data (features_train, labels_train) have both "fast" and "slow" points mixed
### in together--separate them so we can give them different colors in the scatterplot,
### and visually identify them
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


# You will need to complete this function imported from the ClassifyNB script.
# Be sure to change to that code tab to complete this quiz.
clf = classify(features_train, labels_train)
### draw the decision boundary with the text points overlaid
prettyPicture(clf, features_test, labels_test)
output_image("test.png", "png", open("test.png", "rb").read())
print("******************************************************")
"""
        Evaluating the build Classifier 
        Metric used her is the Accuracy
        Accuracy = (Number of Correctly classified)/(Total number of samples)
"""
predicted_labels = clf.predict(features_test)

"""
    First method: calculate the number of correctly predicted labels
                  calculate the accuracy by dividing the number of 
                  correctly predicted labels by the total number of samples 
"""
x = predicted_labels == labels_test


accuracy = float(np.sum(x))/len(x) ## np.sum(x) will sum the number of Trues! (true = 1 and false = 0)

print ("Accuracy of this classifier is " + str(accuracy*100) + "%")

"""
    Second Method:  Accuracy classification score method provided 
                    by the package sklearn.metrics
    see http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
"""
from sklearn.metrics import accuracy_score
accuracy_SecondMethod = accuracy_score(predicted_labels, labels_test)

print("Accuracy provided by the second method using Sklearn: "+ str(accuracy_SecondMethod*100)+"%")










