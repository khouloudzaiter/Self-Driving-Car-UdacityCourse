# -*- coding: utf-8 -*-
"""
Created on Mon Oct 09 17:21:43 2017

@author: khouloud
"""

def classify(features_train, labels_train):   
    ### import the sklearn module for GaussianNB
    from sklearn.naive_bayes import GaussianNB

    ### create classifier
    clf = GaussianNB()
    ### fit the classifier on the training features and labels
    clf.fit(features_train, labels_train)
    ### return the fit classifier
    return clf
    
    ### your code goes here!
    