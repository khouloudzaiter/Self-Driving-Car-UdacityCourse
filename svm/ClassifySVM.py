# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 14:29:29 2017

@author: khouloud
"""
from sklearn import svm

def classify(features_train, labels_train):   
    ### import the sklearn module for GaussianNB
    

    ### create classifier
    clf = svm.SVC(C=10000000.0)
    ### fit the classifier on the training features and labels
    clf.fit(features_train, labels_train)
    ### return the fit classifier
    return clf
    
    ### your code goes here!
    
    
def SVMAccuracy(features_train, labels_train, features_test, labels_test):
    """ compute the accuracy of your Naive Bayes classifier """
    ### import the sklearn module for GaussianNB
    

    ### create classifier
    clf = svm.SVC() #TODO

    ### fit the classifier on the training features and labels
    #TODO
    clf.fit(features_train,labels_train)

    ### use the trained classifier to predict labels for the test features
    pred = clf.predict(features_test) #TODO


    ### calculate and return the accuracy on the test data
    ### this is slightly different than the example, 
    ### where we just print the accuracy
    ### you might need to import an sklearn module
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(pred, labels_test)

    return accuracy