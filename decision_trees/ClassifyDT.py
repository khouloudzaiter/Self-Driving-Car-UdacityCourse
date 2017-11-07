# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 12:21:44 2017

@author: khouloud
"""

def classify(features_train, labels_train):
    
    ### your code goes here--should return a trained decision tree classifer
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier(random_state=0, min_samples_split=50)
    clf.fit(features_train, labels_train)
  
    return clf