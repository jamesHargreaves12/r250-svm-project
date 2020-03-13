import logging
import os
import pickle
from collections import defaultdict

import matplotlib

import numpy as np
from scipy import sparse

import sklearn
from sklearn import datasets
from sklearn.svm import SVC

#TODO I think this should be improved to workout n and then reload the data because at the moment it wastes alot of time when it fails and is not a hard fix
# Alternatively we could just always build the vocab from the training set and the number of features in the est set cannot be greater.
# if OCCURENCE:
#     if PREPROCESSED:
#         raise ValueError("OCCUR + PREPROCESSED feature files do not exist yet")
#     X_train, y_train = sklearn.datasets.load_svmlight_file('aclImdb 2/train/labeledBowOccurence.feat')
#     X_test, y_test = sklearn.datasets.load_svmlight_file('aclImdb 2/test/labeledBowOccurence.feat',
#                                                          n_features=X_train.shape[1])
# else:
#     if PREPROCESSED:
#         n = sum(1 for line in open('aclImdb 2/imdb_james_preprocess_no_pos.vocab'))
#         X_train, y_train = sklearn.datasets.load_svmlight_file('aclImdb 2/train/labeledBowNoPos.feat', n_features=n)
#         X_test, y_test = sklearn.datasets.load_svmlight_file('aclImdb 2/test/labeledBowNoPos.feat', n_features=n)
#     else:
default_coef0 = 0
default_degree = 3


def get_trained_model(params, kernel, X_train, y_train):
    C = params[0]
    gamma = 'auto'
    degree = default_degree
    coef0 = default_coef0
    if len(params) >= 2:
        gamma = 10 ** params[1]
    if len(params) >= 3:
        coef0 = params[2]
    if len(params) >= 4:
        degree = int(params[3])

    model = SVC(C=10 ** C, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0, random_state=12345, cache_size=1024)
    model.fit(X_train, y_train)
    return model

def get_class_accuracies(pred_y, true_y):
    count = 0
    total_counts = defaultdict(int)
    total_correct = defaultdict(int)
    for pred, true in zip(pred_y, true_y):
        true_class = 1 if true > 5 else -1
        if true_class * pred > 0:
            count += 1
            total_correct[int(true)] += 1
        total_counts[int(true)] += 1
    overall = count / len(pred_y)
    logging.info("Overall Accuracy = ", )
    accuracy = {}
    for key in total_counts:
        accuracy[key] = total_correct[key] / total_counts[key]
    return overall,accuracy



if __name__ == "__main__":
    OCCURENCE = True
    PREPROCESSED = False

    X_train, y_train = sklearn.datasets.load_svmlight_file('aclImdb 2/train/labeledBowOccurence.feat')
    X_test, y_test = sklearn.datasets.load_svmlight_file('aclImdb 2/test/labeledBowOccurence.feat', n_features=X_train.shape[1])
    y_train = [1 if a > 5 else -1 for a in y_train]

    C = -1.08436863
    kernel = 'linear'
    model = get_trained_model([C], kernel, occurence=OCCURENCE, preprocessed=PREPROCESSED)
    pred_y = model.predict(X_test)
    print(get_class_accuracies(pred_y,y_test))
