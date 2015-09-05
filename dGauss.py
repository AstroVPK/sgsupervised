import numpy as np

from sklearn.svm import SVC, LinearSVC 
from sklearn.linear_model import LogisticRegression 
from sklearn.grid_search import GridSearchCV

def linearFit(trainingSet, n_jobs=4, magMin=None, magMax=None, **estKargs):
    X, Y, mags = trainingSet.getTrainSet()
    good = True
    if magMin is not None:
        good = np.logical_and(good, mags > magMin)
    if magMax is not None:
        good = np.logical_and(good, mags < magMax)
    if len(good) == len(X):
        X = X[good]
        Y = Y[good]
        mags = mags[good]
    estimator = LinearSVC(**estKargs)
    param_grid = {'C':[1.0, 10.0, 100.0, 1.0e3]}
    clf = GridSearchCV(estimator, param_grid, n_jobs=n_jobs)
    clf.fit(X, Y)
    X, Y, mags = trainingSet.getTestSet()
    good = True
    if magMin is not None:
        good = np.logical_and(good, mags > magMin)
    if magMax is not None:
        good = np.logical_and(good, mags < magMax)
    if len(good) == len(X):
        X = X[good]
        Y = Y[good]
        mags = mags[good]
    score = clf.score(X, Y)
    print "score=", score
    return clf

def logisticFit(trainingSet, n_jobs=4, magMin=None, magMax=None, **estKargs):
    X, Y, mags = trainingSet.getTrainSet()
    good = True
    if magMin is not None:
        good = np.logical_and(good, mags > magMin)
    if magMax is not None:
        good = np.logical_and(good, mags < magMax)
    if len(good) == len(X):
        X = X[good]
        Y = Y[good]
        mags = mags[good]
    estimator = LogisticRegression(**estKargs)
    param_grid = {'C':[ 1.0e5, 5.0e5, 1.0e6, 1.0e7]}
    clf = GridSearchCV(estimator, param_grid, n_jobs=n_jobs)
    clf.fit(X, Y)
    X, Y, mags = trainingSet.getTestSet()
    good = True
    if magMin is not None:
        good = np.logical_and(good, mags > magMin)
    if magMax is not None:
        good = np.logical_and(good, mags < magMax)
    if len(good) == len(X):
        X = X[good]
        Y = Y[good]
        mags = mags[good]
    score = clf.score(X, Y)
    print "score=", score
    return clf

def rbfFit(trainingSet, n_jobs=4, magMin=None, magMax=None):
    X, Y, mags = trainingSet.getTrainSet()
    good = True
    if magMin is not None:
        good = np.logical_and(good, mags > magMin)
    if magMax is not None:
        good = np.logical_and(good, mags < magMax)
    if len(good) == len(X):
        X = X[good]
        Y = Y[good]
        mags = mags[good]
    estimator = SVC()
    param_grid = {'C':[0.1, 1.0, 10.0], 'gamma':[0.1, 1.0, 10.0]}
    clf = GridSearchCV(estimator, param_grid, n_jobs=n_jobs)
    clf.fit(X, Y)
    X, Y, mags = trainingSet.getTestSet()
    good = True
    if magMin is not None:
        good = np.logical_and(good, mags > magMin)
    if magMax is not None:
        good = np.logical_and(good, mags < magMax)
    if len(good) == len(X):
        X = X[good]
        Y = Y[good]
        mags = mags[good]
    score = clf.score(X, Y)
    print "score=", score
    return clf
