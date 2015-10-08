import numpy as np

from sklearn.svm import SVC, LinearSVC 
from sklearn.linear_model import LogisticRegression 
from sklearn.grid_search import GridSearchCV

import supervisedEtl as etl

def linearFit(trainingSet, n_jobs=4, magMin=None, magMax=None, **estKargs):
    X, Y = trainingSet.getTrainSet()
    mags = trainingSet.getTrainMags()
    good = True
    if magMin is not None:
        good = np.logical_and(good, mags > magMin)
    if magMax is not None:
        good = np.logical_and(good, mags < magMax)
    if isinstance(good, np.ndarray):
        X = X[good]
        Y = Y[good]
        mags = mags[good]
    estimator = LinearSVC(**estKargs)
    param_grid = {'C':[1.0, 10.0, 100.0, 1.0e3]}
    clf = GridSearchCV(estimator, param_grid, n_jobs=n_jobs)
    clf.fit(X, Y)
    X, Y = trainingSet.getTestSet()
    mags = trainingSet.getTestMags()
    good = True
    if magMin is not None:
        good = np.logical_and(good, mags > magMin)
    if magMax is not None:
        good = np.logical_and(good, mags < magMax)
    if isinstance(good, np.ndarray):
        X = X[good]
        Y = Y[good]
        mags = mags[good]
    score = clf.score(X, Y)
    print "score=", score
    return clf

def logisticFit(trainingSet, n_jobs=4, magMin=None, magMax=None, featuresCuts=None, mode='train', **estKargs):
    if mode == 'train':
        X, Y = trainingSet.getTrainSet()
        mags = trainingSet.getTrainMags()
    elif mode == 'all':
        X, Y = trainingSet.getAllSet()
        mags = trainingSet.getAllMags()
    good = True
    if magMin is not None:
        good = np.logical_and(good, mags > magMin)
    if magMax is not None:
        good = np.logical_and(good, mags < magMax)
    if featuresCuts is not None:
        for cutIdx in featuresCuts:
            cut = featuresCuts[cutIdx]
            if cut[0] is not None:
                if mode == 'train':
                    cutTransformed = (cut[0] - trainingSet.XmeanTrain[cutIdx])/trainingSet.XstdTrain[cutIdx]
                elif mode == 'all':
                    cutTransformed = (cut[0] - trainingSet.XmeanAll[cutIdx])/trainingSet.XstdAll[cutIdx]
                good = np.logical_and(good, X[:,cutIdx] > cutTransformed)
            if cut[1] is not None:
                if mode == 'train':
                    cutTransformed = (cut[1] - trainingSet.XmeanTrain[cutIdx])/trainingSet.XstdTrain[cutIdx]
                elif mode == 'all':
                    cutTransformed = (cut[1] - trainingSet.XmeanAll[cutIdx])/trainingSet.XstdAll[cutIdx]
                good = np.logical_and(good, X[:,cutIdx] < cutTransformed)
    if isinstance(good, np.ndarray):
        X = X[good]
        Y = Y[good]
        mags = mags[good]
    estimator = LogisticRegression(**estKargs)
    param_grid = {'C':[ 1.0e4, 1.0e5, 5.0e5, 1.0e6, 1.0e7]}
    clf = GridSearchCV(estimator, param_grid, n_jobs=n_jobs)
    shiftMask = np.zeros(Y.shape, dtype=int)
    shiftMask[Y == 0] = -1
    Yshifted = Y.copy()
    Yshifted = Y + shiftMask
    clf.fit(X, Yshifted)
    if mode == 'train':
        X, Y = trainingSet.getTestSet()
        mags = trainingSet.getTestMags()
        good = True
        if magMin is not None:
            good = np.logical_and(good, mags > magMin)
        if magMax is not None:
            good = np.logical_and(good, mags < magMax)
        if featuresCuts is not None:
            for cutIdx in featuresCuts:
                cut = featuresCuts[cutIdx]
                if cut[0] is not None:
                    cutTransformed = (cut[0] - trainingSet.XmeanTrain[cutIdx])/trainingSet.XstdTrain[cutIdx]
                    good = np.logical_and(good, X[:,cutIdx] > cutTransformed)
                if cut[1] is not None:
                    cutTransformed = (cut[1] - trainingSet.XmeanTrain[cutIdx])/trainingSet.XstdTrain[cutIdx]
                    good = np.logical_and(good, X[:,cutIdx] < cutTransformed)
        if isinstance(good, np.ndarray):
            X = X[good]
            Y = Y[good]
            mags = mags[good]
        shiftMask = np.zeros(Y.shape, dtype=int)
        shiftMask[Y == 0] = -1
        Yshifted = Y.copy()
        Yshifted = Y + shiftMask
        score = clf.score(X, Yshifted)
        print "score=", score
    return clf

def rbfFit(trainingSet, n_jobs=4, magMin=None, magMax=None):
    X, Y = trainingSet.getTrainSet()
    mags = trainingSet.getTrainMags()
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
    X, Y = trainingSet.getTestSet()
    mags = trainingSet.getTestMags()
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

def trainPieceWiseBand(cat, band='i', rTraceCut=2.0, snrCut=100.0, polyOrder=3):
    X, Y, mags = etl.extractXY(cat, inputs=['snrPsf', 'extHsmDeconv'], bands=[band], polyOrder=polyOrder)
    trainSet = etl.TrainingSet(X, Y, mags, polyOrder=polyOrder)
    clf = logisticFit(trainSet, n_jobs=2, featuresCuts={0:(None, snrCut), 1:(None, rTraceCut)})
    training = etl.Training(trainSet, clf)
    training.printPolynomial(['snr', 'rTrace'])
    fig = training.plotBoundary(0, xRange=(snrCut, 5.0), yRange=(-200.0, 20.0), overPlotData=True, ylim=(-5.0, 25.0), xlim=(500.0, 0.0),
    frac=0.04, xlabel='S/N PSF HSC-{0}'.format(band.upper()), ylabel=r'$\Delta(I_{xx}+I_{yy})$')
    ax = fig.get_axes()[0]; ax.set_ylabel(ax.get_ylabel() + ' HSC-{0}'.format(band.upper()))
    X, Y, mags = etl.extractXY(cat, inputs=['snrPsf', 'extHsmDeconv'], bands=[band], polyOrder=1)
    trainSet = etl.TrainingSet(X, Y, mags, polyOrder=1)
    clf = logisticFit(trainSet, n_jobs=2, featuresCuts={0:(snrCut, None)})
    training = etl.Training(trainSet, clf)
    training.printPolynomial(['snr', 'rTrace'])
    fig = training.plotBoundary(0, xRange=(500.0, snrCut), yRange=(-20.0, 20.0), overPlotData=False, ylim=(-5.0, 25.0), xlim=(500.0, 0.0), fig=fig)
    return fig
