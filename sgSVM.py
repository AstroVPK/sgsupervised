import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.svm import SVC 
from sklearn.linear_model import LogisticRegression 

import lsst.afw.table as afwTable

def getMags(cat, band, checkExtendedness=True, good=True, checkSNR=True):
    f = cat.get('cmodel.flux.'+band)
    fErr = cat.get('cmodel.flux.err.'+band)
    f0 = cat.get('flux.zeromag.'+band)
    fPsf = cat.get('flux.psf.'+band)
    ex = -2.5*np.log10(fPsf/f)
    if checkExtendedness:
        # Discard objects with extreme extendedness
        good = np.logical_and(good, ex < 5.0)
    if checkSNR:
        good = np.logical_and(good, f/fErr > 5.0)
    rat = f/f0
    mag = -2.5*np.log10(rat)
    if checkExtendedness:
        return mag, ex, good
    return mag, ex

def loadData(inputFile = "sgClassCosmosDeepCoaddSrcMultiBandAll.fits", withMags=True, withShape=True,
             bands=['g', 'r', 'i', 'z', 'y'], standard=True):
    if (not withMags) and (not withShape):
        raise ValueError("I need to use either shapes or magnitudes to train")
    cat = afwTable.SourceCatalog.readFits(inputFile)
    Y = cat.get('stellar')
    nBands = len(bands)
    shape = (len(cat), nBands*(int(withMags)+int(withShape)))
    X = np.zeros(shape)
    good=True
    for i, b in enumerate(bands):
        mag, ex, good = getMags(cat, b, good=good)
        if withMags:
            good = np.logical_and(good, np.logical_not(np.isnan(mag)))
            good = np.logical_and(good, np.logical_not(np.isinf(mag)))
            X[:, i] = mag
        if withShape:
            good = np.logical_and(good, np.logical_not(np.isnan(ex)))
            good = np.logical_and(good, np.logical_not(np.isinf(ex)))
            X[:, nBands+i] = ex
    X = X[good]; Y = Y[good]
    if standard:
        X = preprocessing.scale(X)
    return X, Y

def selectTrainTest(X, nTrain = 0.8, nTest = 0.2):
    nTotal = len(X)
    nTrain = int(nTrain*nTotal)
    nTest = nTotal - nTrain
    indexes = np.random.choice(len(X), nTrain+nTest, replace=False)
    trainIndexes = (indexes[:nTrain],)
    testIndexes = (indexes[nTrain:nTrain+nTest],)
    return trainIndexes, testIndexes

def getClassifier(clfType = 'svc', *args, **kargs):
    if clfType == 'svc':
        return SVC(*args, **kargs)
    elif clfType == 'logit':
        return LogisticRegression(*args, **kargs)
    else:
        raise ValueError("I don't know the classifier type {0}".format(clfType))

def testMagCuts(clf, X_test, Y_test, X, magWidth=1.0, minMag=19.0, maxMag=26.0, num=200,
                doProb=False, probThreshold=0.5, title='SVM Linear'):
    #import ipdb; ipdb.set_trace()
    nBands = (len(X[0])-3)/3
    nColors = nBands - 1
    psfOffset = 3
    cmOffset = psfOffset + nBands
    exOffset = cmOffset + nBands

    mags = np.linspace(minMag, maxMag, num=num)
    starCompl = np.zeros(mags.shape)
    starPurity = np.zeros(mags.shape)
    galCompl = np.zeros(mags.shape)
    galPurity = np.zeros(mags.shape)
    if doProb:
        Probs = np.zeros(mags.shape)
        ProbsMin = np.zeros(mags.shape)
        ProbsMax = np.zeros(mags.shape)
    for i, mag in enumerate(mags):
        idxs = np.where(X[:,cmOffset+2] < mag + magWidth/2)
        X_cuts = X[idxs]
        X_test_cuts = X_test[idxs]
        Y_test_cuts = Y_test[idxs]
        idxs = np.where(X_cuts[:,cmOffset+2] > mag - magWidth/2)
        X_cuts = X_cuts[idxs]
        X_test_cuts = X_test_cuts[idxs]
        Y_test_cuts = Y_test_cuts[idxs]
        starIdxsTrue = np.where(Y_test_cuts == 1)
        galIdxsTrue = np.where(Y_test_cuts == 0)
        Y_predict = clf.predict(X_test_cuts)
        starIdxsPredict = np.where(Y_predict == 1)
        galIdxsPredict = np.where(Y_predict == 0)
        if isinstance(clf, LogisticRegression) and doProb:
            cutProbs = clf.predict_proba(X_test_cuts)[:,1]
            Probs[i] = np.mean(cutProbs[starIdxsTrue])
            minIdxs = np.where(cutProbs[starIdxsTrue] < Probs[i])
            maxIdxs = np.where(cutProbs[starIdxsTrue] > Probs[i])
            ProbsMin[i] = np.mean(cutProbs[starIdxsTrue][minIdxs])
            ProbsMax[i] = np.mean(cutProbs[starIdxsTrue][maxIdxs])
            starIdxsPredict = np.where(cutProbs > probThreshold)
            galIdxsPredict = np.where(cutProbs <= probThreshold)
            Y_predict[starIdxsPredict] = 1
            Y_predict[galIdxsPredict] = 0

        nStarsTrue = np.sum(Y_test_cuts)
        nStarsCorrect = np.sum(Y_predict[starIdxsTrue])
        nStarsPredict = np.sum(Y_predict)
        nGalsTrue = len(Y_test_cuts) - nStarsTrue
        nGalsCorrect = len(galIdxsTrue[0]) - np.sum(Y_predict[galIdxsTrue])
        nGalsPredict = len(Y_predict) - nStarsPredict

        if nStarsTrue > 0:
            starCompl[i] = float(nStarsCorrect)/nStarsTrue
        if nStarsPredict > 0:
            starPurity[i] = float(nStarsCorrect)/nStarsPredict
        if nGalsTrue > 0:
            galCompl[i] = float(nGalsCorrect)/nGalsTrue
        if nGalsPredict > 0:
            galPurity[i] = float(nGalsCorrect)/nGalsPredict

    plt.figure()
    plt.title(title + " (Stars)")
    plt.xlabel("MagCutsCenter")
    plt.ylabel("StarScores")
    plt.plot(mags, starCompl, 'r', label='Completeness')
    plt.plot(mags, starPurity, 'b', label='Purity')
    plt.legend(loc='lower left')
    
    plt.figure()
    plt.title(title + " (Galaxies)")
    plt.xlabel("MagCutsCenter")
    plt.ylabel("GalScores")
    plt.ylim(0.0, 1.0)
    plt.plot(mags, galCompl, 'r', label='Completeness')
    plt.plot(mags, galPurity, 'b', label='Purity')
    plt.legend(loc='lower left')
    
    if doProb:
        fig, ax  = plt.subplots(1)
        plt.title("Predicted Stellar Probabilities for Real Stars")
        plt.xlabel("MagCutsCenter")
        plt.ylabel("P(Star)")
        ax.plot(mags, Probs, 'k')
        ax.fill_between(mags, ProbsMin, ProbsMax, facecolor='grey', alpha=0.5)

    plt.show()

def run():
    X, Y = loadData()
    trainIndexes, testIndexes = selectTrainTest(X)
    X_train = X[trainIndexes]; Y_train = Y[trainIndexes]
    X_test = X[testIndexes]; Y_test = Y[testIndexes]
    clf = getClassifier(clfType='svc')
    clf.fit(X_train, Y_train)
    score = clf.score(X_test, Y_test)
    print "score=", score
    testMagCuts(clf, X_test, Y_test, X[testIndexes], title='SVM RBF', doProb=False)

if __name__ == '__main__':
    run()
