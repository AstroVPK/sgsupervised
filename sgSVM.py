import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import comb

from sklearn import preprocessing
from sklearn.svm import SVC, LinearSVC 
from sklearn.linear_model import LogisticRegression 

import lsst.afw.table as afwTable

def nterms(q, d):
    nterms = 1
    for n in range(1, q+1):
        for m in range(1, min(n, d) + 1):
            nterms += int(comb(n-1, m-1)*comb(d, m))
    return nterms

def phiPol(X, q):
    d = X.shape[-1]
    zDim = nterms(q, d)
    print "The model has {0} dimensions in Z space".format(zDim)
    Xz = np.zeros((X.shape[0], zDim-1)) # The intercept is not included in the input
    Xz[:,range(d)] = X
    count = 0
    if q >= 2:
        for i in range(d):
            for j in range(i, d):
                Xz[:,d + count] = X[:,i]*X[:,j]
                count += 1
    if q >= 3:
        for i in range(d):
            for j in range(i, d):
                for k in range(j, d):
                    Xz[:,d + count] = X[:,i]*X[:,j]*X[:,k]
                    count += 1
    if q >= 4:
        for i in range(d):
            for j in range(i, d):
                for k in range(j, d):
                    for l in range(k, d):
                        Xz[:,d + count] = X[:,i]*X[:,j]*X[:,k]*X[:,l]
                        count += 1
    return Xz

def plotMagEx(cat, band, withHSTLabels=True, magThreshold=23.5, exThreshold=0.04):
    mag, ex, good = getMags(cat, band)
    fig = plt.figure()
    if withHSTLabels:
        stellar = cat.get("stellar")
        stars = np.logical_and(good, stellar)
        gals = np.logical_and(good, np.logical_not(stellar))
        first = np.logical_or(mag >= magThreshold, ex > exThreshold)
        galsFirst = np.logical_and(gals, first)
        galsLater = np.logical_and(gals, np.logical_not(first))
        plt.scatter(mag[galsFirst], ex[galsFirst], marker='.', s=1, color='r', label='Galaxies')
        plt.scatter(mag[stars], ex[stars], marker='.', s=1, color='b', label='Stars')
        plt.scatter(mag[galsLater], ex[galsLater], marker='.', s=1, color='r')
    else:
        plt.scatter(mag[good], ex[good], marker='.', s=1)
    plt.xlabel('Magnitude HSC-'+band.upper(), fontsize=18)
    plt.ylabel('Extendedness HSC-'+band.upper(), fontsize=18)
    plt.xlim((mag[good].min(), mag[good].max()))
    plt.ylim((ex[good].min(), ex[good].max()))
    ax = fig.get_axes()[0]
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(18)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(18)
    plt.legend(loc=1, fontsize=18)
    return fig

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
             bands=['g', 'r', 'i', 'z', 'y'], doMagColors=True):
    if (not withMags) and (not withShape):
        raise ValueError("I need to use either shapes or magnitudes to train")
    if (not withMags) and (doMagColors):
        raise ValueError("I need to have magnitudes to do magnitude color mode")
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
            if withMags:
                X[:, nBands+i] = ex
            else:
                X[:, i] = ex
    X = X[good]; Y = Y[good]
    if doMagColors:
        magIdx = bands.index('r')
        Xtemp = X.copy()
        X[:,0] = Xtemp[:,magIdx] #TODO: Make it possible to use other bands seamlessly
        for i in range(1,len(bands)):
            X[:,i] = Xtemp[:,i-1] - Xtemp[:,i]
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
    elif clfType == 'linearsvc':
        return LinearSVC(*args, **kargs)
    elif clfType == 'logit':
        return LogisticRegression(*args, **kargs)
    else:
        raise ValueError("I don't know the classifier type {0}".format(clfType))

def testMagCuts(clf, X_test, Y_test, X, magWidth=1.0, minMag=18.0, maxMag=27.0, num=200,
                doProb=False, probThreshold=0.5, title='SVM Linear', bands=['g', 'r', 'i', 'z', 'y'],
                doMagColors=True):
    #import ipdb; ipdb.set_trace()
    nBands = len(bands)
    nColors = nBands - 1

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
        if doMagColors:
            idxs = np.where(X[:,0] < mag + magWidth/2)
        else:
            idxs = np.where(X[:,1] < mag + magWidth/2)
        X_cuts = X[idxs]
        X_test_cuts = X_test[idxs]
        Y_test_cuts = Y_test[idxs]
        if doMagColors:
            idxs = np.where(X_cuts[:,0] > mag - magWidth/2)
        else:
            idxs = np.where(X_cuts[:,1] > mag - magWidth/2)
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

    fig = plt.figure()
    
    ax = plt.subplot(1, 2, 0)
    ax.set_title(title + " (Stars)", fontsize=18)
    ax.set_xlabel("Mag Cut Center", fontsize=18)
    ax.set_ylabel("Star Scores", fontsize=18)
    ax.set_xlim(minMag, maxMag)
    ax.set_ylim(0.0, 1.0)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(18)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(18)
    ax.plot(mags, starCompl, 'r', label='Completeness')
    ax.plot(mags, starPurity, 'b', label='Purity')
    ax.legend(loc='lower left', fontsize=18)
    
    ax = plt.subplot(1, 2, 1)
    ax.set_title(title + " (Galaxies)", fontsize=18)
    ax.set_xlabel("Mag Cut Center", fontsize=18)
    ax.set_ylabel("Galaxy Scores", fontsize=18)
    ax.set_xlim(minMag, maxMag)
    ax.set_ylim(0.0, 1.0)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(18)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(18)
    ax.plot(mags, galCompl, 'r', label='Completeness')
    ax.plot(mags, galPurity, 'b', label='Purity')
    ax.legend(loc='lower left', fontsize=18)
    
    if doProb:
        fig, ax  = plt.subplots(1)
        plt.title("Predicted Stellar Probabilities for Real Stars", fontsize=18)
        plt.xlabel("MagCutsCenter", fontsize=18)
        plt.ylabel("P(Star)", fontsize=18)
        ax.plot(mags, Probs, 'k')
        ax.fill_between(mags, ProbsMin, ProbsMax, facecolor='grey', alpha=0.5, doMagColors=doMagColors)

    plt.show()

def run(doMagColors=True):
    X, Y = loadData(doMagColors=doMagColors)
    trainIndexes, testIndexes = selectTrainTest(X)
    X_scaled = preprocessing.scale(X)
    X_train = X_scaled[trainIndexes]; Y_train = Y[trainIndexes]
    X_test = X_scaled[testIndexes]; Y_test = Y[testIndexes]
    clf = getClassifier(clfType='svc', kernel='linear')
    clf.fit(X_train, Y_train)
    score = clf.score(X_test, Y_test)
    print "score=", score
    testMagCuts(clf, X_test, Y_test, X[testIndexes], title='SVM RBF', doProb=False)

if __name__ == '__main__':
    run()
