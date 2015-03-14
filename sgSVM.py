import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import comb
from scipy.optimize import brentq

from sklearn import preprocessing
from sklearn.svm import SVC, LinearSVC 
from sklearn.linear_model import LogisticRegression 
from sklearn.grid_search import GridSearchCV

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

def getMags(cat, band, checkExtendedness=True, good=True, checkSNR=True, catType='hsc'):
    if catType == 'hsc':
        f = cat.get('cmodel.flux.'+band)
        fErr = cat.get('cmodel.flux.err.'+band)
        f0 = cat.get('flux.zeromag.'+band)
        fPsf = cat.get('flux.psf.'+band)
        ex = -2.5*np.log10(fPsf/f)
        rat = f/f0
        mag = -2.5*np.log10(rat)
        if checkSNR:
            good = np.logical_and(good, f/fErr > 5.0)
        if checkExtendedness:
            # Discard objects with extreme extendedness
            good = np.logical_and(good, ex < 5.0)
        if checkExtendedness or checkSNR:
            return mag, ex, good
    elif catType == 'sdss':
        mag = cat.get('cModelMag.'+band)
        magPsf = cat.get('psfMag.'+band)
        ex = magPsf-mag
    else:
        raise ValueError("Unkown catalog type {0}".format(catTYpe))
    return mag, ex

def loadData(catType='hsc', **kargs):
    if catType == 'hsc':
        return _loadDataHSC(**kargs)
    elif catType == 'sdss':
        return _loadDataSDSS(**kargs)
    else:
        raise ValueError("Unkown catalog type {0}".format(catType))

def _loadDataHSC(inputFile = "sgClassCosmosDeepCoaddSrcMultiBandAll.fits", withMags=True, withShape=True,
                 bands=['g', 'r', 'i', 'z', 'y'], doMagColors=True, magCut=None):
    if (not withMags) and (not withShape):
        raise ValueError("I need to use either shapes or magnitudes to train")
    if (not withMags) and (doMagColors):
        raise ValueError("I need to have magnitudes to do magnitude color mode")
    if (not withMags) and (magCut != None):
        raise ValueError("I need to have magnitudes to do magnitude cuts")
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
    if magCut != None:
        mag, ex, good = getMags(cat, 'r', good=good)
        good = np.logical_and(good, mag >= magCut[0])
        good = np.logical_and(good, mag <= magCut[1])
    X = X[good]; Y = Y[good]
    if doMagColors:
        magIdx = bands.index('r')
        Xtemp = X.copy()
        X[:,0] = Xtemp[:,magIdx] #TODO: Make it possible to use other bands seamlessly
        for i in range(1,len(bands)):
            X[:,i] = Xtemp[:,i-1] - Xtemp[:,i]
    return X, Y

def _loadDataSDSS(inputFile = "sgSDSS.fits", withMags=True, withShape=True,
                  bands=['u', 'g', 'r', 'i', 'z'], doMagColors=True, magCut=None):
    if (not withMags) and (not withShape):
        raise ValueError("I need to use either shapes or magnitudes to train")
    if (not withMags) and (doMagColors):
        raise ValueError("I need to have magnitudes to do magnitude color mode")
    if (not withMags) and (magCut != None):
        raise ValueError("I need to have magnitudes to do magnitude cuts")
    cat = afwTable.SimpleCatalog.readFits(inputFile)
    Y = cat.get('stellar')
    nBands = len(bands)
    shape = (len(cat), nBands*(int(withMags)+int(withShape)))
    X = np.zeros(shape)
    for i, b in enumerate(bands):
        mag, ex = getMags(cat, b, catType='sdss')
        if withMags:
            X[:, i] = mag
        if withShape:
            if withMags:
                X[:, nBands+i] = ex
            else:
                X[:, i] = ex
    if magCut != None:
        good = True
        mag, ex = getMags(cat, 'r', catType='sdss')
        good = np.logical_and(good, mag >= magCut[0])
        good = np.logical_and(good, mag <= magCut[1])
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
                doProb=False, probThreshold=0.5, bands=['g', 'r', 'i', 'z', 'y'],
                doMagColors=True, Y_predict=None):
    nBands = len(bands)
    nColors = nBands - 1

    if Y_predict is not None:
        print "I won't use the classifier that you passed, instead I'll used the predicted labels that you passed"
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
        if Y_predict is not None:
            Y_predict_cuts = Y_predict[idxs]
        if doMagColors:
            idxs = np.where(X_cuts[:,0] > mag - magWidth/2)
        else:
            idxs = np.where(X_cuts[:,1] > mag - magWidth/2)
        X_cuts = X_cuts[idxs]
        X_test_cuts = X_test_cuts[idxs]
        Y_test_cuts = Y_test_cuts[idxs]
        if Y_predict is not None:
            Y_predict_cuts = Y_predict_cuts[idxs]
        starIdxsTrue = np.where(Y_test_cuts == 1)
        galIdxsTrue = np.where(Y_test_cuts == 0)
        if Y_predict is None:
            Y_predict_cuts = clf.predict(X_test_cuts)
        starIdxsPredict = np.where(Y_predict_cuts == 1)
        galIdxsPredict = np.where(Y_predict_cuts == 0)
        if isinstance(clf, LogisticRegression) and doProb:
            cutProbs = clf.predict_proba(X_test_cuts)[:,1]
            Probs[i] = np.mean(cutProbs[starIdxsTrue])
            minIdxs = np.where(cutProbs[starIdxsTrue] < Probs[i])
            maxIdxs = np.where(cutProbs[starIdxsTrue] > Probs[i])
            ProbsMin[i] = np.mean(cutProbs[starIdxsTrue][minIdxs])
            ProbsMax[i] = np.mean(cutProbs[starIdxsTrue][maxIdxs])
            starIdxsPredict = np.where(cutProbs > probThreshold)
            galIdxsPredict = np.where(cutProbs <= probThreshold)
            Y_predict_cuts[starIdxsPredict] = 1
            Y_predict_cuts[galIdxsPredict] = 0

        nStarsTrue = np.sum(Y_test_cuts)
        nStarsCorrect = np.sum(Y_predict_cuts[starIdxsTrue])
        nStarsPredict = np.sum(Y_predict_cuts)
        nGalsTrue = len(Y_test_cuts) - nStarsTrue
        nGalsCorrect = len(galIdxsTrue[0]) - np.sum(Y_predict_cuts[galIdxsTrue])
        nGalsPredict = len(Y_predict_cuts) - nStarsPredict

        if nStarsTrue > 0:
            starCompl[i] = float(nStarsCorrect)/nStarsTrue
        if nStarsPredict > 0:
            starPurity[i] = float(nStarsCorrect)/nStarsPredict
        if nGalsTrue > 0:
            galCompl[i] = float(nGalsCorrect)/nGalsTrue
        if nGalsPredict > 0:
            galPurity[i] = float(nGalsCorrect)/nGalsPredict
    if doProb:
        return mags, starCompl, starPurity, galCompl, galPurity, Probs, ProbsMin, ProbsMax
    else:
        return mags, starCompl, starPurity, galCompl, galPurity

def plotMagCuts(clf=None, X_test=None, Y_test=None, X=None, fig=None, linestyle='-', mags=None,
                starCompl=None, starPurity=None, galCompl=None, Probs=None, ProbsMin=None,
                ProbsMax=None, galPurity=None, title='SVM Linear', **kargs):
    if 'doProb' in kargs:
        doProb = kargs['doProb']
    else:
        doProb = False
    if 'minMag' in kargs:
        minMag = kargs['minMag']
    else:
        minMag = 18.0
    if 'maxMag' in kargs:
        maxMag = kargs['maxMag']
    else:
        maxMag = 27.0
    if doProb:
        if mags == None or starCompl == None or starPurity == None or galCompl == None\
           or galPurity == None or Probs == None or ProbsMin == None or ProbsMax == None:
            mags, starCompl, starPurity, galCompl, galPurity, Probs, ProbsMin, ProbsMax = testMagCuts(clf, X_test, Y_test, X, **kargs)
    else:
        if mags == None or starCompl == None or starPurity == None or galCompl == None or galPurity == None:
            mags, starCompl, starPurity, galCompl, galPurity = testMagCuts(clf, X_test, Y_test, X, **kargs)
    if not fig:
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
        hadFig = False
    else:
        ax = fig.get_axes()[0]
        hadFig = True
    ax.plot(mags, starCompl, 'r', label='Completeness', linestyle=linestyle)
    ax.plot(mags, starPurity, 'b', label='Purity', linestyle=linestyle)
    if not hadFig:
        ax.legend(loc='lower left', fontsize=18)
    
    if not hadFig:
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
    else:
        ax = plt.subplot(1, 2, 1)
    ax.plot(mags, galCompl, 'r', label='Completeness', linestyle=linestyle)
    ax.plot(mags, galPurity, 'b', label='Purity', linestyle=linestyle)
    if not hadFig:
        ax.legend(loc='lower left', fontsize=18)
    
    if doProb:
        probs = clf.predict_proba(X_test)
        figProb, ax  = plt.subplots(1)
        plt.title("Logistic Regression", fontsize=18)
        plt.xlabel("Magnitude", fontsize=18)
        plt.ylabel("P(Star)", fontsize=18)
        ax.set_xlim(minMag, maxMag)
        ax.set_ylim(0.0, 1.0)
        ax.scatter(X[:, 0][np.logical_not(Y_test)], probs[:,1][np.logical_not(Y_test)], color='red', marker=".", s=3, label='Galaxies')
        ax.scatter(X[:, 0][Y_test], probs[:,1][Y_test], color='blue', marker=".", s=3, label='Stars')
        #ax.fill_between(mags, ProbsMin, ProbsMax, facecolor='grey', alpha=0.5)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(18)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(18)
        ax.legend(loc='lower left', fontsize=18)

    if doProb:
        return fig, figProb
    else:
        return fig

def plotDecFunc(clf, X, X_plot=None):
    assert X.shape[1] == 2
    if X_plot is not None:
        assert X_plot.shape == X.shape
    else:
        X_plot = X

    decFunc = clf.decision_function(X)

    fig = plt.figure()

    sc = plt.scatter(X_plot[:,0], X_plot[:,1], c=decFunc, marker="o", s=2, edgecolor="none")

    cb = plt.colorbar(sc, use_gridspec=True)

    return fig

def plotDecBdy(clf, mags, X=None, fig=None, Y=None, withScatter=False, linestyle='-', const=None):
    if X is None:
        magsStd = mags
        exMu = 0.0; exSigma = 1.0
    else:
        magMu = np.mean(X[:,0])
        magSigma = np.std(X[:,0])
        magsStd = (mags - magMu)/magSigma
        exMu = np.mean(X[:,1])
        exSigma = np.std(X[:,1])

    def F(ex, mag):
        ex = (ex-exMu)/exSigma
        if isinstance(ex, np.ndarray):
            mag = mag*np.ones(ex.shape) 
            X = np.vstack([mag, ex]).T
            return clf.decision_function(X)
        else:
            retval = clf.decision_function([mag, ex])[0]
            return retval

    exts = np.zeros(mags.shape)
    for i, mag in enumerate(magsStd):
        try:
            exts[i] = brentq(F, -0.2, 1.0, args=(mag,))
        except:
            print "mag=", mag*magSigma + magMu
            figT = plt.figure()
            arr = np.linspace(0.0, 5.0, num=100)
            plt.plot(arr, F(arr, mag))
            return figT

    if const is not None:
        exts = np.ones(exts.shape)*const
        linestyle = ':'
    if fig is None:
        fig = plt.figure()
        if withScatter and Y is not None:
            gals = np.logical_not(Y)
            plt.scatter(X[gals][:,0], X[gals][:,1], marker='.', s=1, color='red', label='Galaxies')
            plt.scatter(X[Y][:,0], X[Y][:,1], marker='.', s=1, color='blue', label='Stars')
        plt.plot(mags, exts, color='k', linestyle=linestyle, linewidth=2)
        ax = fig.get_axes()[0]
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(18)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(18)
        plt.xlabel('Magnitude HSC-R', fontsize=18)
        plt.ylabel('Extendedness HSC-R', fontsize=18)
        ax.legend(loc='upper right', fontsize=18)
    else:
        ax = fig.get_axes()[0]
        ax.plot(mags, exts, color='k', linestyle=linestyle, linewidth=2)

    return fig

def run(doMagColors=True, clfType='svc', param_grid={'C':[1.0, 10.0, 100.0, 1000.0], 'gamma':[0.1, 1.0, 10.0], 'kernel':['rbf']},
        magCut=None, doProb=False, inputFile = 'sgClassCosmosDeepCoaddSrcMultiBandAll.fits', catType='hsc', n_jobs=4, **kargs):
    X, Y = loadData(catType=catType, inputFile=inputFile, doMagColors=doMagColors, magCut=magCut, **kargs)
    trainIndexes, testIndexes = selectTrainTest(X)
    X_scaled = preprocessing.scale(X)
    X_train = X_scaled[trainIndexes]; Y_train = Y[trainIndexes]
    X_test = X_scaled[testIndexes]; Y_test = Y[testIndexes]
    estimator = getClassifier(clfType=clfType)
    clf = GridSearchCV(estimator, param_grid, n_jobs=n_jobs)
    clf.fit(X_train, Y_train)
    score = clf.score(X_test, Y_test)
    print "score=", score
    #plotMagCuts(clf, X_test, Y_test, X[testIndexes], title=clfType, doProb=doProb)
    print "The best estimator parameters are"
    print clf.best_params_
    #coef = clf.best_estimator_.coef_; intercept = clf.best_estimator_.intercept_
    #mu = np.mean(X, axis=0)
    #std = np.std(X, axis=0)
    #coef = coef/std
    #intercept = intercept - np.sum(coef*mu/std)
    #plt.show()
    #return clf, X_train, Y_train, X_test, Y_test, coef, intercept
    #return clf, X_train, Y_train, X_test, Y_test

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build the extreme deconvolution model..")
    parser.add_argument('--clfType', default='svc', type=str,
                        help='Type of classifier to use')
    parser.add_argument('--inputFile', default='sgClassCosmosDeepCoaddSrcMultiBandAll.fits', type=str,
                        help='File containing the input catalog')
    parser.add_argument('--catType', default='hsc', type=str,
                        help='If `hsc` assume the input file is an hsc catalog, `sdss` assume the input file is an sdss catalog.')
    kargs = vars(parser.parse_args())
    run(**kargs)
