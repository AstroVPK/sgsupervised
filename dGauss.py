import numpy as np
import pickle
import itertools
import multiprocessing
from copy import deepcopy

from sklearn.svm import SVC, LinearSVC 
from sklearn.linear_model import LogisticRegression 
from sklearn.grid_search import GridSearchCV
from sklearn.mixture import GMM
from astroML.density_estimation import XDGMM
from astroML.utils import logsumexp
import extreme_deconvolution as xd

import supervisedEtl as etl

def _xdInputs(trainSet, nGauss=None, mode=None, magMin=None, magMax=None, extMax=None,
              n_iter=10, persist=True):

    if nGauss is None:
        raise ValueError('You have to specify the number of Gaussians you want to use with keyword `nGauss`.')
    if mode is None or not mode in ['star', 'gal']:
        raise ValueError('You have to specify what class you are fitting a density for (`star`, or `gal`) with keyword `mode`.')

    X, XErr, Y = trainSet.getAllSet(standardized=False)
    mags = trainSet.getAllMags(); exts = trainSet.getAllExts()
    good = True
    if magMin is not None:
        good = np.logical_and(good, mags > magMin)
    if magMax is not None:
        good = np.logical_and(good, mags < magMax)
    if extMax is not None:
        good = np.logical_and(good, exts < extMax)
    if isinstance(good, np.ndarray):
        X = X[good]
        XErr = XErr[good]
        Y = Y[good]
        mags = mags[good]
        exts = mags[good]
    if mode == 'star':
        X = X[Y]
        XErr = XErr[Y]
        Y = Y[Y]
    elif mode == 'gal':
        X = X[np.logical_not(Y)]
        XErr = XErr[np.logical_not(Y)]
        Y = Y[np.logical_not(Y)]

    return X, XErr

def _xdFit(X, XErr, nGauss, n_iter=10):
    gmm = GMM(nGauss, n_iter=n_iter, covariance_type='full').fit(X)
    amp = gmm.weights_; mean = gmm.means_; covar = gmm.covars_
    xd.extreme_deconvolution(X, XErr, amp, mean, covar)
    clf = XDGMM(nGauss)
    clf.alpha = amp; clf.mu = mean; clf.V = covar
    return clf

def xdFit(trainSet, nGauss=None, mode=None, magMin=None, magMax=None, extMax=None,
          n_iter=10, persist=True):

    X, XErr = _xdInputs(trainSet, nGauss=nGauss, mode=mode, magMin=magMin, magMax=magMax, extMax=extMax,
                        n_iter=n_iter)

    clf = _xdFit(X, XErr, nGauss)

    if persist:
        if magMin is None:
            magMin = 'None'
        if magMax is None:
            magMax = 'None'
        if extMax is None:
            extMax = 'None'
        with open('xdFits/{0}NG{1}MagRange{2}-{3}ExtMax{4}.pkl'.format(mode, nGauss, magMin, magMax, extMax), 'wb') as f:
            pickle.dump(clf, f)

    return clf

def loadXDFits(ngStar=None, ngGal=None, magMin=None, magMax=None, extMax=None):
    if magMin is None:
        magMin = 'None'
    if magMax is None:
        magMax = 'None'
    if extMax is None:
        extMax = 'None'

    with open('xdFits/starNG{0}MagRange{1}-{2}ExtMax{3}.pkl'.format(ngStar, magMin, magMax, extMax), 'rb') as f:
        clfStar = pickle.load(f)
    with open('xdFits/galNG{0}MagRange{1}-{2}ExtMax{3}.pkl'.format(ngGal, magMin, magMax, extMax), 'rb') as f:
        clfGal = pickle.load(f)

    return clfStar, clfGal

def computeStellarPosteriors(X, XErr, ngStar=None, ngGal=None, priorStar=0.5, magMin=None, magMax=None, extMax=None):

    clfStar, clfGal = loadXDFits(ngStar=ngStar, ngGal=ngGal, magMin=magMin, magMax=magMax, extMax=extMax)

    logLStar = logsumexp(clfStar.logprob_a(X, XErr), -1)
    logLGal = logsumexp(clfGal.logprob_a(X, XErr), -1)
    logposteriorStar = logLStar + np.log(priorStar)
    logposteriorGal = logLGal + np.log(1.0 - priorStar)
    posteriorStar = np.exp(logposteriorStar)
    posteriorGal = np.exp(logposteriorGal)
    posteriorStar /= posteriorStar + posteriorGal

    return posteriorStar

class XDClf(object):

    def __init__(self, ngStar=4, ngGal=6, priorStar='auto'):
        self.ngStar = ngStar
        self.ngGal = ngGal
        self.priorStar = priorStar

    def fit(self, X, XErr, Y):
        self.clfStar = _xdFit(X[Y], XErr[Y], self.ngStar)
        self.clfGal = _xdFit(X[np.logical_not(Y)], XErr[np.logical_not(Y)], self.ngGal)
        if self.priorStar == 'auto':
            self._priorStar = np.sum(Y)*1.0/len(Y)

    def predict_proba(self, X, XErr, priorStar=None):
        if priorStar is not None:
            self._priorStar = priorStar
        logLStar = logsumexp(self.clfStar.logprob_a(X, XErr), -1)
        logLGal = logsumexp(self.clfGal.logprob_a(X, XErr), -1)
        logposteriorStar = logLStar + np.log(self._priorStar)
        logposteriorGal = logLGal + np.log(1.0 - self._priorStar)
        posteriorStar = np.exp(logposteriorStar)
        posteriorGal = np.exp(logposteriorGal)
        posteriorStar /= posteriorStar + posteriorGal
        return posteriorStar

    def predict(self, X, XErr):
        posteriorStar = self.predict_proba(X, XErr)
        pred = np.logical_not(posteriorStar < 0.5)
        return pred

    def score(self, X, XErr, Y):
        Ypred = self.predict(X, XErr)
        score = np.sum(Y == Ypred)*1.0/len(Y)
        return score

def _runCV(X, XErr, Y, paramList, nCV, outputQueue):
    scores = np.zeros((len(paramList), 3))
    nTrain = len(X)
    for i, param in enumerate(paramList):
        ngStar, ngGal = param
        scores[i, 0] = ngStar; scores[i, 1] = ngGal
        clf = XDClf(ngStar=ngStar, ngGal=ngGal)
        for j in range(nCV):
            idxs = np.arange(nTrain)
            np.random.shuffle(idxs)
            bdy = int(nTrain*2.0/3)
            training = idxs[:bdy]; validation = idxs[bdy:]
            clf.fit(X[training], XErr[training], Y[training])
            scores[i, 2] += clf.score(X[validation], XErr[validation], Y[validation])
        scores[i, 2] /= nCV
    outputQueue.put(scores)

def xdCV(trainSet, magMin=None, magMax=None, extMax=None, nCV=10,
         param_grid = {'ngStar':[4, 5], 'ngGal':[6, 7]},
         seed=1, n_jobs=1):
    np.random.seed(seed)
    X, XErr, Y = trainSet.getTrainSet(standardized=False)
    mags = trainSet.getTrainMags(); exts = trainSet.getTrainExts()
    good = True
    if magMin is not None:
        good = np.logical_and(good, mags > magMin)
    if magMax is not None:
        good = np.logical_and(good, mags < magMax)
    if extMax is not None:
        good = np.logical_and(good, exts < extMax)
    if isinstance(good, np.ndarray):
        X = X[good]
        XErr = XErr[good]
        Y = Y[good]
    paramList = list(itertools.product(param_grid['ngStar'], param_grid['ngGal']))
    if n_jobs > 1:
        outputQueue = multiprocessing.Queue()
        gridSize = len(paramList); chunkSize = gridSize/n_jobs
        remainder = gridSize - chunkSize*n_jobs
        procList = []
        scores = np.array([]).reshape((0, 3))
        for i in range(n_jobs):
            chunk = deepcopy(paramList[chunkSize*i:chunkSize*(i+1)])
            proc = multiprocessing.Process(target=_runCV, args=(deepcopy(X), deepcopy(XErr), deepcopy(Y), chunk, nCV, outputQueue))
            proc.start()
            procList.append(proc)
        if remainder > 0:
            chunk = deepcopy(paramList[chunkSize*n_jobs:])
            proc = multiprocessing.Process(target=_runCV, args=(deepcopy(X), deepcopy(XErr), deepcopy(Y), chunk, nCV, outputQueue))
            proc.start()
            procList.append(proc)
        for proc in procList:
            scoresProc = outputQueue.get()
            scores = np.vstack((scores, scoresProc))
        for proc in procList:
            proc.join()
    elif n_jobs == 1:
        nTrain = len(X)
        gridSize = len(paramList)
        scores = np.zeros((gridSize, 3))
        for i, (ngStar, ngGal) in enumerate(paramList):
            scores[i, 0] = ngStar; scores[i, 1] = ngGal
            clf = XDClf(ngStar=ngStar, ngGal=ngGal)
            for k in range(nCV):
                idxs = np.arange(nTrain)
                np.random.shuffle(idxs)
                bdy = int(nTrain*2.0/3)
                training = idxs[:bdy]; validation = idxs[bdy:]
                clf.fit(X[training], XErr[training], Y[training])
                scores[i, 2] += clf.score(X[validation], XErr[validation], Y[validation])
            scores[i, 2] /= nCV
    else:
        raise ValueError('`n_jobs` has to be a positive integer')
    return scores

def getCVParamsAndy(trainSet, magMin=18.0, magMax=22.0, extMax=0.4,
                   param_grid = {'ngStar':[5, 10, 15], 'ngGal':[5, 10, 15]}, nCV=10):
    scores = xdCV(trainSet, magMin=magMin, magMax=magMax, extMax=extMax, param_grid=param_grid, nCV=nCV)
    with open('cvRes/magRange{0}-{1}ExtMax{2}.txt'.format(magMin, magMax, extMax), 'w') as f:
        f.write("For mag range {0}--{1}:\n".format(magMin, magMax))
        f.write(scores[np.where(scores[:,2] == scores[:,2].max())][0].__repr__())
        f.write("\n")
        f.write(scores.__repr__())

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
