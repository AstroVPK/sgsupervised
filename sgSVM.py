import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.svm import SVC 
from sklearn.linear_model import LogisticRegression 

def loadData(inputFile = "/Users/josegarmilla/Data/HSC/sgClassCosmosDeepCoaddSrcMultiBandAll.pkl"):
    if os.path.isfile(inputFile):
        print "Loading data from {0}...".format(inputFile)
        with open(inputFile, 'rb') as f:
            xs = ()
            ys = ()
            while 1:
                try:
                    x, y = pickle.load(f)
                    xs += (x,); ys += (y,)
                except EOFError:
                    break
        X = np.concatenate(xs)
        Y = np.concatenate(ys)
        print "We have {0} training objects".format(len(X))
    else:
        print "Can't find file"
        sys.exit(0)
    return X, Y

def preprocessData(X, Y, withMags=True, withShape=True, standard=True):
    nBands = (len(X[0])-3)/3
    nColors = nBands - 1
    psfOffset = 3
    cmOffset = psfOffset + nBands
    exOffset = cmOffset + nBands
    
    if withMags and withShape:
        X = X[:,cmOffset:].astype('float')
    elif withMags:
        X = X[:,cmOffset:cmOffset+5].astype('float')
    elif withShape:
        X = X[:,exOffset:].astype('float')
    else:
        raise ValueError("I need to use either shapes or magnitudes to train")
    Y = Y.astype('int')
    if standard:
        X = preprocessing.scale(X)
    return X, Y
    
def selectTrainTest(X, nTrain = 20000, nTest = 27753):
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
    X_proc, Y_proc = preprocessData(X, Y)
    trainIndexes, testIndexes = selectTrainTest(X)
    X_train = X_proc[trainIndexes]; Y_train = Y_proc[trainIndexes]
    X_test = X_proc[testIndexes]; Y_test = Y_proc[testIndexes]
    clf = getClassifier(clfType='svc')
    clf.fit(X_train, Y_train)
    score = clf.score(X_test, Y_test)
    print "score=", score
    testMagCuts(clf, X_test, Y_test, X[testIndexes], title='SVM RBF', doProb=False)

if __name__ == '__main__':
    run()
