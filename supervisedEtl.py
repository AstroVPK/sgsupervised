import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

import lsst.afw.table as afwTable
from lsst.pex.exceptions import LsstCppException

from sklearn.grid_search import GridSearchCV

import sgSVM as sgsvm

kargOutlier = {'g': {'lOffsetStar':-3.5, 'starDiff':4.0, 'lOffsetGal':-0.8, 'galDiff':3.9},
               'r': {'lOffsetStar':-2.9, 'starDiff':3.4, 'lOffsetGal':0.5, 'galDiff':2.3},
               'i': {'lOffsetStar':0.2, 'starDiff':0.5, 'lOffsetGal':1.7, 'galDiff':1.5},
               'z': {'lOffsetStar':1.0, 'starDiff':0.2, 'lOffsetGal':2.0, 'galDiff':1.4},
               'y': {'lOffsetStar':1.4, 'starDiff':0.2, 'lOffsetGal':2.6, 'galDiff':1.1},
              }

def dropMatchOutliers(cat, good=True, band='i', lOffsetStar=0.2, starDiff=0.3, lOffsetGal=2.0, galDiff=0.8):
    flux = cat.get('cmodel.flux.'+band)
    fluxZero = cat.get('flux.zeromag.'+band)
    mag = -2.5*np.log10(flux/fluxZero)
    noMeas = np.logical_not(np.isfinite(mag))
    magAuto = cat.get('mag.auto')
    try:
        stellar = cat.get('stellar')
    except KeyError:
        stellar = cat.get('mu.class') == 2
    goodStar = np.logical_or(noMeas, np.logical_and(good, np.logical_and(stellar, np.logical_and(mag < magAuto + starDiff, mag > magAuto - lOffsetStar - starDiff))))
    goodGal = np.logical_or(noMeas, np.logical_and(good, np.logical_and(np.logical_not(stellar), np.logical_and(mag < magAuto + galDiff, mag > magAuto - lOffsetGal - galDiff))))

    good = np.logical_or(goodStar, goodGal)

    return good

def getGood(cat, band='i', magCut=None, noParent=False, iBandCut=True):
    if not isinstance(cat, afwTable.tableLib.SourceCatalog) and\
       not isinstance(cat, afwTable.tableLib.SimpleCatalog):
        cat = afwTable.SourceCatalog.readFits(cat)
    flux = cat.get('cmodel.flux.'+band)
    fluxPsf = cat.get('flux.psf.'+band)
    ext = -2.5*np.log10(fluxPsf/flux)
    good = np.logical_and(True, ext < 5.0)
    if iBandCut:
        good = dropMatchOutliers(cat, good=good, band=band, **kargOutlier[band])
    if noParent:
        good = np.logical_and(good, cat.get('parent.'+band) == 0)
    if magCut is not None:
        good = np.logical_and(good, magI > magCut[0])
        good = np.logical_and(good, magI < magCut[1])
    return good

def getId(cat, band='i'):
    return cat.get('multId.'+band)

def getRa(cat, band='i'):
    return cat.get('coord.'+ band + '.ra')

def getDec(cat, band='i'):
    return cat.get('coord.'+ band + '.dec')

def getMag(cat, band='i'):
    f = cat.get('cmodel.flux.'+band)
    f0 = cat.get('flux.zeromag.'+band)
    mag = -2.5*np.log10(f/f0)
    return mag

def getMagErr(cat, band='i'):
    f = cat.get('cmodel.flux.'+band)
    fErr = cat.get('cmodel.flux.err.'+band)
    f0 = cat.get('flux.zeromag.'+band)
    f0Err = cat.get('flux.zeromag.err.'+band)
    rat = f/f0
    ratErr = np.sqrt(np.square(fErr) + np.square(f*f0Err/f0))/f0
    magErr = 2.5/np.log(10.0)*ratErr/rat
    return magErr

def getMagPsf(cat, band='i'):
    f = cat.get('flux.psf.'+band)
    f0 = cat.get('flux.zeromag.'+band)
    mag = -2.5*np.log10(f/f0)
    return mag

def getMagPsfErr(cat, band='i'):
    f = cat.get('flux.psf.'+band)
    fErr = cat.get('flux.psf.err.'+band)
    f0 = cat.get('flux.zeromag.'+band)
    f0Err = cat.get('flux.zeromag.err.'+band)
    rat = f/f0
    ratErr = np.sqrt(np.square(fErr) + np.square(f*f0Err/f0))/f0
    magErr = 2.5/np.log(10.0)*ratErr/rat
    return magErr

def getExt(cat, band='i'):
    f = cat.get('cmodel.flux.'+band)
    fP = cat.get('flux.psf.'+band)
    ext = -2.5*np.log10(fP/f)
    return ext

def getExtErr(cat, band='i', corr=0.3):
    f = cat.get('cmodel.flux.'+band)
    fErr = cat.get('cmodel.flux.err.'+band)
    fP = cat.get('flux.psf.'+band)
    fPErr = cat.get('flux.psf.err.'+band)
    rat = fP/f
    ratErr = np.sqrt(np.square(fPErr) + np.square(fP*fErr/f))/f
    extErr = 2.5/np.log(10.0)*ratErr/rat
    # For now, this is a very crude approximation, the above code is
    # just a placeholder
    extErr = 1.0/f*fErr
    return extErr

def getExtKron(cat, band='i'):
    f = cat.get('flux.kron.'+band)
    fP = cat.get('flux.psf.'+band)
    ext = -2.5*np.log10(fP/f)
    return ext

def getExtHsm(cat, band='i'):
    q, ext = sgsvm.getShape(cat, band, 'hsm')
    return ext

def getExtHsmDeconv(cat, band='i'):
    q, ext = sgsvm.getShape(cat, band, 'hsmDeconv')
    return ext

def getSnr(cat, band='i'):
    f = cat.get('cmodel.flux.'+band)
    fErr = cat.get('cmodel.flux.err.'+band)
    snr = f/fErr
    return snr
    
def getSnrPsf(cat, band='i'):
    f = cat.get('flux.psf.'+band)
    fErr = cat.get('flux.psf.err.'+band)
    snr = f/fErr
    return snr

def getSeeing(cat, band='i'):
    seeing = cat.get('seeing.'+band)
    return seeing

def getDGaussRadInner(cat, band='i'):
    return cat.get('dGauss.radInner.' + band)

def getDGaussRadOuter(cat, band='i'):
    return cat.get('dGauss.radOuter.' + band)

def getDGaussAmpRat(cat, band='i'):
    return cat.get('dGauss.ampRat.' + band)

def getDGaussQInner(cat, band='i'):
    return cat.get('dGauss.qInner.' + band)

def getDGaussQOuter(cat, band='i'):
    return cat.get('dGauss.qOuter.' + band)

def getDGaussThetaInner(cat, band='i'):
    return cat.get('dGauss.thetaInner.' + band)

def getDGaussThetaOuter(cat, band='i'):
    return cat.get('dGauss.thetaOuter.' + band)

def getStellar(cat):
    return cat.get('stellar')

def getMuClass(cat):
    # We get away with doing this because Alexie's catalog has 0 objects
    # with mu.class=3.
    return cat.get('mu.class') == 2

inputsDict = {'id' : getId,
              'ra' : getRa,
              'dec' : getDec,
              'mag' : getMag,
              'magPsf' : getMagPsf,
              'ext' : getExt,
              'extKron' : getExtKron,
              'extHsm' : getExtHsm,
              'extHsmDeconv' : getExtHsmDeconv,
              'snr' : getSnr,
              'snrPsf' : getSnrPsf,
              'seeing' : getSeeing,
              'dGaussRadInner' : getDGaussRadInner,
              'dGaussRadOuter' : getDGaussRadOuter,
              'dGaussAmpRat' : getDGaussAmpRat,
              'dGaussQInner' : getDGaussQInner,
              'dGaussQOuter' : getDGaussQOuter,
              'dGaussThetaInner' : getDGaussThetaInner,
              'dGaussThetaOuter' : getDGaussThetaOuter
              #'fluxRat' : getFluxRat,
              #'fluxPsfRat' : getFluxRatPsf
             }

inputsErrDict = {'mag' : getMagErr,
                 'magPsf' : getMagPsfErr,
                 'ext' : getExtErr
                 #'fluxRat' : getFluxRatErr,
                 #'fluxPsfRat' : getFluxPsfRatErr
                }

outputsDict = {'stellar' : getStellar,
               'mu.class' : getMuClass
              }

def getInputsList():
    return inputsDict.keys()

def getInputsErrList():
    return inputsErrDict.keys()

def getOutputsList():
    return outputsDict.keys()

def getInput(cat, inputName='mag', band='i'):
    """
    Get the input `inputName` from cat `cat` in band `band`. To see the list of valid inputs run
    `getInputsList()`.
    """
    return inputsDict[inputName](cat, band=band)

def getInputErr(cat, inputName='mag', band='i'):
    """
    Get the input's `inputName` error from cat `cat` in band `band`. To see the list of valid inputs run
    `getInputsErrList()`.
    """
    return inputsErrDict[inputName](cat, band=band)

def getOutput(cat, outputName='mu.class'):
    """
    Get the output `outputName` from cat `cat` in band `band`. To see the list of valid outputs run
    `getOutputsList()`.
    """
    return outputsDict[outputName](cat)

class TrainingSet(object):

    def __init__(self, X, Y, XErr=None, ids=None, ras=None, decs=None, mags=None, exts=None,
                 testFrac=0.2, polyOrder=1, bands=None, names=None):
        self.nTotal = len(X)
        self.X = X
        assert len(Y) == self.nTotal
        self.Y = Y
        if XErr is not None:
            assert len(XErr) == self.nTotal
            self.XErr = XErr
        if ids is not None:
            assert len(ids) == self.nTotal
            self.ids = ids
        if ras is not None:
            assert len(ras) == self.nTotal
            self.ras = ras
        if decs is not None:
            assert len(decs) == self.nTotal
            self.decs = decs
        if mags is not None:
            assert len(mags) == self.nTotal
            self.mags = mags
        if exts is not None:
            assert len(exts) == self.nTotal
            self.exts = exts
        if bands is not None:
            self.bands = bands
        if names is not None:
            assert self.X.shape[1] == sgsvm.nterms(polyOrder, len(names)) - 1
            self.names = names
        self.testFrac = testFrac
        self.nTest = int(testFrac*self.nTotal)
        self.nTrain = self.nTotal - self.nTest
        self.polyOrder = polyOrder
        self.selectTrainTest()

    def selectTrainTest(self, randomState=None):
        prng = np.random.RandomState()
        if randomState is None:
            self.randomState = prng.get_state()
        else:
            prng.set_state(randomState)
            self.randomState = randomState
        indexes = prng.choice(self.nTotal, self.nTotal, replace=False)
        self.trainIndexes = (indexes[:self.nTrain],)
        self.testIndexes = (indexes[self.nTrain:self.nTotal],)
        self._computeTransforms()

    def _computeTransforms(self):
        self.XmeanTrain = np.mean(self.X[self.trainIndexes], axis=0)
        self.XstdTrain = np.std(self.X[self.trainIndexes], axis=0)
        self.XmeanTest = np.mean(self.X[self.testIndexes], axis=0)
        self.XstdTest = np.std(self.X[self.testIndexes], axis=0)
        self.XmeanAll = np.mean(self.X, axis=0)
        self.XstdAll = np.std(self.X, axis=0)

    def getTrainSet(self, standardized=True):
        if standardized:
            return (self.X[self.trainIndexes] - self.XmeanTrain)/self.XstdTrain, self.Y[self.trainIndexes]
        else:
            if hasattr(self, 'XErr'):
                return self.X[self.trainIndexes], self.XErr[self.trainIndexes], self.Y[self.trainIndexes]
            else:
                return self.X[self.trainIndexes], self.Y[self.trainIndexes]

    def getTrainIds(self):
        return self.ids[self.trainIndexes]

    def getTrainRas(self):
        return self.ras[self.trainIndexes]

    def getTrainDecs(self):
        return self.decs[self.trainIndexes]

    def getTrainMags(self):
        return self.mags[self.trainIndexes]

    def getTrainExts(self):
        return self.exts[self.trainIndexes]

    def getTestSet(self, standardized=True):
        if standardized:
            return (self.X[self.testIndexes] - self.XmeanTrain)/self.XstdTrain, self.Y[self.testIndexes]
        else:
            if hasattr(self, 'XErr'):
                return self.X[self.testIndexes], self.XErr[self.testIndexes], self.Y[self.testIndexes]
            else:
                return self.X[self.testIndexes], self.Y[self.testIndexes]

    def getTestIds(self):
        return self.ids[self.testIndexes]

    def getTestRas(self):
        return self.ras[self.testIndexes]

    def getTestDecs(self):
        return self.decs[self.testIndexes]

    def getTestMags(self):
        return self.mags[self.testIndexes]

    def getTestExts(self):
        return self.exts[self.testIndexes]

    def getAllSet(self, standardized=True):
        if standardized:
            return (self.X - self.XmeanAll)/self.XstdAll, self.Y
        else:
            if hasattr(self, 'XErr'):
                return self.X, self.XErr, self.Y
            else:
                return self.X, self.Y

    def getAllIds(self):
        return self.ids

    def getAllRas(self):
        return self.ras

    def getAllDecs(self):
        return self.decs

    def getAllMags(self):
        return self.mags

    def getAllExts(self):
        return self.exts

    def applyPreTestTransform(self, X):
        return (X - self.XmeanTrain)/self.XstdTrain

    def applyPostTestTransform(self, X):
        return (X - self.XmeanAll)/self.XstdAll

    def plotLabeledHist(self, idx, physical=True, nBins=100):
        hist, bins = np.histogram(self.X[:,idx], bins=nBins)
        dataStars = self.X[:,idx][self.Y]
        dataGals = self.X[:,idx][np.logical_not(self.Y)]
        fig = plt.figure()
        plt.hist(dataStars, bins=bins, histtype='step', color='blue', label='Stars')
        plt.hist(dataGals, bins=bins, histtype='step', color='red', label='Galaxies')
        return fig

def _extractXY(cat, inputs=['ext'], output='mu.class', bands=['i'], magsType='cmodel', extsType='cmodel', concatBands=True,
              onlyFinite=True, polyOrder=1, withErr=False):
    """
    Load `inputs` from `cat` into `X` and   `output` to `Y`. If onlyFinite is True, then
    throw away all rows with one or more non-finite entries.
    """
    if not isinstance(cat, afwTable.tableLib.SourceCatalog) and\
       not isinstance(cat, afwTable.tableLib.SimpleCatalog):
        try:
            cat = afwTable.SourceCatalog.readFits(cat)
        except LsstCppException:
            cat = afwTable.SimpleCatalog.readFits(cat)
    nRecords = len(cat); nBands = len(bands); nInputs = len(inputs)
    if concatBands:
        ids = np.zeros((nRecords*len(bands), len(inputs)), dtype=int)
        ras = np.zeros((nRecords*len(bands), len(inputs)))
        decs = np.zeros((nRecords*len(bands), len(inputs)))
        X = np.zeros((nRecords*len(bands), len(inputs)))
        Y = np.zeros((nRecords*len(bands),), dtype=bool)
        mags = np.zeros((nRecords*len(bands),))
        exts = np.zeros((nRecords*len(bands),))
        if withErr:
            XErr = np.zeros((nRecords*len(bands), len(inputs)))
        for i, band in enumerate(bands):
            for j, inputName in enumerate(inputs):
                ids[i*nRecords:(i+1)*nRecords, j] = getInput(cat, inputName='id', band=band)
                ras[i*nRecords:(i+1)*nRecords, j] = getInput(cat, inputName='ra', band=band)
                decs[i*nRecords:(i+1)*nRecords, j] = getInput(cat, inputName='dec', band=band)
                X[i*nRecords:(i+1)*nRecords, j] = getInput(cat, inputName=inputName, band=band)
                if withErr:
                    XErr[i*nRecords:(i+1)*nRecords, j] = getInputErr(cat, inputName=inputName, band=band)
            Y[i*nRecords:(i+1)*nRecords] = getOutput(cat, outputName=output)
            mags[i*nRecords:(i+1)*nRecords] = getInput(cat, inputName='mag', band=band)
            exts[i*nRecords:(i+1)*nRecords] = getInput(cat, inputName='ext', band=band)
    else:
        ids = np.zeros((nRecords, len(inputs)*len(bands)), dtype=int)
        ras = np.zeros((nRecords, len(inputs)*len(bands)))
        decs = np.zeros((nRecords, len(inputs)*len(bands)))
        X = np.zeros((nRecords, len(inputs)*len(bands)))
        Y = np.zeros((nRecords,), dtype=bool)
        mags = np.zeros((nRecords,))
        exts = np.zeros((nRecords,))
        if withErr:
            XErr = np.zeros((nRecords, len(inputs)*len(bands)))
        for i, band in enumerate(bands):
            for j, inputName in enumerate(inputs):
                ids[:, i*nInputs + j] = getInput(cat, inputName='id', band=band)
                ras[:, i*nInputs + j] = getInput(cat, inputName='ra', band=band)
                decs[:, i*nInputs + j] = getInput(cat, inputName='dec', band=band)
                X[:, i*nInputs + j] = getInput(cat, inputName=inputName, band=band)
                if withErr:
                    XErr[:, i*nInputs + j] = getInputErr(cat, inputName=inputName, band=band)
        Y = getOutput(cat, outputName=output)
        mags = getInput(cat, inputName='mag', band='i')
        exts = getInput(cat, inputName='ext', band='i')
    if concatBands:
        good = np.ones((nRecords*len(bands),), dtype=bool)
    else:
        good = True
    for i, band in enumerate(bands):
        if concatBands:
            good[i*nRecords:(i+1)*nRecords] = np.logical_and(good[i*nRecords:(i+1)*nRecords], getGood(cat, band=band))
        else:
            good = np.logical_and(good, getGood(cat, band=band))
    if onlyFinite:
        for i in range(X.shape[1]):
            good = np.logical_and(good, np.isfinite(X[:,i]))
            if withErr:
                good = np.logical_and(good, np.isfinite(XErr[:,i]))
    ids = ids[good]; ras = ras[good]; decs = decs[good]; X = X[good]; Y = Y[good]; mags = mags[good]; exts = exts[good]
    if withErr:
        XErr = XErr[good]
    if polyOrder > 1:
        X = sgsvm.phiPol(X, polyOrder)
    if withErr:
        return X, XErr, Y, ids, ras, decs, mags, exts
    else:
        return X, Y, ids, ras, decs, mags, exts

def extractTrainSet(cat, mode='raw', extBand='i', **kargs):
    if not 'withErr' in kargs:
        kargs['withErr'] = False

    if mode in ['colors', 'rats'] and not kargs['withErr']:
        raise ValueError('You are forced to pull out errors for mode {0}, set keyword `withErr` to True'.format(mode))

    if kargs['withErr'] == True:
        X, XErr, Y, ids, ras, decs, mags, exts = _extractXY(cat, **kargs)
    else:
        X, Y, ids, ras, decs, mags, exts = _extractXY(cat, **kargs)

    if 'inputs' in kargs:
        inputs = kargs['inputs']
    else:
        inputs = ['ext']

    if 'concatBands' in kargs:
        concatBands = kargs['concatBands']
    else:
        concatBands = True

    if 'bands' in kargs:
        bands = kargs['bands']
    else:
        bands = ['i']

    if 'polyOrder' in kargs:
        polyOrder = kargs['polyOrder']
    else:
        polyOrder = 1

    if concatBands:
        names = inputs
    else:
        names = []
        for suffix in bands:
            for name in inputs:
                names.append(name + '_' + suffix)

    if mode == 'raw':
        if kargs['withErr'] == True:
            trainSet = TrainingSet(X, Y, XErr=XErr, ids=ids, ras=ras, decs=decs, mags=mags, exts=exts, polyOrder=polyOrder)
        else:
            trainSet = TrainingSet(X, Y, ids=ids, ras=ras, decs=decs, mags=mags, exts=exts, polyOrder=polyOrder)
    elif mode in ['colors', 'rats', 'colshape']:
        assert not concatBands
        assert len(bands) > 1
        cNames = []
        nColors = len(bands) - 1
        if mode == 'colors':
            XCol = np.zeros((X.shape[0], nColors))
            XColErr = np.zeros(XCol.shape)
            XColCov = np.zeros(XCol.shape + XCol.shape[-1:])
        elif mode == 'colshape':
            XColShape = np.zeros((X.shape[0], nColors + 1))
            XColShapeErr = np.zeros(XColShape.shape)
            XColShapeCov = np.zeros(XColShape.shape + XColShape.shape[-1:])
            XCol = XColShape[:, :-1]
            XColErr = XColShapeErr[:, :-1]
            XColCov = XColShapeCov[:, :-1, :-1]
        diag = np.arange(XCol.shape[-1]) # To fill variance terms
        offDiag = diag[:-1] + 1 # To fill covariance terms
        if mode in ['colors', 'colshape']:
            for i in range(nColors):
                cNames.append(bands[i] + '-' + bands[i+1])
                idxB = names.index('mag_'+bands[i])
                idxR = names.index('mag_'+bands[i+1])
                XCol[:, i] = X[:, idxB] - X[:, idxR]
                XColErr[:, i] = XErr[:,idxB]**2 + XErr[:,idxR]**2

            covStack = []
            for i in range(1, len(bands)-1):
                idx = names.index('mag_'+bands[i])
                cov = -XErr[:, idx]**2
                covStack.append(cov)

        covOffDiag = np.vstack(covStack).T

        XColCov[:, diag, diag] = XColErr
        XColCov[:,diag[:-1], offDiag] = covOffDiag
        XColCov[:,offDiag, diag[:-1]] = covOffDiag

        # TODO: Add code to be able to append more inputs given appropiate correlation coefficients
        if mode == 'colshape':
            idx = names.index('ext_'+extBand)
            XColShape[:, -1] = X[:, idx]
            XColShapeCov[:, -1, -1] = XErr[:, idx]

        names = cNames
        if mode == 'colors':
            trainSet = TrainingSet(XCol, Y, XErr=XColCov, ids=ids, ras=ras, decs=decs, mags=mags, exts=exts,\
                                   bands=bands, names=names, polyOrder=polyOrder)
        elif mode == 'colshape':
            names.append('ext_'+extBand)
            trainSet = TrainingSet(XColShape, Y, XErr=XColShapeCov, ids=ids, ras=ras, decs=decs, mags=mags, exts=exts,\
                                   bands=bands, names=names, polyOrder=polyOrder)
    else:
        raise ValueError("Mode {0} not implemented".format(mode))

    return trainSet

class Training(object):

    def __init__(self, trainingSet, clf, preFit=True):
        self.trainingSet = trainingSet
        self.clf = clf
        self.preFit = preFit
        self._computePhysicalFit()

    def predictTrainLabels(self):
        X = self.trainingSet.getTrainSet()[0]
        return self.clf.predict(X)

    def predictTestLabels(self):
        X = self.trainingSet.getTestSet()[0]
        return self.clf.predict(X)

    def predictAllLabels(self):
        X = self.trainingSet.getAllSet()[0]
        return self.clf.predict(X)

    def _computePhysicalFit(self):
        if isinstance(self.clf, GridSearchCV):
            estimator = self.clf.best_estimator_
        else:
            estimator = self.clf

        if self.preFit:
            self.coeffPhys = estimator.coef_[0]/self.trainingSet.XstdTrain
            self.interceptPhys = estimator.intercept_ -\
                        np.sum(estimator.coef_[0]*self.trainingSet.XmeanTrain/self.trainingSet.XstdTrain)
        else:
            self.coeffPhys = estimator.coef_[0]/self.trainingSet.XstdAll
            self.interceptPhys = estimator.intercept_ -\
                        np.sum(estimator.coef_[0]*self.trainingSet.XmeanAll/self.trainingSet.XstdAll)

    def printPhysicalFit(self, normalized=False):
        if normalized:
            print "Normalized coeff:", self.coeffPhys/self.coeffPhys[0]
            print "Normalized intercept:", self.interceptPhys/self.coeffPhys[0]
        else:
            print "coeff:", self.coeffPhys
            print "intercept:", self.interceptPhys

    def printPolynomial(self, names):
        polyOrder = self.trainingSet.polyOrder
        assert self.trainingSet.X.shape[1] == sgsvm.nterms(polyOrder, len(names)) - 1
        nTerms = len(names)

        polyStr = "{0} ".format(self.interceptPhys[0])

        count = 0
        for i in range(len(names)):
            if self.coeffPhys[i] < 0.0:
                polyStr += "{0}*{1} ".format(self.coeffPhys[i], names[i])
            else:
                polyStr += "+{0}*{1} ".format(self.coeffPhys[i], names[i])
            count += 1

        if polyOrder >= 2:
            for i in range(nTerms):
                for j in range(i, nTerms):
                    if self.coeffPhys[count] < 0.0:
                        polyStr += "{0}*{1}*{2} ".format(self.coeffPhys[count], names[i], names[j])
                    else:
                        polyStr += "+{0}*{1}*{2} ".format(self.coeffPhys[count], names[i], names[j])
                    count += 1

        if polyOrder >= 3:
            for i in range(nTerms):
                for j in range(i, nTerms):
                    for k in range(j, nTerms):
                        if self.coeffPhys[count] < 0.0:
                            polyStr += "{0}*{1}*{2}*{3} ".format(self.coeffPhys[count], names[i], names[j], names[k])
                        else:
                            polyStr += "+{0}*{1}*{2}*{3} ".format(self.coeffPhys[count], names[i], names[j], names[k])
                        count += 1

        print polyStr

    def plotScores(self, nBins=50, sType='test', fig=None, linestyle='-', fontSize=18,
                   magRange=None, xlabel='Magnitude', ylabel='Scores', legendLabel=''):
        if sType == 'test':
            mags = self.trainingSet.mags[self.trainingSet.testIndexes]
        elif sType == 'train':
            mags = self.trainingSet.mags[self.trainingSet.trainIndexes]
        elif sType == 'all':
            mags = self.trainingSet.mags
        else:
            raise ValueError("Scores of type {0} are not implemented".format(sType))
        
        if magRange is None:
            magsBins = np.linspace(mags.min(), mags.max(), num=nBins+1)
        else:
            magsBins = np.linspace(magRange[0], magRange[1], num=nBins+1)
        magsCenters = 0.5*(magsBins[:-1] + magsBins[1:])
        complStars = np.zeros(magsCenters.shape)
        purityStars = np.zeros(magsCenters.shape)
        complGals = np.zeros(magsCenters.shape)
        purityGals = np.zeros(magsCenters.shape)
        if sType == 'test':
            pred = self.predictTestLabels()
            truth = self.trainingSet.getTestSet()[1]
        elif sType == 'train':
            pred = self.predictTrainLabels()
            truth = self.trainingSet.getTrainSet()[1]
        elif sType == 'all':
            pred = self.predictAllLabels()
            truth = self.trainingSet.getAllSet()[1]
        for i in range(nBins):
            magCut = np.logical_and(mags > magsBins[i], mags < magsBins[i+1])
            predCut = pred[magCut]; truthCut = truth[magCut]
            goodStars = np.logical_and(predCut, truthCut)
            goodGals = np.logical_and(np.logical_not(predCut), np.logical_not(truthCut))
            if np.sum(truthCut) > 0:
                complStars[i] = float(np.sum(goodStars))/np.sum(truthCut)
            if np.sum(predCut) > 0:
                purityStars[i] = float(np.sum(goodStars))/np.sum(predCut)
            if len(truthCut) - np.sum(truthCut) > 0:
                complGals[i] = float(np.sum(goodGals))/(len(truthCut) - np.sum(truthCut))
            if len(predCut) - np.sum(predCut) > 0:
                purityGals[i] = float(np.sum(goodGals))/(len(predCut) - np.sum(predCut))

        if fig is None:
            fig = plt.figure()
            axGal = fig.add_subplot(1, 2, 1)
            axStar = fig.add_subplot(1, 2, 2)
            axGal.set_title('Galaxies', fontsize=fontSize)
            axStar.set_title('Stars', fontsize=fontSize)
            axGal.set_xlabel(xlabel, fontsize=fontSize)
            axGal.set_ylabel(ylabel, fontsize=fontSize)
            axStar.set_xlabel(xlabel, fontsize=fontSize)
            axStar.set_ylabel(ylabel, fontsize=fontSize)
            axGal.set_ylim((0.0, 1.0))
            axStar.set_ylim((0.0, 1.0))
        else:
            axGal, axStar = fig.get_axes()

        for ax in fig.get_axes():
            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(fontSize)
            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(fontSize)

        axGal.step(magsCenters, complGals, color='red', linestyle=linestyle, label=legendLabel + ' Completeness')
        axGal.step(magsCenters, purityGals, color='blue', linestyle=linestyle, label=legendLabel + ' Purity')
        axStar.step(magsCenters, complStars, color='red', linestyle=linestyle, label=legendLabel + ' Completeness')
        axStar.step(magsCenters, purityStars, color='blue', linestyle=linestyle, label=legendLabel + ' Purity')

        return fig

    def setPhysicalCut(self, cut, tType = 'train'):
        assert self.trainingSet.X.shape[1] == 1

        if isinstance(self.clf, GridSearchCV):
            estimator = self.clf.best_estimator_
        else:
            estimator = self.clf

        if tType == 'train':
            estimator.coef_[0][0] = -1.0*self.trainingSet.XstdTrain[0]
            estimator.intercept_[0] = cut - self.trainingSet.XmeanTrain[0]
            print "Cut in standardized data is at {0}".format(-estimator.intercept_[0]/estimator.coef_[0][0])
        elif tType == 'test':
            estimator.coef_[0][0] = -1.0*self.trainingSet.XstdTest[0]
            estimator.intercept_[0] = cut - self.trainingSet.XmeanTest[0]
            print "Cut in standardized data is at {0}".format(-estimator.intercept_[0]/estimator.coef_[0][0])
        elif tType == 'all':
            estimator.coef_[0][0] = -1.0*self.trainingSet.XstdAll[0]
            estimator.intercept_[0] = cut - self.trainingSet.XmeanAll[0]
            print "Cut in standardized data is at {0}".format(-estimator.intercept_[0]/estimator.coef_[0][0])
        else:
            raise ValueError("Transform of type {0} not implemented".format(tType))

    def _getF(self, fixedIndex, fixedVal, varIndex):
        nTerms = sgsvm.nterms(self.trainingSet.polyOrder, 2)
        vecRoot = np.zeros((2,))
        vecRoot[fixedIndex] = fixedVal
        vecCoeff = np.zeros((nTerms,))
        vecCoeff[:-1] = self.coeffPhys; vecCoeff[-1] = self.interceptPhys[0]
        vec = np.zeros((nTerms,))
        def F(x):
            vecRoot[varIndex] = x
            vec[0] = vecRoot[0]; vec[1] = vecRoot[1]; vec[-1] = 1.0
            count = 0
            if self.trainingSet.polyOrder >= 2:
                for i in range(2):
                    for j in range(i, 2):
                        vec[2 + count] = vecRoot[i]*vecRoot[j]
                        count += 1
            if self.trainingSet.polyOrder >= 3:
                for i in range(2):
                    for j in range(i, 2):
                        for k in range(j, 2):
                            vec[2 + count] = vecRoot[i]*vecRoot[j]*vecRoot[k]
                            count += 1
            if self.trainingSet.polyOrder >= 4:
                for i in range(2):
                    for j in range(i, 2):
                        for k in range(j, 2):
                            for l in range(k, 2):
                                vec[2 + count] = vecRoot[i]*vecRoot[j]*vecRoot[k]*vecRoot[l]
                                count += 1
            if self.trainingSet.polyOrder >= 5:
                for i in range(2):
                    for j in range(i, 2):
                        for k in range(j, 2):
                            for l in range(k, 2):
                                for m in range(l, 2):
                                    vec[2 + count] = vecRoot[i]*vecRoot[j]*vecRoot[k]*vecRoot[l]*vecRoot[m]
                                    count += 1
            if self.trainingSet.polyOrder >= 6:
                raise ValueError("Polynomials with order higher than 5 are not implemented.")

            return np.dot(vecCoeff, vec)
        return F

    def findZero(self, fixedIndex, fixedVal, zeroRange=None, chooseSol=1):
        assert self.trainingSet.X.shape[1] == sgsvm.nterms(self.trainingSet.polyOrder, 2) - 1
        assert chooseSol in [0, 1]

        if fixedIndex == 0:
            varIndex = 1
        elif fixedIndex == 1:
            varIndex = 0
        else:
            raise ValueError('I can only take indexes in [0, 1]')

        if zeroRange is None:
            zeroRange = (self.trainingSet.X[:,varIndex].min(),self.trainingSet.X[:,varIndex].max())

        if self.trainingSet.polyOrder == 1:
            sol = (-self.coeffPhys[fixedIndex]*fixedVal - self.interceptPhys[0])/self.coeffPhys[varIndex]
            if sol < zeroRange[0] or sol > zeroRange[1]:
                print "WARNING: Solution outside of range {0}".format(zeroRange)
            return sol
        elif self.trainingSet.polyOrder == 2:
            if varIndex == 0:
                A = self.coeffPhys[2]
                B = self.coeffPhys[0] + self.coeffPhys[3]*fixedVal
                C = self.coeffPhys[1]*fixedVal + self.coeffPhys[4]*fixedVal**2 + self.interceptPhys[0]
            elif varIndex == 1:
                A = self.coeffPhys[4]
                B = self.coeffPhys[1] + self.coeffPhys[3]*fixedVal
                C = self.coeffPhys[0]*fixedVal + self.coeffPhys[2]*fixedVal**2 + self.interceptPhys[0]
            discr = np.sqrt(B**2 - 4*A*C)
            sol1 = (-B + discr)/(2*A); sol2 = (-B - discr)/(2*A)
            if sol1 > zeroRange[0] and sol1 < zeroRange[1] and sol2 < zeroRange[0] or sol2 > zeroRange[1]:
                return sol1
            elif sol1 < zeroRange[0] or sol1 > zeroRange[1] and sol2 > zeroRange[0] and sol2 < zeroRange[1]:
                return sol2
            elif sol1 < zeroRange[0] or sol1 > zeroRange[1] and sol2 < zeroRange[0] or sol2 > zeroRange[1]:
                print "WARNING: No solution was found in the specified interval, I'll return the one that's closer"
                d1 = min(np.absolute(sol1 - zeroRange[0]), np.absolute(sol1 - zeroRange[1]))
                d2 = min(np.absolute(sol2 - zeroRange[0]), np.absolute(sol2 - zeroRange[1]))

                if d1 <= d2:
                    return sol1
                elif d2 < d1:
                    return sol2
            elif sol1 > zeroRange[0] and sol1 < zeroRange[1] and sol2 > zeroRange[0] and sol2 < zeroRange[1]:
                print "WARNING: Two solutions were found in the specified interval, I'll return solution {0}".format(chooseSol)
                if chooseSol == 1:
                    return sol1
                elif chooseSol == 2:
                    return sol2
        elif self.trainingSet.polyOrder > 2:
            F = self._getF(fixedIndex, fixedVal, varIndex)
            try:
                return brentq(F, zeroRange[0], zeroRange[1])
            except ValueError as e:
                figT = plt.figure()
                arr = np.linspace(zeroRange[0], zeroRange[1], num=100)
                ys = np.zeros(arr.shape)
                for i in range(len(arr)):
                    ys[i] = F(arr[i])
                plt.plot(arr, ys)
                plt.title("varRange={0}".format(fixedVal))
                plt.show()
                raise e

    def plotBoundary(self, rangeIndex, xRange=None, nPoints=100, fig=None, overPlotData=False,
                     xlim=None, ylim=None, xlabel=None, ylabel=None, yRange=None, frac=0.03,
                     withTrueLabels=True, fontSize=18, asLogX=False):
        assert self.trainingSet.X.shape[1] == sgsvm.nterms(self.trainingSet.polyOrder, 2) - 1

        if rangeIndex == 0:
            varIndex = 1
        elif rangeIndex == 1:
            varIndex = 0
        else:
            raise ValueError('I can only take indexes in [0, 1]')

        if xRange is None:
            xRange = (self.trainingSet.X[:,rangeIndex].min(),self.trainingSet.X[:,rangeIndex].max())

        if asLogX:
            xGrid = np.linspace(np.log10(xRange[0]), np.log10(xRange[1]), num=nPoints)
            xGrid = np.power(10.0, xGrid)
        else:
            xGrid = np.linspace(xRange[0], xRange[1], num=nPoints)
        yGrid = np.zeros(xGrid.shape)

        for i, fixedVal in enumerate(xGrid):
            yGrid[i] = self.findZero(rangeIndex, fixedVal, zeroRange=yRange)

        if fig is None:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
        else:
            ax = fig.get_axes()[0]

        if asLogX:
            plotting = ax.semilogx
        else:
            plotting = ax.plot

        if overPlotData:
            if withTrueLabels:
                nSample = int(frac*len(self.trainingSet.X))
                idxSample = np.random.choice(len(self.trainingSet.X), nSample, replace=False)
                for i in idxSample:
                    if self.trainingSet.Y[i]:
                        plotting(self.trainingSet.X[i,rangeIndex], self.trainingSet.X[i, varIndex], marker='.', markersize=1, color='blue')
                    else:
                        plotting(self.trainingSet.X[i,rangeIndex], self.trainingSet.X[i, varIndex], marker='.', markersize=1, color='red')
            else:
                Xtest, Ytest = self.trainingSet.getTestSet()
                testIndexes = self.trainingSet.testIndexes
                Z = self.clf.predict_proba(Xtest)[:,1]
                sc = ax.scatter(self.trainingSet.X[testIndexes, rangeIndex], self.trainingSet.X[testIndexes, varIndex], c=Z, marker='.', s=2, edgecolors="none")
                cb = fig.colorbar(sc)
                cb.set_label('P(Star)', fontsize=fontSize)
                cb.ax.tick_params(labelsize=fontSize)

        plotting(xGrid, yGrid, color='black')

        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        if xlabel is not None:
            ax.set_xlabel(xlabel, fontsize=fontSize)
        if ylabel is not None:
            ax.set_ylabel(ylabel, fontsize=fontSize)

        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontSize)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontSize)

        return fig

    def plotPMap(self, xRange, yRange, xN, yN, asLogX=False, fontSize=18, xlabel=None, ylabel=None, cbLabel=None):
        if asLogX:
            xx, yy = np.meshgrid(np.linspace(np.log10(xRange[0]), np.log10(xRange[1]), num=xN),
                                 np.linspace(yRange[0], yRange[1], num=yN))
            xx = np.power(10.0, xx)
        else:
            xx, yy = np.meshgrid(np.linspace(xRange[0], xRange[1], num=xN),
                                 np.linspace(yRange[0], yRange[1], num=yN))
        X = np.vstack((xx.flatten(), yy.flatten())).T
        if self.trainingSet.polyOrder > 1:
            X = sgsvm.phiPol(X, self.trainingSet.polyOrder)
        X = self.trainingSet.applyPreTestTransform(X)
        Z = self.clf.predict_proba(X)[:,1]
        zz = Z.reshape(xx.shape)
        fig = plt.figure()
        plt.pcolor(xx, yy, zz)
        cb = plt.colorbar()
        if xlabel is not None:
            plt.xlabel(xlabel, fontsize=fontSize)
        if ylabel is not None:
            plt.ylabel(ylabel, fontsize=fontSize)
        if cbLabel is not None:
            cb.set_label(cbLabel, fontsize=fontSize)
        ax = fig.get_axes()[0]
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontSize-2)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontSize-2)
        if asLogX:
            ax.set_xscale('log')
        ax.set_xlim(xRange)
        ax.set_ylim(yRange)
        return fig
