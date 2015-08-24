import numpy as np

import lsst.afw.table as afwTable
from lsst.pex.exceptions import LsstCppException

kargOutlier = {'g': {'lOffsetStar':-3.5, 'starDiff':3.9, 'lOffsetGal':-0.8, 'galDiff':3.7},
               'r': {'lOffsetStar':-2.2, 'starDiff':2.7, 'lOffsetGal':0.5, 'galDiff':2.3},
               'i': {'lOffsetStar':0.2, 'starDiff':0.3, 'lOffsetGal':2.0, 'galDiff':0.8},
               'z': {'lOffsetStar':1.0, 'starDiff':0.0, 'lOffsetGal':2.5, 'galDiff':0.8},
               'y': {'lOffsetStar':1.4, 'starDiff':0.0, 'lOffsetGal':2.5, 'galDiff':0.8},
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

def getMag(cat, band='i'):
    f = cat.get('cmodel.flux.'+band)
    f0 = cat.get('flux.zeromag.'+band)
    mag = -2.5*np.log10(f/f0)
    return mag

def getExt(cat, band='i'):
    f = cat.get('cmodel.flux.'+band)
    fP = cat.get('flux.psf.'+band)
    ext = -2.5*np.log10(fP/f)
    return ext

def getSnr(cat, band='i'):
    f = cat.get('cmodel.flux.'+band)
    fErr = cat.get('cmodel.flux.err.'+band)
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

inputsDict = {'mag' : getMag,
              'ext' : getExt,
              'snr' : getSnr,
              'seeing' : getSeeing,
              'dGaussRadInner' : getDGaussRadInner,
              'dGaussRadOuter' : getDGaussRadOuter,
              'dGaussAmpRat' : getDGaussAmpRat,
              'dGaussQInner' : getDGaussQInner,
              'dGaussQOuter' : getDGaussQOuter,
              'dGaussThetaInner' : getDGaussThetaInner,
              'dGaussThetaOuter' : getDGaussThetaOuter
             }

outputsDict = {'stellar' : getStellar,
               'mu.class' : getMuClass
              }

def getInputsList():
    return inputsDict.keys()

def getOutputsList():
    return outputsDict.keys()

def getInput(cat, inputName='mag', band='i'):
    """
    Get the input `inputName` from cat `cat` in band `band`. To see the list of valid inputs run
    `getInputsList()`.
    """
    return inputsDict[inputName](cat, band=band)

def getOutput(cat, outputName='mu.class'):
    """
    Get the output `outputName` from cat `cat` in band `band`. To see the list of valid outputs run
    `getOutputsList()`.
    """
    return outputsDict[outputName](cat)

def extractXY(cat, inputs=['ext'], output='mu.class', bands=['i'], concatBands=True,
              onlyFinite=True):
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
    nRecords = len(cat); nBands = len(bands)
    if concatBands:
        X = np.zeros((nRecords*len(bands), len(inputs)))
        Y = np.zeros((nRecords*len(bands),), dtype=int)
        for i, band in enumerate(bands):
            for j, inputName in enumerate(inputs):
                X[i*nRecords:(i+1)*nRecords, j] = getInput(cat, inputName=inputName, band=band)
            Y[i*nRecords:(i+1)*nRecords] = getOutput(cat, outputName=output)
    else:
        X = np.zeros((nRecords, len(inputs)*len(bands)))
        Y = np.zeros((nRecords,), dtype=int)
        for i, band in enumerate(bands):
            for j, inputName in enumerate(inputs):
                X[:, i*nBands + j] = getInput(cat, inputName=inputName, band=band)
        Y = getOutput(cat, outputName=output)
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
    X = X[good]; Y = Y[good]
    return X, Y

class TrainingSet(object):

    def __init__(self, X, Y, testFrac=0.2):
        self.X = X
        self.Y = Y
        self.nTotal = len(X)
        self.nTest = int(testFrac*self.nTotal)
        self.nTrain = self.nTotal - self.nTest
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
        self.XmeanPre = np.mean(self.X[self.trainIndexes], axis=0)
        self.XstdPre = np.std(self.X[self.trainIndexes], axis=0)
        self.XmeanPost = np.mean(self.X, axis=0)
        self.XstdPost = np.std(self.X, axis=0)

    def getPreTestTrainingSet(self, standardized=True):
        if standardized:
            return (self.X[self.trainIndexes] - self.XmeanPre)/self.XstdPre, self.Y[self.trainIndexes]
        else:
            return self.X[self.trainIndexes], self.Y[self.trainIndexes]

    def getTestSet(self, standardized=True):
        if standardized:
            return (self.X[self.testIndexes] - self.XmeanPre)/self.XstdPre, self.Y[self.testIndexes]
        else:
            return self.X[self.testIndexes], self.Y[self.testIndexes]

    def getPostTestTrainingSet(self, standardized=True):
        if standardized:
            return (self.X - self.XmeanPost)/self.XstdPost, self.Y
        else:
            return self.X, self.Y

    def applyPreTestTransform(self, X):
        return (X - self.XmeanPre)/self.XstdPre

    def applyPostTestTransform(self, X):
        return (X - self.XmeanPost)/self.XstdPost
