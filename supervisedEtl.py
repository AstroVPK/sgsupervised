import numpy as np

import lsst.afw.table as afwTable
from lsst.pex.exceptions import LsstCppException

kargOutlier = {'g': {'lOffsetStar':-3.5, 'starDiff':3.9, 'lOffsetGal':-0.8, 'galDiff':3.7},
               'r': {'lOffsetStar':-2.2, 'starDiff':2.7, 'lOffsetGal':0.5, 'galDiff':2.3},
               'i': {'lOffsetStar':0.2, 'starDiff':0.3, 'lOffsetGal':2.0, 'galDiff':0.8},
               'z': {'lOffsetStar':1.0, 'starDiff':0.0, 'lOffsetGal':2.5, 'galDiff':0.8},
               'y': {'lOffsetStar':1.4, 'starDiff':0.0, 'lOffsetGal':2.5, 'galDiff':0.8},
              }

def getGood(cat, band='i', magCut=None, noParent=False, iBandCut=True,
            starDiff=1.0, galDiff=2.0, magAutoShift=0.0):
    if not isinstance(cat, afwTable.tableLib.SourceCatalog) and\
       not isinstance(cat, afwTable.tableLib.SimpleCatalog):
        cat = afwTable.SourceCatalog.readFits(cat)
    flux = cat.get('cmodel.flux.'+band)
    fluxPsf = cat.get('flux.psf.'+band)
    ext = -2.5*np.log10(fluxPsf/flux)
    good = np.logical_and(True, ext < 5.0)
    if iBandCut:
        for b in ['g', 'r', 'i', 'z', 'y']:
            good = dropMatchOutliers(cat, good=good, band=b, **kargOutlier[b])
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
    if onlyFinite:
        good = True
        for i in range(X.shape[1]):
            good = np.logical_and(good, np.isfinite(X[:,i]))
        X = X[good]; Y = Y[good]
    return X, Y

class TrainingSet(object):

    def __init__(X, Y, testFrac=0.2):
        self.X = X
        self.Y = Y

def transformXY(X, Y):
    """
    Standardizes the data to zero mean unit variance columns.
    """
    Xmean = np.mean(X, axis=0)
    Xstd = np.std(X, axis=0)

    Xtransform = (X[good] - Xmean
