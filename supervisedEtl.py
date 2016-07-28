import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

import lsst.afw.table as afwTable
from lsst.pex.exceptions import LsstCppException

from sklearn.grid_search import GridSearchCV

import sgSVM as sgsvm

kargOutlier = {'g': {'lOffsetStar':-3.5, 'starDiff':4.0, 'lOffsetGal':-2.8, 'galDiff':4.9},
               'r': {'lOffsetStar':-2.9, 'starDiff':3.4, 'lOffsetGal':-2.5, 'galDiff':4.8},
               'i': {'lOffsetStar':-0.05, 'starDiff':0.58, 'lOffsetGal':-2.3, 'galDiff':4.5},
               'z': {'lOffsetStar':1.0, 'starDiff':0.2, 'lOffsetGal':-1.0, 'galDiff':3.9},
               'y': {'lOffsetStar':1.4, 'starDiff':0.2, 'lOffsetGal':-1.6, 'galDiff':4.6},
              }

def dropMatchOutliers(cat, good=True, band='i', lOffsetStar=0.2, starDiff=0.3, lOffsetGal=2.0, galDiff=0.8):
    try:
        flux = cat.get('cmodel.flux.'+band)
        fluxZero = cat.get('flux.zeromag.'+band)
        mag = -2.5*np.log10(flux/fluxZero)
    except KeyError:
        mag = cat.get(band+'mag')
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

def getGoodStats(cat, bands=['g', 'r', 'i', 'z', 'y']):
    if 'g' in bands:
        try:
            hasPhotG = np.isfinite(cat.get('cmodel.flux.g'))
        except KeyError:
            hasPhotG = np.isfinite(cat.get('gmag'))
    else:
        hasPhotG = np.zeros(len(cat), dtype=bool)
    if 'r' in bands:
        try:
            hasPhotR = np.isfinite(cat.get('cmodel.flux.r'))
        except:
            hasPhotR = np.isfinite(cat.get('rmag'))
    else:
        hasPhotR = np.zeros(len(cat), dtype=bool)
    if 'i' in bands:
        try:
            hasPhotI = np.isfinite(cat.get('cmodel.flux.i'))
        except KeyError:
            hasPhotI = np.isfinite(cat.get('imag'))
    else:
        hasPhotI = np.zeros(len(cat), dtype=bool)
    if 'z' in bands:
        try:
            hasPhotZ = np.isfinite(cat.get('cmodel.flux.z'))
        except KeyError:
            hasPhotZ = np.isfinite(cat.get('zmag'))
    else:
        hasPhotZ = np.zeros(len(cat), dtype=bool)
    if 'y' in bands:
        try:
            hasPhotY = np.isfinite(cat.get('cmodel.flux.y'))
        except KeyError:
            hasPhotY = np.isfinite(cat.get('ymag'))
    else:
        hasPhotY = np.zeros(len(cat), dtype=bool)
    hasPhotAny = np.logical_or(np.logical_or(np.logical_or(hasPhotG, hasPhotR), np.logical_or(hasPhotI, hasPhotZ)), hasPhotY)
    print "I removed {0} objects that don't have photometry in any band".format(len(hasPhotAny) - np.sum(hasPhotAny))
    good = hasPhotAny
    for band in bands:
        try:
            flux = cat.get('cmodel.flux.'+band)
            fluxPsf = cat.get('flux.psf.'+band)
            ext = -2.5*np.log10(fluxPsf/flux)
        except KeyError:
            ext = cat.get(band+'ext')
        noExtExt = np.logical_and(good, ext < 5.0)
        print "I removed {0} objects with extreme extendedness in {1}".format(np.sum(good)-np.sum(noExtExt), band)
        good = noExtExt
        noMatchOutlier = np.logical_and(good, dropMatchOutliers(cat, good=good, band=band, **kargOutlier[band]))
        print "I removed {0} match outliers in {1}".format(np.sum(good)-np.sum(noMatchOutlier), band)
        good = noMatchOutlier
    return good

def getGood(cat, band='i', magCut=None, noParent=False, iBandCut=True):
    #if not isinstance(cat, afwTable.tableLib.SourceCatalog) and\
    #   not isinstance(cat, afwTable.tableLib.SimpleCatalog):
    #    cat = afwTable.SourceCatalog.readFits(cat)
    #flux = cat.get('cmodel.flux.'+band)
    #fluxPsf = cat.get('flux.psf.'+band)
    #ext = -2.5*np.log10(fluxPsf/flux)
    #good = np.logical_and(True, ext < 5.0)
    #if iBandCut:
    #    good = dropMatchOutliers(cat, good=good, band=band, **kargOutlier[band])
    #if noParent:
    #    good = np.logical_and(good, cat.get('parent.'+band) == 0)
    #if magCut is not None:
    #    good = np.logical_and(good, magI > magCut[0])
    #    good = np.logical_and(good, magI < magCut[1])
    #return good
    if not isinstance(band, list):
        band = [band]
    return getGoodStats(cat, bands=band)

def getId(cat, band='i'):
    if band == 'fromDB':
        return cat.get('id.2')
    elif band == 'forced':
        return cat.get('id')
    else:
        return cat.get('multId.'+band)

def getRa(cat, band='i'):
    if band == 'fromDB':
        return cat.get('coord.ra')
    else:
        return cat.get('coord.'+ band + '.ra')

def getDec(cat, band='i'):
    if band == 'fromDB':
        return cat.get('coord.dec')
    else:
        return cat.get('coord.'+ band + '.dec')

def getSeeing(cat, band='i'):
    return cat.get('seeing.'+ band)

def getMag(cat, band='i'):
    try:
        f = cat.get('cmodel.flux.'+band)
        f0 = cat.get('flux.zeromag.'+band)
        mag = -2.5*np.log10(f/f0)
    except KeyError:
        mag = cat.get(band+'mag')
    return mag

def getMagErr(cat, band='i'):
    try:
        f = cat.get('cmodel.flux.'+band)
        fErr = cat.get('cmodel.flux.err.'+band)
        f0 = cat.get('flux.zeromag.'+band)
        f0Err = cat.get('flux.zeromag.err.'+band)
        rat = f/f0
        ratErr = np.sqrt(np.square(fErr) + np.square(f*f0Err/f0))/f0
        magErr = 2.5/np.log(10.0)*ratErr/rat
    except KeyError:
        try:
            magErr = cat.get(band+'mag.cmodel.err')
        except KeyError:
            magErr = cat.get(band+'cmodel.mag.err')
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
    try:
        f = cat.get('cmodel.flux.'+band)
        fP = cat.get('flux.psf.'+band)
        ext = -2.5*np.log10(fP/f)
    except KeyError:
        ext = cat.get(band+'ext')
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

def getExtHsmDeconvNorm(cat, band='i'):
    q, ext = sgsvm.getShape(cat, band, 'hsmDeconv', deconvType='traceNorm')
    return ext

def getExtHsmDeconvLinear(cat, band='i'):
    q, ext = sgsvm.getShape(cat, band, 'hsmDeconvLinear')
    return ext

def getSnr(cat, band='i'):
    try:
        f = cat.get('cmodel.flux.'+band)
        fErr = cat.get('cmodel.flux.err.'+band)
        snr = f/fErr
    except KeyError:
        try:
            snr = 2.5/np.log(10.0)/cat.get(band+'mag.cmodel.err')
        except KeyError:
            snr = 2.5/np.log(10.0)/cat.get(band+'cmodel.mag.err')
    return snr
    
def getSnrPsf(cat, band='i'):
    try:
        f = cat.get('flux.psf.'+band)
        fErr = cat.get('flux.psf.err.'+band)
        snr = f/fErr
    except KeyError:
        snr = 2.5/np.log(10.0)/cat.get(band+'mag.psf.err')
    return snr

def getSnrAp(cat, band='i'):
    f = cat.get('flux.sinc.'+band)
    fErr = cat.get('flux.sinc.err.'+band)
    snr = f/fErr
    return snr

def getSeeing(cat, band='i'):
    seeing = cat.get('seeing.'+band)
    return seeing

def getDGaussRadInner(cat, band='i'):
    return cat.get('dGauss.radInner.' + band)

def getDGaussRadOuter(cat, band='i'):
    return cat.get('dGauss.radOuter.' + band)

def getDGaussRadRat(cat, band='i'):
    return cat.get('dGauss.radOuter.' + band)/cat.get('dGauss.radInner.' + band)

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
              'seeing' : getSeeing,
              'mag' : getMag,
              'magPsf' : getMagPsf,
              'ext' : getExt,
              'extKron' : getExtKron,
              'extHsm' : getExtHsm,
              'extHsmDeconv' : getExtHsmDeconv,
              'extHsmDeconvNorm' : getExtHsmDeconvNorm,
              'extHsmDeconvLinear' : getExtHsmDeconvLinear,
              'snr' : getSnr,
              'snrPsf' : getSnrPsf,
              'snrAp' : getSnrAp,
              'seeing' : getSeeing,
              'dGaussRadInner' : getDGaussRadInner,
              'dGaussRadOuter' : getDGaussRadOuter,
              'dGaussRadRat' : getDGaussRadRat,
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
                 snrs=None, seeings=None, testFrac=0.2, polyOrder=1, bands=None, names=None):
        self.nTotal = len(X)
        self.X = X
        assert len(Y) == self.nTotal
        self.Y = Y
        #TODO: Implement this to always return views of X ommiting columns
        #self.include = np.ones((X.shape[1],), dtype=bool)
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
        if snrs is not None:
            self.snrs = snrs
        if seeings is not None:
            self.seeings = seeings
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

    def setTestFrac(self, value):
        if value < 0.0 or value > 1.0:
            raise ValueError("The test fraction is a number between 0 and 1.")
        self.testFrac = value
        self.nTest = int(self.testFrac*self.nTotal)
        self.nTrain = self.nTotal - self.nTest
        self.selectTrainTest()

    def _computeTransforms(self):
        self.XmeanTrain = np.mean(self.X[self.trainIndexes], axis=0)
        self.XstdTrain = np.std(self.X[self.trainIndexes], axis=0)
        self.XmeanTest = np.mean(self.X[self.testIndexes], axis=0)
        self.XstdTest = np.std(self.X[self.testIndexes], axis=0)
        self.XmeanAll = np.mean(self.X, axis=0)
        self.XstdAll = np.std(self.X, axis=0)

    def genTrainSubset(self, cuts=None, cols=None):
        good = np.ones(self.Y.shape, dtype=bool)
        kargsSub = {}
        if cuts is not None:
            assert isinstance(cuts, dict)
            for idx in cuts:
                if cuts[idx][0] is not None:
                    good = np.logical_and(good, self.X[:,idx] > cuts[idx][0])
                if cuts[idx][1] is not None:
                    good = np.logical_and(good, self.X[:,idx] < cuts[idx][1])
        if cols is not None:
            assert isinstance(cols, list)
            Xsub = self.X[:, cols][good]
            if hasattr(self, 'XErr'):
                kargsSub['XErr'] = self.XErr[:, np.ix_(cols, cols)][good]
            if hasattr(self, 'names'):
                kargsSub['names'] = []
                for i in cols:
                    kargsSub['names'].append(self.names[i])
            if hasattr(self, 'bands'):
                if len(self.bands) == 1:
                    kargsSub['bands'] = self.bands
                else:
                    bandsSet = set()
                    for name in kargsSub['names']:
                        if name[-1] in ['g', 'r', 'i', 'z', 'y']:
                            bandsSet.add(name[-1])
                    kargsSub['bands'] = list(bandsSet)
        else:
            Xsub = self.X[good]
            if hasattr(self, 'XErr'):
                kargsSub['XErr'] = self.XErr[good]
            if hasattr(self, 'names'):
                kargsSub['names'] = self.names
            if hasattr(self, 'bands'):
                kargsSub['bands'] = self.bands
        Ysub = self.Y[good]
        if hasattr(self, 'ids'):
            kargsSub['ids'] = self.ids[good]
        if hasattr(self, 'ras'):
            kargsSub['ras'] = self.ras[good]
        if hasattr(self, 'decs'):
            kargsSub['decs'] = self.decs[good]
        if hasattr(self, 'mags'):
            kargsSub['mags'] = self.mags[good]
        if hasattr(self, 'exts'):
            kargsSub['exts'] = self.exts[good]
        if hasattr(self, 'snrs'):
            kargsSub['snrs'] = self.snrs[good]
        if hasattr(self, 'seeings'):
            kargsSub['seeings'] = self.seeings[good]
        return TrainingSet(Xsub, Ysub, **kargsSub)

    def getConcatSet(self, keep=[]):
        """
        Merge columns that have suffixes, except those in `keep` (if `keep` is not empty)
        """

        if hasattr(self, 'XErr'):
            raise RuntimeError("I don't know how to handle covariance matrices during concatenation")

        namesMerge = []
        namesConcat = []
        for n in self.names:
            if n[-1] in self.bands and n[-2] == '_' and not n[:-2] in keep:
                if not n[:-2] in namesMerge:
                    namesMerge.append(n[:-2])
                    namesConcat.append(n[:-2])
            else:
                namesConcat.append(n)

        shapeXConcat = (self.X.shape[0]*len(self.bands), self.X.shape[1] - len(namesMerge)*(len(self.bands) - 1))
        XConcat = np.zeros(shapeXConcat)
        YConcat = np.zeros((self.X.shape[0]*len(self.bands),), dtype=bool)
        for i, n in enumerate(namesConcat):
            for j, b in enumerate(self.bands):
                if n in namesMerge:
                    idx = self.names.index(n + '_' + b)
                else:
                    idx = self.names.index(n)
                XConcat[j*self.X.shape[0]:(j+1)*self.X.shape[0], i] = self.X[:, idx]
        for i, b in enumerate(self.bands):
            YConcat[i*self.X.shape[0]:(i+1)*self.X.shape[0]] = self.Y

        kargsConcat = {}
        kargsConcat['names'] = namesConcat
        kargsConcat['bands'] = self.bands
        if hasattr(self, 'ids'):
            ids = np.zeros((self.X.shape[0]*len(self.bands), self.ids.shape[1]), dtype=int)
            for i, b in enumerate(self.bands):
                ids[i*self.X.shape[0]:(i+1)*self.X.shape[0]] = self.ids
            kargsConcat['ids'] = ids
        if hasattr(self, 'ras'):
            ras = np.zeros((self.X.shape[0]*len(self.bands), self.ras.shape[1]))
            for i, b in enumerate(self.bands):
                ras[i*self.X.shape[0]:(i+1)*self.X.shape[0]] = self.ras
            kargsConcat['ras'] = ras
        if hasattr(self, 'decs'):
            decs = np.zeros((self.X.shape[0]*len(self.bands), self.decs.shape[1]))
            for i, b in enumerate(self.bands):
                decs[i*self.X.shape[0]:(i+1)*self.X.shape[0]] = self.decs
            kargsConcat['decs'] = decs
        if hasattr(self, 'mags'):
            mags = np.zeros((self.X.shape[0]*len(self.bands),))
            for i, b in enumerate(self.bands):
                mags[i*self.X.shape[0]:(i+1)*self.X.shape[0]] = self.mags
            kargsConcat['mags'] = mags
        if hasattr(self, 'exts'):
            exts = np.zeros((self.X.shape[0]*len(self.bands),))
            for i, b in enumerate(self.bands):
                exts[i*self.X.shape[0]:(i+1)*self.X.shape[0]] = self.exts
            kargsConcat['exts'] = exts
        if hasattr(self, 'snrs'):
            snrs = np.zeros((self.X.shape[0]*len(self.bands),))
            for i, b in enumerate(self.bands):
                snrs[i*self.X.shape[0]:(i+1)*self.X.shape[0]] = self.snrs
            kargsConcat['snrs'] = snrs
        if hasattr(self, 'seeings'):
            seeings = np.zeros((self.X.shape[0]*len(self.bands),))
            for i, b in enumerate(self.bands):
                seeings[i*self.X.shape[0]:(i+1)*self.X.shape[0]] = self.seeings
            kargsConcat['seeings'] = seeings
        return TrainingSet(XConcat, YConcat, **kargsConcat)

    def getTrainSet(self, standardized=True):
        if standardized:
            if hasattr(self, 'XErr'):
                return (self.X[self.trainIndexes] - self.XmeanTrain)/self.XstdTrain, self.XErr[self.trainIndexes], self.Y[self.trainIndexes]
            else:
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

    def getTrainMags(self, band=None):
        if band is None:
            return self.mags[self.trainIndexes]
        else:
            return self.mags[:, self.bands.index(band)][self.trainIndexes]

    def getTrainExts(self, band=None):
        if band is None:
            return self.exts[self.trainIndexes]
        else:
            if band == 'best':
                idxBest = np.argmax(self.snrs[self.trainIndexes], axis=1)
                idxArr = np.arange(self.nTrain)
                return self.exts[self.trainIndexes][idxArr, idxBest], 1.0/self.snrs[self.trainIndexes][idxArr, idxBest]
            else:
                return self.exts[:, self.bands.index(band)][self.trainIndexes]

    def getTestSet(self, standardized=True):
        if standardized:
            if hasattr(self, 'XErr'):
                return (self.X[self.testIndexes] - self.XmeanTrain)/self.XstdTrain, self.XErr[self.testIndexes], self.Y[self.testIndexes]
            else:
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

    def getTestMags(self, band=None):
        if band is None:
            return self.mags[self.testIndexes]
        else:
            return self.mags[:, self.bands.index(band)][self.testIndexes]

    def getTestExts(self, band=None):
        if band is None:
            return self.exts[self.testIndexes]
        else:
            if band == 'best':
                idxBest = np.argmax(self.snrs[self.testIndexes], axis=1)
                idxArr = np.arange(self.nTest)
                return self.exts[self.testIndexes][idxArr, idxBest], 1.0/self.snrs[self.testIndexes][idxArr, idxBest]
            else:
                return self.exts[:, self.bands.index(band)][self.testIndexes]

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

    def getAllMags(self, band=None):
        if band is None:
            return self.mags
        else:
            return self.mags[:, self.bands.index(band)]

    def getAllExts(self, band=None):
        if band is None:
            return self.exts
        else:
            if band == 'best':
                idxBest = np.argmax(self.snrs, axis=1)
                idxArr = np.arange(self.nTotal)
                return self.exts[idxArr, idxBest], 1.0/self.snrs[idxArr, idxBest]
            else:
                return self.exts[:, self.bands.index(band)]

    def genColExtTrainSet(self, mode='all', standardized=False, extsMean=None, extsStd=None,
                          magsMean=None, magsStd=None):
        if mode == 'all':
            XSub, XErrSub, Y = self.getAllSet(standardized=standardized)
            exts, extsErr = self.getAllExts(band='best')
        elif mode == 'train':
            XSub, XErrSub, Y = self.getTrainSet(standardized=standardized)
            exts, extsErr = self.getTrainExts(band='best')
        elif mode == 'test':
            XSub, XErrSub, Y = self.getTestSet(standardized=standardized)
            exts, extsErr = self.getTestExts(band='best')
            if standardized:
                if extsMean is None:
                    try:
                        extsMean = self.extsMean
                    except AttributeError:
                        extsMean = np.mean(self.getTrainExts(band='best')[0])
                if extsStd is None:
                    try:
                        extsStd = self.extsStd
                    except AttributeError:
                        extsStd = np.std(self.getTrainExts(band='best')[0])
                if magsMean is None:
                    try:
                        magsMean = self.magsMean
                    except AttributeError:
                        magsMean = np.mean(self.getTrainMags(band='i'))
                if magsStd is None:
                    try:
                        magsStd = self.magsStd
                    except AttributeError:
                        magsStd = np.std(self.getTrainMags(band='i'))
        else:
            raise ValueError("Mode {0} doesn't exist!".format(mode))
        if standardized: 
            if mode == 'all':
                mags = self.getAllMags(band='i')
            elif mode == 'train':
                mags = self.getTrainMags(band='i')
            elif mode == 'test':
                mags = self.getTestMags(band='i')
            if extsMean is None or extsStd is None:
                print "Computing and saving standardization transform for extendedness"
                extsMean = np.mean(exts)
                extsStd = np.std(exts)
            if magsMean is None or magsStd is None:
                print "Computing and saving standardization transform for magnitudes"
                magsMean = np.mean(mags)
                magsStd = np.std(mags)
            exts = (exts - extsMean)/extsStd
            mags = (mags - magsMean)/magsStd
            self.extsMean = extsMean
            self.extsStd = extsStd
            self.magsMean = magsMean
            self.magsStd = magsStd
            XSub = np.concatenate((XSub, mags[:,None]), axis=1)
        X = np.concatenate((XSub, exts[:,None]), axis=1)
        covShapeSub = XErrSub.shape
        dimSub = covShapeSub[1]
        assert dimSub == covShapeSub[2]
        covShape = (covShapeSub[0], dimSub+1, dimSub+1)
        XErr = np.zeros(covShape)
        xxSub, yySub = np.meshgrid(np.arange(dimSub), np.arange(dimSub), indexing='ij')
        XErr[:, xxSub, yySub] = XErrSub
        XErr[:, dimSub, dimSub] = extsErr
        return X, XErr, Y
    
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

def _extractXY(cat, inputs=['ext'], output='mu.class', bands=['i'], magsType='mag', extsType='ext', concatBands=True,
              onlyFinite=True, polyOrder=1, withErr=False, snrType='snr', fromDB=False):
    """
    Load `inputs` from `cat` into `X` and   `output` to `Y`. If onlyFinite is True, then
    throw away all rows with one or more non-finite entries.
    """
    if fromDB and concatBands:
        raise RuntimeError("Can't concatenate bands when data is coming from the database.")
    if not isinstance(cat, afwTable.tableLib.SourceCatalog) and\
       not isinstance(cat, afwTable.tableLib.SimpleCatalog):
        try:
            cat = afwTable.SourceCatalog.readFits(cat)
        except LsstCppException:
            cat = afwTable.SimpleCatalog.readFits(cat)
    nRecords = len(cat); nBands = len(bands); nInputs = len(inputs)
    if concatBands:
        ids = np.zeros((nRecords*len(bands),), dtype=long)
        ras = np.zeros((nRecords*len(bands),))
        decs = np.zeros((nRecords*len(bands),))
        X = np.zeros((nRecords*len(bands), len(inputs)))
        Y = np.zeros((nRecords*len(bands),), dtype=bool)
        mags = np.zeros((nRecords*len(bands),))
        exts = np.zeros((nRecords*len(bands),))
        snrs = np.zeros((nRecords*len(bands),))
        seeings = np.zeros((nRecords*len(bands),))
        if withErr:
            XErr = np.zeros((nRecords*len(bands), len(inputs)))
        for i, band in enumerate(bands):
            ids[i*nRecords:(i+1)*nRecords] = getInput(cat, inputName='id', band=band)
            ras[i*nRecords:(i+1)*nRecords] = getInput(cat, inputName='ra', band=band)
            decs[i*nRecords:(i+1)*nRecords] = getInput(cat, inputName='dec', band=band)
            for j, inputName in enumerate(inputs):
                X[i*nRecords:(i+1)*nRecords, j] = getInput(cat, inputName=inputName, band=band)
                if withErr:
                    XErr[i*nRecords:(i+1)*nRecords, j] = getInputErr(cat, inputName=inputName, band=band)
            Y[i*nRecords:(i+1)*nRecords] = getOutput(cat, outputName=output)
            mags[i*nRecords:(i+1)*nRecords] = getInput(cat, inputName=magsType, band=band)
            exts[i*nRecords:(i+1)*nRecords] = getInput(cat, inputName=extsType, band=band)
            snrs[i*nRecords:(i+1)*nRecords] = getInput(cat, inputName=snrType, band=band)
            seeings[i*nRecords:(i+1)*nRecords] = getInput(cat, inputName='seeing', band=band)
    else:
        if fromDB:
            ids = np.zeros((nRecords,), dtype=long)
            ras = np.zeros((nRecords,))
            decs = np.zeros((nRecords,))
        else:
            ids = np.zeros((nRecords, len(bands)), dtype=long)
            ras = np.zeros((nRecords, len(bands)))
            decs = np.zeros((nRecords, len(bands)))
            seeings = np.zeros((nRecords, len(bands)))
        X = np.zeros((nRecords, len(inputs)*len(bands)))
        Y = np.zeros((nRecords,), dtype=bool)
        mags = np.zeros((nRecords, len(bands)))
        exts = np.zeros((nRecords, len(bands)))
        snrs = np.zeros((nRecords, len(bands)))
        if withErr:
            XErr = np.zeros((nRecords, len(inputs)*len(bands)))
        if fromDB:
            ids[:] = getInput(cat, inputName='id', band='fromDB')
            ras[:] = getInput(cat, inputName='ra', band='fromDB')
            decs[:] = getInput(cat, inputName='dec', band='fromDB')
        for i, band in enumerate(bands):
            if not fromDB:
                try:
                    ids[:, i] = getInput(cat, inputName='id', band=band)
                except KeyError:
                    ids[:, i] = getInput(cat, inputName='id', band='forced')
                ras[:, i] = getInput(cat, inputName='ra', band=band)
                decs[:, i] = getInput(cat, inputName='dec', band=band)
                seeings[:, i] = getInput(cat, inputName='seeing', band=band)
            mags[:, i] = getInput(cat, inputName=magsType, band=band)
            exts[:, i] = getInput(cat, inputName=extsType, band=band)
            snrs[:, i] = getInput(cat, inputName=snrType, band=band)
            for j, inputName in enumerate(inputs):
                X[:, i*nInputs + j] = getInput(cat, inputName=inputName, band=band)
                if withErr:
                    XErr[:, i*nInputs + j] = getInputErr(cat, inputName=inputName, band=band)
        Y = getOutput(cat, outputName=output)
    if concatBands:
        good = np.ones((nRecords*len(bands),), dtype=bool)
    else:
        good = True
    if concatBands:
        for i, band in enumerate(bands):
            good[i*nRecords:(i+1)*nRecords] = np.logical_and(good[i*nRecords:(i+1)*nRecords], getGood(cat, band=bands))
    else:
        good = np.logical_and(good, getGood(cat, band=bands))
    if onlyFinite:
        for i in range(X.shape[1]):
            good = np.logical_and(good, np.isfinite(X[:,i]))
            if withErr:
                good = np.logical_and(good, np.isfinite(XErr[:,i]))
            good = np.logical_and(good, np.isfinite(snrs[:,i]))
    ids = ids[good]; ras = ras[good]; decs = decs[good]; X = X[good]; Y = Y[good]; mags = mags[good]; exts = exts[good];
    snrs = snrs[good]; 
    if not fromDB:
        seeings = seeings[good]
    if withErr:
        XErr = XErr[good]
    if polyOrder > 1:
        X = sgsvm.phiPol(X, polyOrder)
    if withErr:
        if not fromDB:
            return X, XErr, Y, ids, ras, decs, mags, exts, snrs, seeings
        else:
            return X, XErr, Y, ids, ras, decs, mags, exts, snrs
    else:
        if not fromDB:
            return X, Y, ids, ras, decs, mags, exts, snrs, seeings
        else:
            return X, Y, ids, ras, decs, mags, exts, snrs

def extractTrainSet(cat, mode='raw', extBand='i', colType='mag', **kargs):
    if not 'withErr' in kargs:
        kargs['withErr'] = False

    if mode in ['colors', 'rats'] and not kargs['withErr']:
        raise ValueError('You are forced to pull out errors for mode {0}, set keyword `withErr` to True'.format(mode))

    if 'fromDB' in kargs:
        fromDB = kargs['fromDB']
    else:
        fromDB = False

    if kargs['withErr'] == True:
        if fromDB:
            X, XErr, Y, ids, ras, decs, mags, exts, snrs = _extractXY(cat, **kargs)
        else:
            X, XErr, Y, ids, ras, decs, mags, exts, snrs, seeings = _extractXY(cat, **kargs)
    else:
        X, Y, ids, ras, decs, mags, exts, snrs, seeings = _extractXY(cat, **kargs)

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
            trainSet = TrainingSet(X, Y, XErr=XErr, ids=ids, ras=ras, decs=decs, bands=bands, names=names, mags=mags, exts=exts, snrs=snrs, seeings=seeings, polyOrder=polyOrder)
        else:
            trainSet = TrainingSet(X, Y, ids=ids, ras=ras, decs=decs, bands=bands, names=names, mags=mags, exts=exts, snrs=snrs, seeings=seeings, polyOrder=polyOrder)
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
                idxB = names.index(colType + '_'+bands[i])
                idxR = names.index(colType + '_'+bands[i+1])
                XCol[:, i] = X[:, idxB] - X[:, idxR]
                XColErr[:, i] = XErr[:,idxB]**2 + XErr[:,idxR]**2

            covStack = []
            for i in range(1, len(bands)-1):
                idx = names.index(colType + '_' + bands[i])
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
            if fromDB:
                trainSet = TrainingSet(XCol, Y, XErr=XColCov, ids=ids, ras=ras, decs=decs, mags=mags, exts=exts,\
                                       snrs=snrs, bands=bands, names=names, polyOrder=polyOrder)
            else:
                trainSet = TrainingSet(XCol, Y, XErr=XColCov, ids=ids, ras=ras, decs=decs, mags=mags, exts=exts,\
                                       snrs=snrs, seeings=seeings, bands=bands, names=names, polyOrder=polyOrder)
        elif mode == 'colshape':
            names.append('ext_'+extBand)
            if fromDB:
                trainSet = TrainingSet(XColShape, Y, XErr=XColShapeCov, ids=ids, ras=ras, decs=decs, mags=mags, exts=exts,\
                                       snrs=snrs, bands=bands, names=names, polyOrder=polyOrder)
            else:
                trainSet = TrainingSet(XColShape, Y, XErr=XColShapeCov, ids=ids, ras=ras, decs=decs, mags=mags, exts=exts,\
                                       snrs=snrs, seeings=seeings, bands=bands, names=names, polyOrder=polyOrder)
    else:
        raise ValueError("Mode {0} not implemented".format(mode))

    return trainSet

class Training(object):

    def __init__(self, trainingSet, clf, preFit=True):
        self.trainingSet = trainingSet
        self.clf = clf
        self.preFit = preFit
        if hasattr(self.clf, 'best_estimator_'):
            self.estimator = self.clf.best_estimator_
        else:
            self.estimator = self.clf
        if hasattr(self.estimator, 'coef_'):
            self._computePhysicalFit()

    def predictTrainLabels(self, standardized=True, **kargsPred):
        X = self.trainingSet.getTrainSet(standardized=standardized)[0]
        try:
            return self.clf.predict(X, **kargsPred)
        except TypeError:
            X, XErr, Y = self.trainingSet.getTrainSet(standardized=standardized)
            try:
                return self.clf.predict(X, XErr, **kargsPred)
            except TypeError:
                mags = self.trainingSet.getTrainMags(band='i')
                return self.clf.predict(X, XErr, mags, **kargsPred)

    def predictTestLabels(self, standardized=True, **kargsPred):
        X = self.trainingSet.getTestSet(standardized=standardized)[0]
        try:
            return self.clf.predict(X, **kargsPred)
        except TypeError:
            X, XErr, Y = self.trainingSet.getTestSet(standardized=standardized)
            try:
                return self.clf.predict(X, XErr, **kargsPred)
            except TypeError:
                mags = self.trainingSet.getTestMags(band='i')
                return self.clf.predict(X, XErr, mags, **kargsPred)

    def predictAllLabels(self, standardized=True, **kargsPred):
        X = self.trainingSet.getAllSet(standardized=standardized)[0]
        try:
            return self.clf.predict(X, **kargsPred)
        except TypeError:
            X, XErr, Y = self.trainingSet.getAlltSet(standardized=standardized)
            try:
                return self.clf.predict(X, XErr, **kargsPred)
            except TypeError:
                mags = self.trainingSet.getAllMags(band='i')
                return self.clf.predict(X, XErr, mags, **kargsPred)

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
        if hasattr(self.estimator, 'coef_'):
            if normalized:
                print "Normalized coeff:", self.coeffPhys/self.coeffPhys[0]
                print "Normalized intercept:", self.interceptPhys/self.coeffPhys[0]
            else:
                print "coeff:", self.coeffPhys
                print "intercept:", self.interceptPhys
        else:
            print "WARNING: I can't print the physical fit of a {0}".format(self.estimator.__class__)

    def printPolynomial(self, names):
        if hasattr(self.estimator, 'coef_'):
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
        else:
            print "WARNING: I can't print the physical fit of a {0}".format(self.estimator.__class__)

    def plotScores(self, nBins=50, sType='test', fig=None, linestyle='-', fontSize=18,
                   magRange=None, xlabel='Magnitude', ylabel='Scores', legendLabel='',
                   standardized=True, suptitle=None, kargsPred={}, xlim=None, colExt=False, svm=False):
        if sType == 'test':
            mags = self.trainingSet.getTestMags(band='i')
        elif sType == 'train':
            mags = self.trainingSet.getTrainMags(band='i')
        elif sType == 'all':
            mags = self.trainingSet.getAllMags(band='i')
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
        if colExt:
            assert not svm
            X, XErr, Y = self.trainingSet.genColExtTrainSet(mode=sType)
            pred = self.clf.predict(X, XErr, mags, **kargsPred)
            truth = Y
        elif svm:
            X, XErr, Y = self.trainingSet.genColExtTrainSet(mode=sType, standardized=True)
            pred = self.clf.predict(X)
            truth = Y
        else:
            if sType == 'test':
                pred = self.predictTestLabels(standardized=standardized, **kargsPred)
                truth = self.trainingSet.getTestSet()[1]
            elif sType == 'train':
                pred = self.predictTrainLabels(standardized=standardized, **kargsPred)
                truth = self.trainingSet.getTrainSet()[1]
            elif sType == 'all':
                pred = self.predictAllLabels(standardized=standardized, **kargsPred)
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
            fig = plt.figure(figsize=(16, 6), dpi=120)
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
            if suptitle is not None:
                fig.suptitle(suptitle, fontsize=fontSize)
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

        axGal.legend(loc='lower left', fontsize=fontSize-2)
        axStar.legend(loc='lower left', fontsize=fontSize-2)
        
        if xlim is not None:
            for ax in fig.get_axes():
                ax.set_xlim(xlim)

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

    def _getFCoef(self, fixedIndex, fixedVal, varIndex):
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

    def _getF(self, fixedIndex, fixedVal, varIndex):
        assert self.trainingSet.polyOrder == 1
        # TODO: Make it possible to use this function for polyOrder > 1
        Xphys = np.zeros((1, self.trainingSet.X.shape[1]))
        Xphys[0, fixedIndex] = fixedVal
        def F(x):
            Xphys[0, varIndex] = x
            if self.preFit:
                X = self.trainingSet.applyPreTestTransform(Xphys)
            else:
                X = self.trainingSet.applyPostTestTransform(Xphys)
            return self.estimator.decision_function(X)
        return F

    def findZero(self, fixedIndex, varIndex, fixedVal, zeroRange=None, chooseSol=1, fallbackRange=None):
        assert chooseSol in [0, 1]

        if zeroRange is None:
            zeroRange = (self.trainingSet.X[:,varIndex].min(),self.trainingSet.X[:,varIndex].max())
        
        if hasattr(self.clf, 'best_estimator_'):
            estimator = self.clf.best_estimator_
        else:
            estimator = self.clf

        if hasattr(estimator, 'coef_'):
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
                F = self._getFCoef(fixedIndex, fixedVal, varIndex)
                try:
                    return brentq(F, zeroRange[0], zeroRange[1])
                except ValueError as e:
                    if fallbackRange is not None:
                        try:
                            return brentq(F, fallbackRange[0], fallbackRange[1])
                        except ValueError as e:
                            figT = plt.figure()
                            arr = np.linspace(zeroRange[0], zeroRange[1], num=100)
                            ys = np.zeros(arr.shape)
                            for i in range(len(arr)):
                                ys[i] = F(arr[i])
                            plt.plot(arr, ys)
                            plt.title("varRange={0}".format(fixedVal))
                            if os.environ.get('DISPLAY') is not None:
                                plt.show()
                            raise e
                    else:
                        figT = plt.figure()
                        arr = np.linspace(zeroRange[0], zeroRange[1], num=100)
                        ys = np.zeros(arr.shape)
                        for i in range(len(arr)):
                            ys[i] = F(arr[i])
                        plt.plot(arr, ys)
                        plt.title("varRange={0}".format(fixedVal))
                        if os.environ.get('DISPLAY') is not None:
                            plt.show()
                        raise e
        else:
            F = self._getF(fixedIndex, fixedVal, varIndex)
            try:
                return brentq(F, zeroRange[0], zeroRange[1])
            except ValueError as e:
                if fallbackRange is not None:
                    try:
                        return brentq(F, fallbackRange[0], fallbackRange[1])
                    except ValueError as e:
                        figT = plt.figure()
                        arr = np.linspace(zeroRange[0], zeroRange[1], num=100)
                        ys = np.zeros(arr.shape)
                        for i in range(len(arr)):
                            ys[i] = F(arr[i])
                        plt.plot(arr, ys)
                        plt.title("varRange={0}".format(fixedVal))
                        if os.environ.get('DISPLAY') is not None:
                            plt.show()
                        raise e
                else:
                    figT = plt.figure()
                    arr = np.linspace(zeroRange[0], zeroRange[1], num=100)
                    ys = np.zeros(arr.shape)
                    for i in range(len(arr)):
                        ys[i] = F(arr[i])
                    plt.plot(arr, ys)
                    plt.title("varRange={0}".format(fixedVal))
                    if os.environ.get('DISPLAY') is not None:
                        plt.show()
                    raise e

    def getDecBoundary(self, rangeIndex, varIndex, fixedIndexes=None, fixedVals=None, xRange=None, 
                       nPoints=100, yRange=None, asLogX=False, fallbackRange=None):

        if self.trainingSet.X.shape[1] > sgsvm.nterms(self.trainingSet.polyOrder, 2) - 1:
            if fixedIndexes is None or fixedVals is None:
                raise ValueError("If there are more than two inputs I need to know the indexes and values of fixed variables.")
            fixedIndexes.append(rangeIndex)
            fixedVals.append(0.0)
        else:
            assert fixedIndexes is None
            assert fixedVals is None

        if xRange is None:
            xRange = (self.trainingSet.X[:,rangeIndex].min(),self.trainingSet.X[:,rangeIndex].max())

        if asLogX:
            xGrid = np.linspace(np.log10(xRange[0]), np.log10(xRange[1]), num=nPoints)
            xGrid = np.power(10.0, xGrid)
        else:
            xGrid = np.linspace(xRange[0], xRange[1], num=nPoints)
        yGrid = np.zeros(xGrid.shape)

        for i, fixedVal in enumerate(xGrid):
            if fixedIndexes is not None and fixedVals is not None:
                fixedVals[-1] = fixedVal
                yGrid[i] = self.findZero(fixedIndexes, varIndex, fixedVals, zeroRange=yRange, fallbackRange=fallbackRange)
            else:
                yGrid[i] = self.findZero(rangeIndex, varIndex, fixedVal, zeroRange=yRange, fallbackRange=fallbackRange)

        return xGrid, yGrid

    def plotBoundary(self, rangeIndex, varIndex, xRange=None, nPoints=100, fig=None, overPlotData=False,
                     xlim=None, ylim=None, xlabel=None, ylabel=None, yRange=None, frac=0.03,
                     withTrueLabels=True, fontSize=18, asLogX=False):

        xGrid, yGrid = self.getDecBoundary(rangeIndex, varIndex, xRange=xRange, nPoints=nPoints, yRange=yRange, asLogX=asLogX)

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
        try:
            Z = self.clf.predict_proba(X)[:,1]
        except AttributeError:
            Z = self.clf.predict(X).astype('float')
        zz = Z.reshape(xx.shape)
        fig = plt.figure(dpi=120)
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

class BoxClf(object):

    def __init__(self):
        self._xBdy = 0.0
        self._yBdy = 0.0

    def _setX(self, value):
        self._xBdy = value

    def _setY(self, value):
        self._yBdy = value

    def predict(self, X):
       return np.logical_and(X[:,0] < self._xBdy, X[:, 1] < self._yBdy)

    def plotBox(self, trainSet=None, frac=0.01, xlim=(18.0, 26.0), ylim=(-0.4, 1.0)):
        fig = plt.figure(dpi=120)
        ax = fig.add_subplot(1, 1, 1)
        if trainSet is not None:
            size = int(len(trainSet.X)*frac)
            choice = np.random.choice(len(trainSet.X), size=size)
            for idx in choice:
                if trainSet.Y[idx]:
                    plt.plot(trainSet.X[idx, 0], trainSet.X[idx, 1], marker='.', markersize=1, color='blue')
                else:
                    plt.plot(trainSet.X[idx, 0], trainSet.X[idx, 1], marker='.', markersize=1, color='red')
        ax.plot([self._xBdy, self._xBdy], [ylim[0], self._yBdy], color='black')
        ax.plot([xlim[0], self._xBdy], [self._yBdy, self._yBdy], color='black')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        return fig

class IsochroneReader(object):

    def __init__(self, iType='LSST', stringZ='p00', stringA='p0', suffix=None, stringY=None):
        dirHome = os.path.expanduser('~')
        self.isochrones = {}
        if stringY is None:
            fName = os.path.join(dirHome, 'Data/isochrones/{0}/feh{1}afe{2}.{0}'.format(iType, stringZ, stringA))
        else:
            fName = os.path.join(dirHome, 'Data/isochrones/{0}/feh{1}afe{2}y{3}.{0}'.format(iType, stringZ, stringA, stringY))
        if suffix is not None:
            fName += '_2'
        self.readFile(fName)

    def readFile(self, fName):
       with open(fName) as f: 
           for line in f:
               if line[:4] == '#AGE':
                   match = re.match(r"^#AGE=([0-9][0-9].[0-9]*| [0-9].[0-9]*) EEPS=([0-9]*)", line)
                   age = float(match.group(1))
                   eeps = int(match.group(2))
               elif line[:4] == '#EEP':
                   cols = line.split()
                   cols[0] = cols[0][1:]
                   self.isochrones[age] = {}
                   for col in cols:
                       self.isochrones[age][col] = np.zeros((eeps,))
                   count = 0
               elif line[0] == ' ':
                   values = line.split() 
                   for i, col in enumerate(cols):
                       self.isochrones[age][col][count] = values[i]
                   count += 1
               else:
                   continue
