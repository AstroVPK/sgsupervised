import numpy as np
import matplotlib.pyplot as plt

import lsst.afw.table as afwTable

import sgSVM as sgsvm

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

def getExt(cat, band):
    flux = cat.get('cmodel.flux.'+band)
    fluxPsf = cat.get('flux.psf.'+band)
    fluxZero = cat.get('flux.zeromag.'+band)
    mag = -2.5*np.log10(flux/fluxZero)
    ext = -2.5*np.log10(fluxPsf/flux)
    return ext

def getMag(cat, band):
    flux = cat.get('cmodel.flux.'+band)
    fluxZero = cat.get('flux.zeromag.'+band)
    mag = -2.5*np.log10(flux/fluxZero)
    return mag

def getPsfMag(cat, band):
    fluxPsf = cat.get('flux.psf.'+band)
    fluxZero = cat.get('flux.zeromag.'+band)
    mag = -2.5*np.log10(fluxPsf/fluxZero)
    return mag

def _getExtExtLayout(nBands):
    if nBands == 1:
        raise ValueError("I need at least to bands to generate this plot")
    elif nBands == 2:
        return 1, 1
    elif nBands == 3:
        return 1, 3
    elif nBands == 4:
        return 2, 3
    elif nBands == 5:
        return 5, 2

def makeExtExtPlot(cat, bands=['g', 'r', 'i', 'z', 'y'], fontSize=14, size=1,
                   plotStars=True, plotGals=True, xlim=(-0.03, 0.5), ylim=(-0.03, 0.5),
                   magCut=None, byMagCuts=False,
                   magCuts=[(18.0, 22.0), (22.0, 24.0), (24.0, 25.0), (25.0, 26.0)], computeCorr=False,
                   seed=0, nPerm=100):

    np.random.seed(seed)
    nBands = len(bands)

    if byMagCuts:
        assert magCuts is not None
        assert nBands == 2
        nRow, nColumn = 2, 2
    else:
        nRow, nColumn = _getExtExtLayout(nBands)

    if not isinstance(cat, afwTable.tableLib.SourceCatalog) and\
       not isinstance(cat, afwTable.tableLib.SimpleCatalog):
        cat = afwTable.SourceCatalog.readFits(cat)

    fig = plt.figure()
    count = 1

    if not byMagCuts:
        magCuts = [magCut]

    stellar = cat.get('stellar')
    for magCut in magCuts:
        good = True
        for b in bands:
            good = np.logical_and(good, getGood(cat, band=b, magCut=magCut))

        for i in range(nBands):
            flux_i = cat.get('cmodel.flux.'+bands[i])
            fluxPsf_i = cat.get('flux.psf.'+bands[i])
            ext_i = -2.5*np.log10(fluxPsf_i/flux_i)
            #ext_i = (flux_i - fluxPsf_i)/flux_i
            for j in range(i+1, nBands):
                flux_j = cat.get('cmodel.flux.'+bands[j])
                fluxPsf_j = cat.get('flux.psf.'+bands[j])
                ext_j = -2.5*np.log10(fluxPsf_j/flux_j)
                #ext_j = (flux_j - fluxPsf_j)/flux_j
                good = np.logical_and(good, np.isfinite(ext_i))
                good = np.logical_and(good, np.isfinite(ext_j))
                goodStars = np.logical_and(stellar, good)
                #goodStars = np.logical_and(goodStars, ext_i < 0.01)
                #goodStars = np.logical_and(goodStars, ext_i**2+ext_j**2 < 0.05**2)
                goodGals = np.logical_and(np.logical_not(stellar), good)
                ax = fig.add_subplot(nRow, nColumn, count)
                if xlim is not None:
                    ax.set_xlim(xlim)
                if ylim is not None:    
                    ax.set_ylim(ylim)
                ax.set_xlabel('Extendedness HSC-{0}'.format(bands[i].upper()), fontsize=fontSize)
                ax.set_ylabel('Extendedness HSC-{0}'.format(bands[j].upper()), fontsize=fontSize)
                for tick in ax.xaxis.get_major_ticks():
                    tick.label.set_fontsize(fontSize)
                for tick in ax.yaxis.get_major_ticks():
                    tick.label.set_fontsize(fontSize)
                if magCut is not None:
                    ax.set_title('{0} < Magnitude HSC-I < {1}'.format(magCut[0], magCut[1]), fontsize=fontSize)
                if plotStars:
                    ax.scatter(ext_i[goodStars], ext_j[goodStars], marker='.', s=size)
                if plotGals:
                    ax.scatter(ext_i[goodGals], ext_j[goodGals], marker='.', s=size)
                if computeCorr:
                    if plotStars:
                        corr = np.corrcoef(np.vstack((ext_i[goodStars], ext_j[goodStars])))
                        print "{0} < Magnitude HSC-I < {1} (Stars): Corr={2}".format(magCut[0], magCut[1], corr[0,1])
                        corrList = np.zeros((nPerm,))
                        for k in range(nPerm):
                            iPerm = np.random.permutation(ext_i[goodStars])
                            jPerm = np.random.permutation(ext_j[goodStars])
                            corrList[k] = np.corrcoef(np.vstack((iPerm, jPerm)))[0,1]
                        print "meanCorr={0}, stdCorr={1}, p-value={2}".format(np.mean(corrList), np.std(corrList), np.sum(np.logical_and(True, corrList >= corr[0,1]))*1.0/nPerm)
                    if plotGals:
                        corr = np.corrcoef(np.vstack((ext_i[goodGals], ext_j[goodGals])))
                        print "{0} < Magnitude HSC-I < {1} (Galaxies): Corr={2}".format(magCut[0], magCut[1], corr[0,1])
                        corrList = np.zeros((nPerm,))
                        for k in range(nPerm):
                            iPerm = np.random.permutation(ext_i[goodGals])
                            jPerm = np.random.permutation(ext_j[goodGals])
                            corrList[k] = np.corrcoef(np.vstack((iPerm, jPerm)))[0,1]
                        print "meanCorr={0}, stdCorr={1}, p-value={2}".format(np.mean(corrList), np.std(corrList), np.sum(np.logical_and(True, corrList >= corr[0,1]))*1.0/nPerm)
                count += 1
    return fig

def makeMatchMagPlot(cat, fontSize=18, lOffsetStar=0.2, starDiff=0.3, lOffsetGal=2.0, galDiff=0.8, band='i',
                     xlim=(16.5, 29.0), ylim=(16.5, 29.0), maxExt=2.0, minExt=-0.4):
    """
    Make MAG_AUTO vs CModel plot to identify problematic matches. Default values are meant for HSC-I.
    """
    if not isinstance(cat, afwTable.tableLib.SourceCatalog) and\
       not isinstance(cat, afwTable.tableLib.SimpleCatalog):
        cat = afwTable.SourceCatalog.readFits(cat)
    fluxI = cat.get('cmodel.flux.'+band)
    fluxPsfI = cat.get('flux.psf.'+band)
    fluxZeroI = cat.get('flux.zeromag.'+band)
    magI = -2.5*np.log10(fluxI/fluxZeroI)
    extI = -2.5*np.log10(fluxPsfI/fluxI)
    magAuto = cat.get('mag.auto')
    try:
        stellar = cat.get('stellar')
    except KeyError:
        stellar = cat.get('mu.class') == 2
    good = np.logical_and(True, np.abs(magI - magAuto) < 10.0)
    good = np.logical_and(good, extI < maxExt)
    good = np.logical_and(good, extI > minExt)
    goodStar = np.logical_and(good, stellar)
    goodGal = np.logical_and(good, np.logical_not(stellar))
    x = np.linspace(15.0, 30.0, num=100)
    y = np.linspace(15.0, 30.0, num=100)

    fig = plt.figure()
    cbar_ax = fig.add_axes([0.91, 0.1, 0.02, 0.8])
    axStar = fig.add_subplot(1, 2, 1)
    axStar.set_title('Putative Stars', fontsize=fontSize)
    axStar.set_xlabel('MAG_AUTO F814W', fontsize=fontSize)
    axStar.set_ylabel('CModel Magnitude HSC-{0}'.format(band.upper()), fontsize=fontSize)
    axStar.set_xlim(xlim); axStar.set_ylim(ylim)
    axGal = fig.add_subplot(1, 2, 2)
    axGal.set_title('Putative Galaxies', fontsize=fontSize)
    axGal.set_xlabel('MAG_AUTO F814W', fontsize=fontSize)
    axGal.set_ylabel('CModel Magnitude HSC-{0}'.format(band.upper()), fontsize=fontSize)
    axGal.set_xlim(xlim); axGal.set_ylim(ylim)
    
    for ax in [axStar, axGal]:
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontSize-2)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontSize-2)

    scStar = axStar.scatter(magAuto[goodStar], magI[goodStar], marker='.', s=8, c=extI[goodStar], edgecolors='none')
    axStar.plot(x, y+starDiff, linestyle='-', color='black')
    axStar.plot(x, y-lOffsetStar-starDiff, linestyle='-', color='black')
    scGal = axGal.scatter(magAuto[goodGal], magI[goodGal], marker='.', s=8, c=extI[goodGal], edgecolors='none')
    axGal.plot(x, y+galDiff, linestyle='-', color='black')
    axGal.plot(x, y-lOffsetGal-galDiff, linestyle='-', color='black')


    cb = fig.colorbar(scGal, cax=cbar_ax, use_gridspec=True)
    cb.ax.tick_params(labelsize=fontSize)
    cb.set_label('Extendedness', fontsize=fontSize)


    return fig

def makeMatchMagPlotMulti(cat, band='i'):
   """
   Wrapper used to keep track of the matching settings.
   """
   return makeMatchMagPlot(cat, band=band, **kargOutlier[band])
    
def getMatchOutNum(cat):
    good = dropMatchOutliers(cat, band='g', lOffsetStar=-3.5, starDiff=3.9, lOffsetGal=-0.8, galDiff=3.7)
    good = dropMatchOutliers(cat, good=good, band='r', lOffsetStar=-2.2, starDiff=2.7, lOffsetGal=0.5, galDiff=2.3)
    good = dropMatchOutliers(cat, good=good, band='i')
    good = dropMatchOutliers(cat, good=good, band='z', lOffsetStar=1.0, starDiff=0.0, lOffsetGal=2.5)
    good = dropMatchOutliers(cat, good=good, band='y', lOffsetStar=1.4, starDiff=0.0, lOffsetGal=2.5)
    return good

def makeExtSeeingSnrPlot(cat, band, size=1, withLabels=False, fontSize=18):
    if not isinstance(cat, afwTable.tableLib.SourceCatalog) and\
       not isinstance(cat, afwTable.tableLib.SimpleCatalog):
        cat = afwTable.SourceCatalog.readFits(cat)
    flux = cat.get('cmodel.flux.'+band)
    fluxPsf = cat.get('flux.psf.'+band)
    ext = -2.5*np.log10(fluxPsf/flux)
    fluxErr = cat.get('cmodel.flux.err.'+band)
    snr = flux/fluxErr
    seeing = cat.get('seeing.'+band)
    good = getGood(cat, band)
    if withLabels:
        stellar = cat.get('stellar')

    fig = plt.figure()

    axSeeing = fig.add_subplot(1, 2, 1)
    axSnr = fig.add_subplot(1, 2, 2)
    axSeeing.set_xlabel('Seeing (arcseconds)', fontsize=fontSize)
    axSnr.set_xlabel('S/N', fontsize=fontSize)
    axSeeing.set_ylabel('Extendedness', fontsize=fontSize)
    axSnr.set_ylabel('Extendedness', fontsize=fontSize)
    #axSeeing.set_xlim((0.69, 0.82))
    axSnr.set_xlim((1.0, 30.0))
    axSeeing.set_ylim((-0.1, 4.0))
    axSnr.set_ylim((-0.1, 4.0))
    for ax in [axSeeing, axSnr]:
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontSize)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontSize)
    if withLabels:
        stars = np.logical_and(good, stellar)
        gals = np.logical_and(good, np.logical_not(stellar))
        axSeeing.scatter(seeing[gals], ext[gals], marker='.', s=size, color='red', label='Galaxies')
        axSeeing.scatter(seeing[stars], ext[stars], marker='.', s=size, color='blue', label='Stars')
        axSnr.scatter(snr[gals], ext[gals], marker='.', s=size, color='red', label='Galaxies')
        axSnr.scatter(snr[stars], ext[stars], marker='.', s=size, color='blue', label='Stars')
    else:
        axSeeing.scatter(seeing[good], ext[good], marker='.', s=size)
        axSnr.scatter(snr[good], ext[good], marker='.', s=size)

    return fig

def makeMagSnrPlot(cat, band, size=1, log=True):
    if not isinstance(cat, afwTable.tableLib.SourceCatalog) and\
       not isinstance(cat, afwTable.tableLib.SimpleCatalog):
        cat = afwTable.SourceCatalog.readFits(cat)
    flux = cat.get('cmodel.flux.'+band)
    fluxErr = cat.get('cmodel.flux.err.'+band)
    fluxZero = cat.get('flux.zeromag.'+band)
    mag = -2.5*np.log10(flux/fluxZero)
    snr = flux/fluxErr
    good = getGood(cat, band)

    fig = plt.figure()
    if log:
        plt.scatter(mag[good], np.log10(snr[good]), marker='.', s=size)
    else:
        plt.scatter(mag[good], snr[good], marker='.', s=size)

    return fig

def makeSeeingExPlot(cat, bands, size=1, fontSize=14, withLabels=False,
                     xlim=None, ylim=None, trueSample=False, magMin=None,
                     extMax=None, type='ext'):
    if not isinstance(cat, afwTable.tableLib.SourceCatalog) and\
       not isinstance(cat, afwTable.tableLib.SimpleCatalog):
        cat = afwTable.SourceCatalog.readFits(cat)
    fig = plt.figure()
    nRow, nColumn = _getExtHistLayout(len(bands))
    for i in range(len(bands)):
        ax = fig.add_subplot(nRow, nColumn, i+1)
        band = bands[i]
        flux = cat.get('cmodel.flux.'+band)
        fluxPsf = cat.get('flux.psf.'+band)
        if withLabels:
            stellar = cat.get('stellar')
        seeing = cat.get('seeing.'+band)*0.16*2.35
        seeingSet = set(seeing)
        if withLabels:
            meanStar = np.zeros((len(seeingSet),))
            meanGal = np.zeros((len(seeingSet),))
            stdStar = np.zeros((len(seeingSet),))
            stdGal = np.zeros((len(seeingSet),))
        ext = -2.5*np.log10(fluxPsf/flux)
        if type == 'ext':
            data = ext
            ax.set_ylabel('Extendedness HSC-'+band.upper(), fontsize=fontSize)
        elif type == 'edext':
            fluxExp = cat.get('cmodel.exp.flux.'+band)
            fluxDev = cat.get('cmodel.dev.flux.'+band)
            data = -2.5*np.log10(fluxExp/fluxDev)
            ax.set_ylabel('Exp_mag-Dev_mag HSC-'+band.upper(), fontsize=fontSize)
        elif type == 'rexp':
            q, data = sgsvm.getShape(cat, band, 'exp')
            ax.set_ylabel('rExp HSC-'+band.upper(), fontsize=fontSize)
        elif type == 'rdev':
            q, data = sgsvm.getShape(cat, band, 'dev')
            ax.set_ylabel('rDev HSC-'+band.upper(), fontsize=fontSize)
        else:
            data = cat.get(type + '.' + band)
            ax.set_ylabel(type + ' HSC-'+band.upper(), fontsize=fontSize)
        good = getGood(cat, band)
        if magMin:
            fluxZero = cat.get('flux.zeromag.'+band)
            mag = -2.5*np.log10(flux/fluxZero)
            good = np.logical_and(good, mag > magMin)
        if extMax:
            good = np.logical_and(good, ext < extMax)
        ax.set_xlabel('Seeing HSC-'+band.upper(), fontsize=fontSize)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        if withLabels:
            good = np.logical_and(good, np.isfinite(data))
            gals = np.logical_and(good, np.logical_not(stellar))
            stars = np.logical_and(good, stellar)
            globalMeanStar = np.mean(data[stars])
            globalMeanGal = np.mean(data[gals])
            for j, s in enumerate(seeingSet):
                sample = np.logical_and(True, seeing == s)
                sampleStar = np.logical_and(stars, sample); nStar = np.sum(sampleStar)
                sampleGal = np.logical_and(gals, sample); nGal = np.sum(sampleGal)
                meanStar[j] = np.mean(data[sampleStar])
                meanGal[j] = np.mean(data[sampleGal])
                if nStar > 1:
                    stdStar[j] = np.std(data[sampleStar])/np.sqrt(nStar-1)
                else:
                    stdStar[j] = 2*globalMeanGal
                if nGal > 1:
                    stdGal[j] = np.std(data[sampleGal])/np.sqrt(nGal-1)
                else:
                    stdGal[j] = 2*globalMeanGal
            ax.errorbar(np.array(list(seeingSet)), meanStar, yerr=stdStar, fmt='o', color='blue')
            ax.errorbar(np.array(list(seeingSet)), meanGal, yerr=stdGal, fmt='o', color='red')
            ax.axhline(y=globalMeanStar, xmin=0.0, xmax=1.0, linewidth=1, linestyle='--', color='blue')
            ax.axhline(y=globalMeanGal, xmin=0.0, xmax=1.0, linewidth=1, linestyle='--', color='red')
            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(fontSize)
            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(fontSize)
    return fig

def makeMagExPlot(cat, band, size=1, fontSize=14, withLabels=False,
                  xlim=(17.5, 28.0), ylim=(-0.05, 0.5), trueSample=False,
                  frac=0.02, type='ext', data=None, xType='mag', kargGood={}):
    if not isinstance(cat, afwTable.tableLib.SourceCatalog) and\
       not isinstance(cat, afwTable.tableLib.SimpleCatalog):
        cat = afwTable.SourceCatalog.readFits(cat)
    fig = plt.figure()
    if isinstance(band, list) or isinstance(band, tuple):
        bands = band
        nRow, nColumn = _getExtHistLayout(len(band))
        for i in range(min(nRow*nColumn, len(bands))):
            ax = fig.add_subplot(nRow, nColumn, i+1)
            band = bands[i]
            flux = cat.get('cmodel.flux.'+band)
            fluxPsf = cat.get('flux.psf.'+band)
            fluxZero = cat.get('flux.zeromag.'+band)
            if withLabels:
                try:
                    stellar = cat.get('stellar')
                except KeyError:
                    stellar = cat.get('mu.class') == 2
            if xType == 'mag':
                mag = -2.5*np.log10(flux/fluxZero)
            elif xType == 'seeing':
                mag = cat.get('seeing.'+band)
            else:
                raise ValueError("I don't recognize the xType value")
            ext = -2.5*np.log10(fluxPsf/flux)
            good = getGood(cat, band, **kargGood)
            if data is None:
                if type == 'ext':
                    data = ext
                    ax.set_ylabel('Extendedness HSC-'+band.upper(), fontsize=fontSize)
                elif type == 'rexp':
                    q, data = sgsvm.getShape(cat, band, 'exp')
                    ax.set_ylabel('rExp HSC-'+band.upper(), fontsize=fontSize)
                elif type == 'rdev':
                    q, data = sgsvm.getShape(cat, band, 'dev')
                    ax.set_ylabel('rDev HSC-'+band.upper(), fontsize=fontSize)
                else:
                    data = cat.get(type + '.' + band)
                    ax.set_ylabel(type + ' HSC-'+band.upper(), fontsize=fontSize)
            ax.set_xlabel('Magnitude HSC-'+band.upper(), fontsize=fontSize)
            if xlim is not None:
                ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            if withLabels:
                gals = np.logical_and(good, np.logical_not(stellar))
                stars = np.logical_and(good, stellar)
                if trueSample:
                    for i in range(int(frac*len(mag))):
                        if stars[i]:
                            ax.plot(mag[i], data[i], marker='.', markersize=size, color='blue')
                        elif gals[i]:
                            ax.plot(mag[i], data[i], marker='.', markersize=size, color='red')
                else:
                    ax.scatter(mag[gals], data[gals], marker='.', s=size, color='red', label='Galaxies')
                    ax.scatter(mag[stars], data[stars], marker='.', s=size, color='blue', label='Stars')
            else:
                if trueSample:
                    for i in range(int(frac*len(mag))):
                        if good[i]:
                            ax.plot(mag[i], data[i], marker='.', markersize=size, color='black')
                else:
                    ax.scatter(mag[good], data[good], marker='.', s=size)
            data = None
            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(fontSize)
            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(fontSize)
            if withLabels:
                ax.legend(loc=1, fontsize=18)
        return fig
    flux = cat.get('cmodel.flux.'+band)
    fluxPsf = cat.get('flux.psf.'+band)
    fluxZero = cat.get('flux.zeromag.'+band)
    if withLabels:
        try:
            stellar = cat.get('stellar')
        except KeyError:
            stellar = cat.get('mu.class') == 2
    mag = -2.5*np.log10(flux/fluxZero)
    ext = -2.5*np.log10(fluxPsf/flux)
    good = getGood(cat, band, **kargGood)
    if data is None:
        if type == 'ext':
            data = ext
            plt.ylabel('Extendedness HSC-'+band.upper(), fontsize=fontSize)
        elif type == 'rexp':
            q, data = sgsvm.getShape(cat, band, 'exp')
            plt.ylabel('rExp HSC-'+band.upper(), fontsize=fontSize)
        elif type == 'rdev':
            q, data = sgsvm.getShape(cat, band, 'dev')
            plt.ylabel('rDev HSC-'+band.upper(), fontsize=fontSize)
        else:
            data = cat.get(type + '.' + band)
            plt.ylabel(type + ' HSC-'+band.upper(), fontsize=fontSize)
    plt.xlabel('Magnitude HSC-'+band.upper(), fontsize=fontSize)
    plt.xlim(xlim)
    plt.ylim(ylim)
    if withLabels:
        gals = np.logical_and(good, np.logical_not(stellar))
        stars = np.logical_and(good, stellar)
        if trueSample:
            for i in range(int(frac*len(mag))):
                if stars[i]:
                    plt.plot(mag[i], data[i], marker='.', markersize=size, color='blue')
                else:
                    plt.plot(mag[i], data[i], marker='.', markersize=size, color='red')
        else:
            plt.scatter(mag[gals], data[gals], marker='.', s=size, color='red', label='Galaxies')
            plt.scatter(mag[stars], data[stars], marker='.', s=size, color='blue', label='Stars')
    else:
        plt.scatter(mag[good], data[good], marker='.', s=size)
    ax = fig.get_axes()[0]
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontSize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontSize)
    if withLabels:
        plt.legend(loc=1, fontsize=18)
    return fig

def _getExtHistLayout(nCuts):
    if nCuts == 1:
        return 1, 1
    elif nCuts == 2:
        return 1, 2
    elif nCuts == 3:
        return 1, 3
    elif nCuts == 4:
        return 2, 2
    elif nCuts == 5:
        return 3, 2
    else:
        raise ValueError("Using more than 5 cuts is not implemented")


def makeExtHist(cat, band, magCuts=None, nBins=100, fontSize=14, withLabels=False,
                normed=False, xlim=None, type='ext', data=None):
    if not isinstance(cat, afwTable.tableLib.SourceCatalog) and\
       not isinstance(cat, afwTable.tableLib.SimpleCatalog):
        cat = afwTable.SourceCatalog.readFits(cat)
    if magCuts is None:
        magCuts = [(18.0, 26.0)]
    elif not isinstance(magCuts, list):
        assert isinstance(magCuts, tuple)
        assert len(magCuts) == 2
        magCuts = [magCuts]
    nCuts = len(magCuts)
    nRow, nColumn = _getExtHistLayout(nCuts)
    flux = cat.get('cmodel.flux.'+band)
    fluxPsf = cat.get('flux.psf.'+band)
    fluxZero = cat.get('flux.zeromag.'+band)
    if withLabels:
        try:
            stellar = cat.get('stellar')
        except KeyError:
            stellar = cat.get('mu.class') == 2
    mag = -2.5*np.log10(flux/fluxZero)
    ext = -2.5*np.log10(fluxPsf/flux)
    fluxI = cat.get('cmodel.flux.i')
    fluxZeroI = cat.get('flux.zeromag.i')
    magI = -2.5*np.log10(fluxI/fluxZeroI)
    if data is None:
        if type == 'ext':
            data = ext
        elif type == 'rexp':
            q, data = sgsvm.getShape(cat, band, 'exp')
        elif type == 'rdev':
            q, data = sgsvm.getShape(cat, band, 'dev')
        else:
            data = cat.get(type + '.' + band)
    good = getGood(cat, band)
    good = np.logical_and(good, np.isfinite(data))
    fig = plt.figure()
    for i in range(nRow*nColumn):
        magCut = magCuts[i]
        goodCut = np.logical_and(good, magI >= magCut[0])
        goodCut = np.logical_and(goodCut, magI <= magCut[1])
        ax = fig.add_subplot(nRow, nColumn, i+1)
        ax.set_xlabel('Extendedness HSC-' + band.upper(), fontsize=fontSize)
        if xlim is not None:
            ax.set_xlim(xlim)
        if normed:
            ax.set_ylabel('Probability Density', fontsize=fontSize)
        else:
            ax.set_ylabel('Object Counts', fontsize=fontSize)
        ax.set_title('{0} < Magnitude HSC-I < {1}'.format(*magCut), fontsize=fontSize)
        hist, bins = np.histogram(data[goodCut], bins=nBins, range=xlim)
        if withLabels:
            # Make sure the same binning is being used to make meaningful comparisons
            gals = np.logical_and(goodCut, np.logical_not(stellar))
            stars = np.logical_and(goodCut, stellar)
            if np.sum(stars) > 0:
                ax.hist(data[stars], bins=bins, histtype='step', normed=normed, color='blue', label='Stars')
            if normed:
                ylim = ax.get_ylim()
                ax.set_ylim(ylim)
            if np.sum(gals) > 0:
                ax.hist(data[gals], bins=bins, histtype='step', normed=normed, color='red', label='Galaxies')
        else:
            ax.hist(data[goodCut], bins=bins, histtype='step', normed=normed, color='black')
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontSize)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontSize)
        if withLabels:
            ax.legend(loc=1, fontsize=fontSize)
    return fig

def testClfs(clfList, X, Y, cols=None, magCol=None, fig=None, colList=None, pltKargs=None, magIdx=2, printCoeffs=False):
    if cols is not None:
        Xsub = X[:,cols]
    else:
        Xsub = X
    trainIndexes, testIndexes = sgsvm.selectTrainTest(Xsub)
    trainMean = np.mean(Xsub[trainIndexes], axis=0); trainStd = np.std(Xsub[trainIndexes], axis=0)
    X_train = (Xsub[trainIndexes] - trainMean)/trainStd; Y_train = Y[trainIndexes]
    X_test = (Xsub[testIndexes] - trainMean)/trainStd; Y_test = Y[testIndexes]
    for i, clf in enumerate(clfList):
        if colList is None:
            X_trainSub = X_train
            X_testSub = X_test
        else:
            X_trainSub = X_train[:,colList[i]]
            X_testSub = X_test[:,colList[i]]
        clf.fit(X_trainSub, Y_train)
        print "score_{0}=".format(i), clf.score(X_testSub, Y_test)
        if pltKargs is None:
            kargs = {}
        else:
            kargs = pltKargs[i]
        if fig is None:
            fig = sgsvm.plotMagCuts(clf, X_test=X_testSub, Y_test=Y_test, X=X[testIndexes][:,magIdx], **kargs)
        else:
            fig = sgsvm.plotMagCuts(clf, X_test=X_testSub, Y_test=Y_test, X=X[testIndexes][:,magIdx], fig=fig, **kargs)
        if printCoeffs:
            trainMean = np.mean(Xsub, axis=0); trainStd = np.std(Xsub, axis=0)
            X_train = (Xsub - trainMean)/trainStd; Y_train = Y
            if colList is None:
                X_trainSub = X_train
            else:
                X_trainSub = X_train[:,colList[i]]
                trainMean = np.mean(Xsub[:, colList[i]], axis=0); trainStd = np.std(Xsub[:, colList[i]], axis=0)
            clf.fit(X_trainSub, Y_train)
            print "coeffs*std=", clf.coef_
            coeffs = clf.coef_/trainStd
            coeffs = coeffs[0]
            intercept = clf.intercept_ - np.sum(clf.coef_*trainMean/trainStd)
            intercept = intercept[0]
            #coeffs /= intercept; intercept /= intercept
            print "coeffs=", coeffs
            print "intercept=", intercept
            if len(coeffs) == 1:
                print "Cut={0}".format(-intercept/coeffs[0])
    return fig

def testLinearModels(bands=['g', 'r', 'i', 'z', 'y'], catType='hsc', inputFile='sgClassCosmosDeepCoaddSrcHsc-119320150325GRIZY.fits',
                     doMagColors=False, magCut=None, galSub=False, galFrac=0.1, equalNumbers=True, magIdx=2, **kargs):
    clfList = []
    colList = []
    pltKargs = []
    X, Y = sgsvm.loadData(bands=bands, catType=catType, inputFile=inputFile, doMagColors=doMagColors, magCut=magCut, 
                          withDepth=False, withSeeing=False, withDevShape=False, withExpShape=False, withDevMag=False,
                          withExpMag=False, withFracDev=False, **kargs)
    if galSub:
        X, Y = sgsvm.galaxySubSample(X, Y, galFrac=galFrac, equalNumbers=equalNumbers)

    # Extendedness cut
    colList.append([6])
    clfList.append(sgsvm.getClassifier(clfType='linearsvc', C=10.0))
    pltKargs.append({'linestyle':':', 'xlabel':'Magnitude HSC-I', 'title': 'Linear SVM with Equal Numbers'})
    # Extendedness cut and apparent mag
    colList.append([1, 6])
    clfList.append(sgsvm.getClassifier(clfType='linearsvc', C=10.0))
    pltKargs.append({'linestyle':'--'})
    # All bands
    colList.append([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    clfList.append(sgsvm.getClassifier(clfType='linearsvc', C=10.0))
    pltKargs.append({'linestyle':'-'})
    fig = testClfs(clfList, X, Y, colList=colList, pltKargs=pltKargs, printCoeffs=True, magIdx=magIdx)
    return fig

def testRBFModels(bands=['g', 'r', 'i', 'z', 'y'], catType='hsc', inputFile='sgClassCosmosDeepCoaddSrcHsc-119320150325GRIZY.fits',
                  doMagColors=False, magCut=None, galSub=False, galFrac=0.1, equalNumbers=True, magIdx=2, **kargs):
    clfList = []
    colList = []
    pltKargs = []
    X, Y = sgsvm.loadData(bands=bands, catType=catType, inputFile=inputFile, doMagColors=doMagColors, magCut=magCut, 
                          withDepth=False, withSeeing=False, withDevShape=False, withExpShape=False, withDevMag=False,
                          withExpMag=False, withFracDev=False, **kargs)
    if galSub:
        X, Y = sgsvm.galaxySubSample(X, Y, galFrac=galFrac, equalNumbers=equalNumbers)

    # Extendedness cut and apparent mag
    colList.append([1, 6])
    clfList.append(sgsvm.getClassifier(clfType='svc', C=10.0, gamma=0.1))
    pltKargs.append({'linestyle':'--'})
    # All bands
    colList.append([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    clfList.append(sgsvm.getClassifier(clfType='svc', C=10.0, gamma=0.1))
    pltKargs.append({'linestyle':'-'})
    fig = testClfs(clfList, X, Y, colList=colList, pltKargs=pltKargs, printCoeffs=False, magIdx=magIdx)
    return fig

def testSingleBandModels(band='r', catType='hsc', inputFile='sgClassCosmosDeepCoaddSrcHsc-119320150325GRIZY.fits',
                         doMagColors=False, magCut=None, X=None, Y=None, **kargs):
    clfList = []
    colList = []
    pltKargs = []
    if X is None or Y is None:
        bands = ['i', band]
        X, Y = sgsvm.loadData(bands=bands, catType=catType, inputFile=inputFile, doMagColors=doMagColors, magCut=magCut, 
                              withDepth=False, withSeeing=False, withDevShape=True, withExpShape=True, withDevMag=True,
                              withExpMag=True, withFracDev=True, **kargs)
    # Linear Model
    #colList.append([1, 3, 6, 7, 10, 11, 13, 15])
    #colList.append([7])
    colList.append([1, 3, 6, 7, 10, 11, 13, 15])
    clfList.append(sgsvm.getClassifier(clfType='linearsvc', C=10.0))
    pltKargs.append({'linestyle':'--', 'xlabel':'Magnitude HSC-I', 'title': 'Single Band SVM'})
    # Extendedness cut and apparent mag
    #colList.append([1, 3, 7, 11, 13, 15])
    colList.append([1, 3, 6, 7, 10, 11, 13, 15])
    clfList.append(sgsvm.getClassifier(clfType='svc', C=10.0, gamma=0.1))
    pltKargs.append({'linestyle':'-'})
    # Extendedness cut
    colList.append([3])
    clfList.append(sgsvm.getClassifier(clfType='linearsvc', C=10.0))
    pltKargs.append({'linestyle':':'})
    fig = testClfs(clfList, X, Y, colList=colList, pltKargs=pltKargs, magIdx=1)
    return fig, X, Y

def plotClfsBdy(band='r', catType='hsc', inputFile='sgClassCosmosDeepCoaddSrcHsc-119320150325GRIZY.fits',
                doMagColors=False, magCut=None, frac=0.3, size=4, galSub=False, galFrac=0.1, equalNumbers=True,
                **kargs):

    bands = [band]
    X, Y = sgsvm.loadData(bands=bands, catType=catType, inputFile=inputFile, doMagColors=doMagColors, magCut=magCut, 
                          withDepth=False, withSeeing=False, withDevShape=False, withExpShape=False, withDevMag=False,
                          withExpMag=False, withFracDev=False, **kargs)

    if galSub:
        X, Y = sgsvm.galaxySubSample(X, Y, galFrac=galFrac, equalNumbers=equalNumbers)

    # Extendedness cut
    Xsub = X[:, [1]]
    trainMean = np.mean(Xsub, axis=0); trainStd = np.std(Xsub, axis=0)
    X_train = (Xsub - trainMean)/trainStd; Y_train = Y
    clf = sgsvm.getClassifier(clfType='linearsvc', C=10.0)
    clf.fit(X_train, Y_train)
    coeffs = clf.coef_/trainStd
    intercept = clf.intercept_ - np.sum(clf.coef_*trainMean/trainStd)
    cut = -intercept[0]/coeffs[0]
    fig = plt.figure()
    nPlot = int(frac*len(X))
    indexes = np.random.choice(len(X), nPlot, replace=False)
    for idx in indexes:
        if Y[idx]:
            plt.plot(X[idx, 0], X[idx, 1], marker='.', markersize=size, color='blue')
        else:
            plt.plot(X[idx, 0], X[idx, 1], marker='.', markersize=size, color='red')

    plt.plot((18.0, 28.0), (cut, cut), linestyle=':', linewidth=3, color='black')

    plt.xlim((20.0, 27.0)); plt.ylim((-0.03, 0.15))
    plt.xlabel('Magnitude HSC-{0}'.format(band.upper()), fontsize=18)
    plt.ylabel('Extendedness HSC-{0}'.format(band.upper()), fontsize=18)

    ax = fig.get_axes()[0]
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(18)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(18)

    Xsub = X
    trainMean = np.mean(Xsub, axis=0); trainStd = np.std(Xsub, axis=0)
    X_train = (Xsub - trainMean)/trainStd; Y_train = Y

    mags = np.linspace(20.0, 27.0, num=200)

    # Linear Model
    clf = sgsvm.getClassifier(clfType='linearsvc', C=10.0)
    clf.fit(X_train, Y_train)
    fig = sgsvm.plotDecBdy(clf, mags, X=Xsub, Y=Y, linestyle='--', xlim=(20.0, 27.0), ylim=(-0.03, 0.15), fig=fig)

    # RBF Kernel
    clf = sgsvm.getClassifier(clfType='svc', C=10.0, gamma=0.1)
    clf.fit(X_train, Y_train)
    fig = sgsvm.plotDecBdy(clf, mags, X=Xsub, Y=Y, linestyle='-', fig=fig)

    return fig

def makeSingleBandPlot(X=None, Y=None, galSub=False):
    fig, X, Y = sgsvm.fitBands(cols=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50], makePlots=['r'], clfType='svc', X=X, Y=Y,
                               compareToExtCut=True, galSub=galSub, linestyle='-', clfKargs={'C':10.0, 'gamma':0.1},
                               skipBands=['g', 'i', 'z', 'y'], withCV=False)
    fig, X, Y = sgsvm.fitBands(cols=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50], makePlots=['r'], clfType='logit', X=X, Y=Y,
                               compareToExtCut=False, galSub=galSub, linestyle='--', fig=fig, skipBands=['g', 'i', 'z', 'y'])
    return fig, X, Y

def fitBandsSingleExp(bands=['g', 'r', 'i', 'z', 'y'], **kargs):
    for b in bands:
        inputFile = '/u/garmilla/Data/HSC/sgClassCosmosSrcHsc-121120150413{0}.fits'.format(b.upper())
        sgsvm.fitBands(bands=[b], inputFile=inputFile, samePop=False, iBandCut=False, sameBandCut=True, galDiff=10.0, starDiff=10.0, 
                       **kargs)

def plotPred(mags, Y, Ypred, magLim=(18.0, 27.0), magWidth=1.0, nMag=100, title=None, band='r', xlabel=None):

    magCuts = np.linspace(magLim[0], magLim[1], num=nMag)
    starCompleteness = np.zeros(magCuts.shape)
    starPurity = np.zeros(magCuts.shape)
    galCompleteness = np.zeros(magCuts.shape)
    galPurity = np.zeros(magCuts.shape)

    for i, magCut in enumerate(magCuts):
        cut = np.logical_and(mags > magCut-magWidth/2, mags < magCut + magWidth/2)
        Ycut = Y[cut]; YpredCut = Ypred[cut]
        good = Ycut == YpredCut
        goodStar = np.logical_and(good, Ycut)
        goodGal = np.logical_and(good, np.logical_not(Ycut))
        goodStarPred = np.logical_and(good, YpredCut)
        goodGalPred = np.logical_and(good, np.logical_not(YpredCut))
        nStar = np.sum(Ycut); nGal = len(Ycut) - nStar
        nStarPred = np.sum(YpredCut); nGalPred = len(YpredCut) - nStarPred
        if nStar > 0:
            starCompleteness[i] = 1.0*np.sum(goodStar)/nStar
        else:
            starCompleteness[i] = 0.0
        if nStarPred > 0:
            starPurity[i] = 1.0*np.sum(goodStar)/nStarPred
        else:
            starPurity[i] = 0.0
        if nGal > 0:
            galCompleteness[i] = 1.0*np.sum(goodGal)/nGal
        else:
            galCompleteness[i] = 0.0
        if nGalPred > 0:
            galPurity[i] = 1.0*np.sum(goodGal)/nGalPred
        else:
            galPurity[i] = 0.0

    fig = plt.figure()
    axGal = plt.subplot(1, 2, 1)
    axStar = plt.subplot(1, 2, 2)
    if title is not None:
        axGal.set_title(title + " (Galaxies)", fontsize=18)
        axStar.set_title(title + " (Stars)", fontsize=18)
    else:
        axGal.set_title("Galaxies", fontsize=18)
        axStar.set_title("Stars", fontsize=18)
    if xlabel is None:
        axStar.set_xlabel("Magnitude", fontsize=18)
        axGal.set_xlabel("Magnitude", fontsize=18)
    else:
        axStar.set_xlabel(xlabel, fontsize=18)
        axGal.set_xlabel(xlabel, fontsize=18)

    axStar.set_ylabel("Scores", fontsize=18)
    axGal.set_ylabel("Scores", fontsize=18)
    axStar.set_ylim(0.0, 1.0)
    axGal.set_ylim(0.0, 1.0)
    for ax in [axStar, axGal]:
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(18)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(18)

    axStar.plot(magCuts, starCompleteness, 'r', label='Completeness')
    axStar.plot(magCuts, starPurity, 'b', label='Purity')
    axGal.plot(magCuts, galCompleteness, 'r', label='Completeness')
    axGal.plot(magCuts, galPurity, 'b', label='Purity')

    axStar.legend(loc='lower left', fontsize=18)
    axGal.legend(loc='lower left', fontsize=18)

    return fig

def makeRegaussPred(cat, band):
    mag = -2.5*np.log10(cat.get('cmodel.flux.'+band)/cat.get('flux.zeromag.'+band))
    Y = cat.get('stellar')
    res = cat.get('shape.hsm.regauss.resolution.' + band)
    resFlag = cat.get('shape.hsm.regauss.flags.' + band)
    zeroOut = np.logical_and(resFlag, np.isnan(res))
    res[zeroOut] = 0.0
    good = np.logical_and(np.isfinite(mag), np.isfinite(res))
    mag = mag[good]; Y = Y[good]; res = res[good]
    Ypred = np.zeros(Y.shape, dtype=bool)
    Ypred = res < 1.0/3
    return mag, Y, Ypred

def plotCMag(cat, fontSize=18):
    cat = afwTable.SourceCatalog.readFits(cat)
    magR = -2.5*np.log10(cat.get('cmodel.flux.r')/cat.get('flux.zeromag.r'))
    magG = -2.5*np.log10(cat.get('cmodel.flux.g')/cat.get('flux.zeromag.g'))
    extR = -2.5*np.log10(cat.get('flux.psf.r')/cat.get('cmodel.flux.r'))
    extG = -2.5*np.log10(cat.get('flux.psf.g')/cat.get('cmodel.flux.g'))
    stellar = cat.get('stellar')
    good = np.logical_and(np.isfinite(magR), np.isfinite(magG))
    good = np.logical_and(good, stellar)
    good = np.logical_and(good, extG < 5.0)
    good = np.logical_and(good, extR < 5.0)
    data = np.vstack((magG[good]-magR[good], magR[good]))
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(data)
    x = np.linspace(-1.0, 3.0, num=100)
    y = np.linspace(18.0, 27.0, num=100)
    X, Y = np.meshgrid(x, y)
    points = np.vstack((X.flatten(), Y.flatten()))
    Z = kde(points); Z = Z.reshape((100, 100))
    fig = plt.figure()
    plt.scatter(magG[good]-magR[good], magR[good], marker='.', s=1, color='black')
    plt.contour(X, Y, Z)
    ax = fig.get_axes()[0]
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontSize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontSize)
    ax.set_xlabel('g-r', fontsize=fontSize)
    ax.set_ylabel('r', fontsize=fontSize)
    ax.set_title('Point Sources in COSMOS', fontsize=fontSize)
    ax.set_xlim((-1.0, 2.0))
    ax.set_ylim((19.0, 27.0))
    plt.gca().invert_yaxis()
    return fig
