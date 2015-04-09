import numpy as np
import matplotlib.pyplot as plt

import lsst.afw.table as afwTable

import sgSVM as sgsvm

def getGood(cat, band):
    if not isinstance(cat, afwTable.tableLib.SourceCatalog) and\
       not isinstance(cat, afwTable.tableLib.SimpleCatalog):
        cat = afwTable.SourceCatalog.readFits(cat)
    flux = cat.get('cmodel.flux.'+band)
    fluxPsf = cat.get('flux.psf.'+band)
    ext = -2.5*np.log10(fluxPsf/flux)
    good = np.logical_and(True, ext < 5.0)
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
                     extMax=None):
    if not isinstance(cat, afwTable.tableLib.SourceCatalog) and\
       not isinstance(cat, afwTable.tableLib.SimpleCatalog):
        cat = afwTable.SourceCatalog.readFits(cat)
    fig = plt.figure()
    nRow, nColumn = _getExtHistLayout(len(bands))
    for i in range(nRow*nColumn):
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
            gals = np.logical_and(good, np.logical_not(stellar))
            stars = np.logical_and(good, stellar)
            for j, s in enumerate(seeingSet):
                sample = np.logical_and(True, seeing == s)
                sampleStar = np.logical_and(stars, sample); nStar = np.sum(sampleStar)
                sampleGal = np.logical_and(gals, sample); nGal = np.sum(sampleGal)
                meanStar[j] = np.mean(ext[sampleStar])
                meanGal[j] = np.mean(ext[sampleGal])
                stdStar[j] = np.std(ext[sampleStar])/np.sqrt(nStar-1)
                stdGal[j] = np.std(ext[sampleGal])/np.sqrt(nGal-1)
            ax.errorbar(np.array(list(seeingSet)), meanStar, yerr=stdStar, fmt='o', color='blue')
            ax.errorbar(np.array(list(seeingSet)), meanGal, yerr=stdGal, fmt='o', color='red')
    return fig

def makeMagExPlot(cat, band, size=1, fontSize=18, withLabels=False,
                  xlim=(17.5, 28.0), ylim=(-0.05, 0.5), trueSample=False,
                  frac=0.1, type='ext', data=None, xType='mag'):
    if not isinstance(cat, afwTable.tableLib.SourceCatalog) and\
       not isinstance(cat, afwTable.tableLib.SimpleCatalog):
        cat = afwTable.SourceCatalog.readFits(cat)
    fig = plt.figure()
    if isinstance(band, list) or isinstance(band, tuple):
        bands = band
        nRow, nColumn = _getExtHistLayout(len(band))
        for i in range(nRow*nColumn):
            ax = fig.add_subplot(nRow, nColumn, i+1)
            band = bands[i]
            flux = cat.get('cmodel.flux.'+band)
            fluxPsf = cat.get('flux.psf.'+band)
            fluxZero = cat.get('flux.zeromag.'+band)
            if withLabels:
                stellar = cat.get('stellar')
            if xType == 'mag':
                mag = -2.5*np.log10(flux/fluxZero)
            elif xType == 'seeing':
                mag = cat.get('seeing.'+band)
            else:
                raise ValueError("I don't recognize the xType value")
            ext = -2.5*np.log10(fluxPsf/flux)
            good = getGood(cat, band)
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
                        else:
                            ax.plot(mag[i], data[i], marker='.', markersize=size, color='red')
                else:
                    ax.scatter(mag[stars], data[stars], marker='.', s=size, color='blue', label='Stars')
                    ax.scatter(mag[gals], data[gals], marker='.', s=size, color='red', label='Galaxies')
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
        stellar = cat.get('stellar')
    mag = -2.5*np.log10(flux/fluxZero)
    ext = -2.5*np.log10(fluxPsf/flux)
    good = getGood(cat, band)
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
            plt.scatter(mag[stars], data[stars], marker='.', s=size, color='blue', label='Stars')
            plt.scatter(mag[gals], data[gals], marker='.', s=size, color='red', label='Galaxies')
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
    if nCuts == 2:
        return 1, 2
    elif nCuts == 3:
        return 1, 3
    elif nCuts == 4:
        return 2, 2
    else:
        raise ValueError("Using more than 4 cuts is not implemented")


def makeExtHist(cat, band, magCuts=None, nBins=100, fontSize=14, withLabels=False,
                normed=False, xlim=None, type='ext', data=None):
    if not isinstance(cat, afwTable.tableLib.SourceCatalog) and\
       not isinstance(cat, afwTable.tableLib.SimpleCatalog):
        cat = afwTable.SourceCatalog.readFits(cat)
    if magCuts is None:
        magCuts = [(23.0, 24.0)]
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
        stellar = cat.get('stellar')
    mag = -2.5*np.log10(flux/fluxZero)
    ext = -2.5*np.log10(fluxPsf/flux)
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
    fig = plt.figure()
    for i in range(nRow*nColumn):
        magCut = magCuts[i]
        goodCut = np.logical_and(good, mag >= magCut[0])
        goodCut = np.logical_and(goodCut, mag <= magCut[1])
        ax = fig.add_subplot(nRow, nColumn, i+1)
        ax.set_xlabel('Extendedness HSC-' + band.upper(), fontsize=fontSize)
        if xlim is not None:
            ax.set_xlim(xlim)
        if normed:
            ax.set_ylabel('Probability Density', fontsize=fontSize)
        else:
            ax.set_ylabel('Object Counts', fontsize=fontSize)
        ax.set_title('{0} < Magnitude < {1}'.format(*magCut), fontsize=fontSize)
        if withLabels:
            # Make sure the same binning is being used to make meaningful comparisons
            hist, bins = np.histogram(data[goodCut], bins=nBins, range=xlim)
            gals = np.logical_and(goodCut, np.logical_not(stellar))
            stars = np.logical_and(goodCut, stellar)
            ax.hist(data[stars], bins=bins, histtype='step', normed=normed, color='blue', label='Stars')
            if normed:
                ylim = ax.get_ylim()
                ax.set_ylim(ylim)
            ax.hist(data[gals], bins=bins, histtype='step', normed=normed, color='red', label='Galaxies')
        else:
            ax.hist(data[goodCut], bins=nBins, histtype='step', normed=normed, color='black')
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontSize)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontSize)
        if withLabels:
            ax.legend(loc=1, fontsize=fontSize)
    return fig
