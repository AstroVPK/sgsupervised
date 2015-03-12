import numpy as np
import matplotlib.pyplot as plt

import lsst.afw.table as afwTable

def getGood(cat, band):
    if not isinstance(cat, afwTable.tableLib.SourceCatalog):
        cat = afwTable.SourceCatalog.readFits(cat)
    flux = cat.get('cmodel.flux.'+band)
    fluxPsf = cat.get('flux.psf.'+band)
    ext = -2.5*np.log10(fluxPsf/flux)
    good = np.logical_and(True, ext < 5.0)
    return good

def makeExtSeeingSnrPlot(cat, band, size=1, withLabels=False, fontSize=18):
    if not isinstance(cat, afwTable.tableLib.SourceCatalog):
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
    if not isinstance(cat, afwTable.tableLib.SourceCatalog):
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

def makeMagExPlot(cat, band, size=1, fontSize=18, withLabels=False):
    if not isinstance(cat, afwTable.tableLib.SourceCatalog):
        cat = afwTable.SourceCatalog.readFits(cat)
    flux = cat.get('cmodel.flux.'+band)
    fluxPsf = cat.get('flux.psf.'+band)
    fluxZero = cat.get('flux.zeromag.'+band)
    if withLabels:
        stellar = cat.get('stellar')
    mag = -2.5*np.log10(flux/fluxZero)
    ext = -2.5*np.log10(fluxPsf/flux)
    good = getGood(cat, band)

    fig = plt.figure()
    plt.xlabel('Magnitude HSC-'+band.upper(), fontsize=fontSize)
    plt.ylabel('Extendedness', fontsize=fontSize)
    plt.xlim((18.0, 28.0))
    plt.ylim((-0.1, 4.0))
    if withLabels:
        gals = np.logical_and(good, np.logical_not(stellar))
        stars = np.logical_and(good, stellar)
        plt.scatter(mag[gals], ext[gals], marker='.', s=size, color='red', label='Galaxies')
        plt.scatter(mag[stars], ext[stars], marker='.', s=size, color='blue', label='Stars')
    else:
        plt.scatter(mag[good], ext[good], marker='.', s=size)
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

def makeExtHist(cat, band, magCuts=None, nBins=100, fontSize=18, withLabels=False,
                nBinsStar=None, nBinsGal=None, normed=False):
    if not isinstance(cat, afwTable.tableLib.SourceCatalog):
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
        if nBinsStar is None:
            nBinsStar = nBins
        if nBinsGal is None:
            nBinsGal = nBins
    mag = -2.5*np.log10(flux/fluxZero)
    ext = -2.5*np.log10(fluxPsf/flux)
    good = getGood(cat, band)
    good = np.logical_and(good, ext <= 3.0)
    fig = plt.figure()
    for i in range(nRow*nColumn):
        magCut = magCuts[i]
        goodCut = np.logical_and(good, mag >= magCut[0])
        goodCut = np.logical_and(goodCut, mag <= magCut[1])
        ax = fig.add_subplot(nRow, nColumn, i+1)
        ax.set_xlabel('Extendedness', fontsize=fontSize)
        if normed:
            ax.set_ylabel('Probability Density', fontsize=fontSize)
        else:
            ax.set_ylabel('Object Counts', fontsize=fontSize)
        ax.set_title('{0} < Magnitude < {1}'.format(*magCut), fontsize=fontSize)
        if withLabels:
            gals = np.logical_and(goodCut, np.logical_not(stellar))
            stars = np.logical_and(goodCut, stellar)
            ax.hist(ext[gals], bins=nBinsGal, histtype='step', normed=normed, color='red', label='Galaxies')
            if normed:
                ylim = ax.get_ylim()
                ax.set_ylim(ylim)
            ax.hist(ext[stars], bins=nBinsStar, histtype='step', normed=normed, color='blue', label='Stars')
        else:
            ax.hist(ext[goodCut], bins=nBins, histtype='step', normed=normed, color='black')
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontSize)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontSize)
        if withLabels:
            ax.legend(loc=1, fontsize=18)
    return fig
