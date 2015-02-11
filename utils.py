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
