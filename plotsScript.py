import os
import pickle
import re
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz
import pyfits
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors.kde import KernelDensity
from astroML.plotting.tools import draw_ellipse

import supervisedEtl as etl
import dGauss

import lsst.afw.table as afwTable
from lsst.pex.exceptions import LsstCppException

import utils

import sgSVM as sgsvm

def cutsPlots():
    fontSize = 18
    magRange = (18.0, 26.0)
    nBins = 30
    cuts = [0.001, 0.01, 0.02]
    style = ['--', '-', ':']
    with open('trainSet.pkl', 'rb') as f:
        trainSet = pickle.load(f)
    X, Y = trainSet.getTrainSet()
    clf = LinearSVC()
    clf.fit(X, Y)
    training = etl.Training(trainSet, clf)
    training.setPhysicalCut(cuts[0])
    figScores = training.plotScores(nBins=nBins, magRange=magRange, linestyle=style[0], legendLabel='Cut={0}'.format(cuts[0]), fontSize=fontSize)
    for i in range(1, len(cuts)):
        training.setPhysicalCut(cuts[i])
        figScores = training.plotScores(nBins=nBins, magRange=magRange, fig=figScores, linestyle=style[i],\
                                        legendLabel='Cut={0}'.format(cuts[i]), fontSize=fontSize)
    for ax in figScores.get_axes():
        ax.legend(loc='lower left', fontsize=fontSize)
    figExtMag = plt.figure()
    X, Y = trainSet.getTestSet(standardized=False); mags = trainSet.getTestMags()
    for i in range(len(mags)):
        if Y[i]:
            plt.plot(mags[i], X[i], marker='.', markersize=1, color='blue')
        else:
            plt.plot(mags[i], X[i], marker='.', markersize=1, color='red')
    for i in range(len(cuts)):
        plt.plot((18.0, 26.0), (cuts[i], cuts[i]), linestyle=style[i], color='black', linewidth=2)
    for ax in figExtMag.get_axes():
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontSize)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontSize)
    plt.xlabel('Magnitude', fontsize=fontSize)
    plt.ylabel('Mag_psf-Mag_model', fontsize=fontSize)
    plt.ylim((-0.01, 0.1))
    plt.xlim((18.0, 26.0))
    plt.show()

def plotScores(mags, Y, YProb, threshold=0.5, linestyle='-', fig=None):
    magsBins = np.linspace(18.0, 26.0, num=30)
    magsCenters = 0.5*(magsBins[:-1] + magsBins[1:])
    complStars = np.zeros(magsCenters.shape)
    purityStars = np.zeros(magsCenters.shape)
    complGals = np.zeros(magsCenters.shape)
    purityGals = np.zeros(magsCenters.shape)
    YPred = np.logical_not(YProb < threshold)
    for i in range(len(magsCenters)):
        magCut = np.logical_and(mags > magsBins[i], mags < magsBins[i+1])
        predCut = YPred[magCut]; truthCut = Y[magCut]
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

    legendLabel = 'P(Star)={0}'.format(threshold)
    if fig is None:
        fontSize = 18
        xlabel = 'Magnitude'
        ylabel = 'Scores'
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
        for ax in fig.get_axes():
            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(fontSize)
            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(fontSize)
    else:
        axGal, axStar = fig.get_axes()

    axGal.step(magsCenters, complGals, color='red', linestyle=linestyle, label=legendLabel + ' Completeness')
    axGal.step(magsCenters, purityGals, color='blue', linestyle=linestyle, label=legendLabel + ' Purity')
    axStar.step(magsCenters, complStars, color='red', linestyle=linestyle, label=legendLabel + ' Completeness')
    axStar.step(magsCenters, purityStars, color='blue', linestyle=linestyle, label=legendLabel + ' Purity')

    return fig

def plotPosterior(X, posteriors, catType='hsc', magBin=(18.0, 26.0), fontSize=14):
    fig = plt.figure()
    fig.suptitle('{0} < i < {1}'.format(magBin[0], magBin[1]), fontsize=16)
    if catType == 'hsc':
        colors = ['g-r', 'r-i', 'i-z', 'z-y']
    elif catType == 'sdss':
        colors = ['u-g', 'g-r', 'r-i', 'i-z']

    xlims = [(-0.5, 2.0), (-0.5, 2.0), (-0.5, 2.0), (-0.3, 2.5), (-0.3, 2.5), (-0.3, 1.2)]
    ylims = [(-0.5, 2.5), (-0.2, 1.2), (-0.2, 0.4), (-0.2, 1.2), (-0.2, 0.4), (-0.2, 0.4)]
    cPairs= [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    #cbar_ax = fig.add_axes([0.91, 0.1, 0.02, 0.8])
    for n, (j, i) in enumerate(cPairs):
        ax = plt.subplot(2, 3, n+1)
        ax.locator_params(axis='x', nbins=4)
        ax.set_xlabel(colors[j], fontsize=fontSize)
        ax.set_ylabel(colors[i], fontsize=fontSize)
        ax.set_xlim(xlims[n]); ax.set_ylim(ylims[n])
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontSize-2)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontSize-2)
        #sc = ax.scatter(X[:,j], X[:,i], c=posteriors, marker=".", s=5, edgecolors="none")
        sc = ax.scatter(X[:,j], X[:,i], marker=".", s=5, edgecolors="none")
    #cb = fig.colorbar(sc, cax=cbar_ax, use_gridspec=True)
    #cb.set_label('Star Posterior', fontsize=fontSize)
    #cb.ax.tick_params(labelsize=fontSize)
    #plt.tight_layout()
    plt.subplots_adjust(right=0.85)

def colExPlots():
    with open('trainSet.pkl', 'rb') as f:
        trainSet = pickle.load(f)
    fontSize = 18
    magBins = [(18.0, 22.0), (22.0, 24.0), (24.0, 25.0), (25.0, 26.0)]
    gaussians = [(10, 10), (10, 10), (10, 10), (10, 10)]
    X, XErr, Y = trainSet.getTrainSet(standardized=False)
    mags = trainSet.getTrainMags()
    X = X[:5000]; XErr[:5000]; Y = Y[:5000]; mags = mags[:5000]
    clfs = []
    for i, magBin in enumerate(magBins):
        good = np.logical_and(magBin[0] < mags, mags < magBin[1])
        ngStar, ngGal = gaussians[i]
        clf = dGauss.XDClf(ngStar=ngStar, ngGal=ngGal)
        clf.fit(X[good], XErr[good], Y[good])
        clfs.append(clf)
        #YPred = clf.predict(X[good], XErr[good])
        #plotPosterior(X[good][Y[good]], YPred[good][Y[good]])
    #X, XErr, Y = trainSet.getTestSet(standardized=False)
    #mags = trainSet.getTestMags()
    X, XErr, Y = trainSet.getTrainSet(standardized=False)
    mags = trainSet.getTrainMags()
    X = X[:300000]; XErr[:300000]; Y = Y[:300000]; mags = mags[:300000]
    YProb = np.zeros(Y.shape)
    YPred = np.zeros(Y.shape, dtype=bool)
    for i, magBin in enumerate(magBins):
        good = np.logical_and(magBin[0] < mags, mags < magBin[1])
        YProb[good] = clfs[i].predict_proba(X[good], XErr[good])
        YPred[good] = clfs[i].predict(X[good], XErr[good])
        mpl.rcParams['figure.figsize'] = 16, 10
        plotPosterior(X[good][Y[good]], YPred[good][Y[good]], magBin=magBin)
        #plt.tight_layout()
        #plotPosterior(X[good][np.logical_not(Y[good])], YPred[good][np.logical_not(Y[good])])
        plt.savefig('/u/garmilla/Desktop/colorColorStars{0}-{1}.png'.format(magBin[0], magBin[1]), bbox_inches='tight')
        #plt.savefig('/u/garmilla/Desktop/colorColorGalaxies{0}-{1}.png'.format(magBin[0], magBin[1]), bbox_inches='tight')
    print "Score={0}".format(np.sum(YPred == Y)*1.0/len(Y))


    figScores = plotScores(mags, Y, YProb)
    figScores = plotScores(mags, Y, YProb, threshold=0.1, fig=figScores, linestyle='--')
    figScores = plotScores(mags, Y, YProb, threshold=0.9, fig=figScores, linestyle=':')
    for ax in figScores.get_axes():
        ax.legend(loc='lower left', fontsize=fontSize)

    fig = plt.figure()
    for i in range(len(mags)):
        if Y[i]:
            plt.plot(mags[i], YProb[i], marker='.', markersize=1, color='blue')
        else:
            plt.plot(mags[i], YProb[i], marker='.', markersize=1, color='red')
    plt.xlabel('Magnitude', fontsize=fontSize)
    plt.ylabel('P(Star)', fontsize=fontSize)
    plt.xlim((18.0, 26.0))
    plt.ylim((0.0, 1.0))
    for ax in fig.get_axes():
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontSize)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontSize)

    #plt.show()

cgr = (-0.00816446, -0.08366937, -0.00726883)
cri = (0.00231810,  0.01284177, -0.03068248)
ciz = (0.00130204, -0.16922042, -0.01374245)
czi = (-0.00680620,  0.01353969,  0.01479369)
Aiz = ciz[2] - czi[2]
Biz = 1.0 + ciz[1] + czi[1]
Ari = cri[2]
Bri = 1.0 + cri[1]
Agr = cgr[2]
Bgr = 1.0 + cgr[1]

def _fromHscToSdss(grHsc, riHsc, izHsc, giveClosest=True):
    Ciz = ciz[0] - czi[0] - izHsc
    izSdss1 = (-Biz + np.sqrt(Biz**2-4*Aiz*Ciz))/2/Aiz
    izSdss2 = (-Biz - np.sqrt(Biz**2-4*Aiz*Ciz))/2/Aiz
    Cri1 = cri[0] - ciz[0] - ciz[1]*izSdss1 - ciz[2]*izSdss1**2 - riHsc
    Cri2 = cri[0] - ciz[0] - ciz[1]*izSdss2 - ciz[2]*izSdss2**2 - riHsc
    riSdss1 = (-Bri + np.sqrt(Bri**2-4*Ari*Cri1))/2/Ari
    riSdss2 = (-Bri - np.sqrt(Bri**2-4*Ari*Cri1))/2/Ari
    riSdss3 = (-Bri + np.sqrt(Bri**2-4*Ari*Cri2))/2/Ari
    riSdss4 = (-Bri - np.sqrt(Bri**2-4*Ari*Cri2))/2/Ari
    riStack = np.vstack((riSdss1, riSdss2, riSdss3, riSdss4))
    izStack = np.vstack((izSdss1, izSdss1, izSdss2, izSdss2))
    d1 = np.square(riSdss1 - riHsc) + np.square(izSdss1 - izHsc)
    d2 = np.square(riSdss2 - riHsc) + np.square(izSdss1 - izHsc)
    d3 = np.square(riSdss3 - riHsc) + np.square(izSdss2 - izHsc)
    d4 = np.square(riSdss4 - riHsc) + np.square(izSdss2 - izHsc)
    dStack = np.vstack((d1, d2, d3, d4))
    idxMinD = np.argmin(dStack, axis=0)
    idxArange = np.arange(len(riHsc))
    riSdss = riStack[idxMinD, idxArange]
    izSdss = izStack[idxMinD, idxArange]
    Cgr = cgr[0] - cri[0] - cri[1]*riSdss - cri[2]*riSdss1**2 - grHsc
    grSdss1 = (-Bgr + np.sqrt(Bgr**2 - 4*Agr*Cgr))/2/Agr
    grSdss2 = (-Bgr - np.sqrt(Bgr**2 - 4*Agr*Cgr))/2/Agr
    grStack = np.vstack((grSdss1, grSdss2))
    d1 = np.square(grSdss1 - grHsc)
    d2 = np.square(grSdss2 - grHsc)
    dStack = np.vstack((d1, d2))
    idxMinD = np.argmin(dStack, axis=0)
    grSdss = grStack[idxMinD, idxArange]
    return grSdss, riSdss, izSdss

def _fromSdssToHsc(gSdss, rSdss, iSdss, zSdss):
    grSdss = gSdss - rSdss
    riSdss = rSdss - iSdss
    izSdss = iSdss - zSdss
    ziSdss = zSdss - iSdss
    gHsc = gSdss + cgr[0] + cgr[1]*grSdss + cgr[2]*grSdss**2
    rHsc = rSdss + cri[0] + cri[1]*riSdss + cri[2]*riSdss**2
    iHsc = iSdss + ciz[0] + ciz[1]*izSdss + ciz[2]*izSdss**2
    zHsc = zSdss + czi[0] + czi[1]*ziSdss + czi[2]*ziSdss**2
    return gHsc, rHsc, iHsc, zHsc

def _getAbsoluteMagR(riSdss):
    return 4.0 + 11.86*riSdss - 10.74*riSdss**2 + 5.99*riSdss**3 - 1.20*riSdss**4

def _getPColors(g, r, i):
    P1 = np.zeros(g.shape)
    P2 = np.zeros(g.shape)
    As = np.zeros((g.shape[0], 3, 3))
    Bs = np.zeros((g.shape[0], 3))
    isW = np.zeros(g.shape, dtype=bool)
    P1w = 0.928*g - 0.556*r - 0.372*i - 0.425
    P2w = -0.227*g + 0.792*r -0.567*i + 0.050
    isInW = np.logical_and(P1w > -0.2, P1w < 0.6)
    P1[isInW] = P1w[isInW]
    P2[isInW] = P2w[isInW]
    isW[isInW] = True
    P1x = r - i
    P2x = 0.707*g - 0.707*r - 0.988
    isInX = np.logical_and(P1x > 0.8, P1x < 1.6)
    P1[isInX] = P1x[isInX]
    P2[isInX] = P2x[isInX]
    isW[isInX] = False
    if np.any(np.logical_and(isInW, isInX)):
        both = np.logical_and(isInW, isInX)
        bothW = np.logical_and(both, P2w**2 < P2x**2)
        P1[bothW] = P1w[bothW]
        P2[bothW] = P2w[bothW]
        isW[bothW] = True
    if np.any(np.logical_and(np.logical_not(isInW), np.logical_not(isInX))):
        isInNan = np.logical_and(np.logical_not(isInW), np.logical_not(isInX))
        isInNanW = np.logical_and(isInNan, P2w**2 < P2x**2)
        isInNanX = np.logical_and(isInNan, P2x**2 <= P2w**2)
        P1[isInNanW] = P1w[isInNanW]
        P2[isInNanW] = P2w[isInNanW]
        isW[isInNanW] = True
        P1[isInNanX] = P1x[isInNanX]
        P2[isInNanX] = P2x[isInNanX]
        isW[isInNanX] = False
        #isNotInNan = np.logical_not(isInNan)
        #plt.scatter(g[isNotInNan] - i[isNotInNan], r[isNotInNan] - i[isNotInNan], marker='.', s=1, color='blue')
        #plt.scatter(g[isInNan] - i[isInNan], r[isInNan] - i[isInNan], marker='.', s=1, color='red')
        #plt.show()
        #P1[isInNan] = np.nan
        #P2[isInNan] = np.nan
        #raise ValueError("I've found an object that is no regime!")
    isX = np.logical_not(isW)
    As[isW] = np.array([[0.928, -0.556, -0.372],
                        [-0.227, 0.792, -0.567],
                        [0.0, 0.0, 1.0]])
    Bs[isW, 0] = P1[isW] + 0.425; Bs[isW, 1] = 0.0 - 0.050; Bs[isW, 2] = i[isW]
    As[isX] = np.array([[0.0, 1.0, -1.0],
                        [0.707, -0.707, 0.0],
                        [0.0, 0.0, 1.0]])
    Bs[isX, 0] = P1[isX]; Bs[isX, 1] = 0.0 + 0.988; Bs[isX, 2] = i[isX]
    gris = np.linalg.solve(As, Bs)
    grProj = gris[:,0] - gris[:,1]
    riProj = gris[:,1] - gris[:,2]
    return P1, P2, grProj, riProj

def _getMsGrSdss(ri):
    return 1.39*(1.0 - np.exp(-4.9*ri**3 - 2.45*ri**2 -1.68*ri - 0.050))

def _getMsGrHsc(ri, A, B):
    #return A*(1.0 - np.exp(B*ri**3 + C*ri**2 + D*ri + E))
    return A + B*ri

def _getMsIzHsc(ri, A, B):
    #return A + B*ri + C*ri**2 + D*ri**3
    return A + B*ri

def _fitGriSlHsc(gr, ri, sigma=None):
    #popt, pcov = curve_fit(_getMsGrHsc, ri, gr, p0=(1.39, -4.9, -2.45, -1.68, -0.050), sigma=sigma)
    popt, pcov = curve_fit(_getMsGrHsc, ri, gr, p0=(0.0, 2.0), sigma=sigma)
    return popt, pcov

def _fitRizSlHsc(ri, iz, sigma=None):
    #popt, pcov = curve_fit(_getMsIzHsc, ri, iz, p0=(0.0, 0.5, 0.0, 0.0), sigma=sigma)
    popt, pcov = curve_fit(_getMsGrHsc, ri, iz, p0=(0.0, 0.5), sigma=sigma)
    return popt, pcov

def _loadCKData(stringZ, stringT):
    data = pyfits.getdata('/u/garmilla/Data/castelli_kurucz/ck{0}/ck{0}_{1}.fits'.format(stringZ, stringT))
    return data

def _loadFilter(band):
    if band == 'y':
        band = band.upper()
    filt = np.loadtxt('/u/garmilla/Data/HSC/HSC-{0}.dat'.format(band))
    filt[:,0] = filt[:,0]*10
    filt[:,0] = np.sort(filt[:,0])
    return filt

def _regridSed(lamSed, fSed, lamFilt):
    interpSed = interp1d(lamSed, fSed, kind='linear')
    fSedRegrid = np.zeros(lamFilt.shape)
    for i in range(len(fSedRegrid)):
        fSedRegrid[i] = interpSed(lamFilt[i])
    return fSedRegrid

def computeColor(stringZ, stringT, stringG, color):
    dataCK = _loadCKData(stringZ, stringT)
    bandBlue = color[0]
    bandRed = color[2]
    filterBlue = _loadFilter(bandBlue)
    filterRed = _loadFilter(bandRed)
    sedL = dataCK.field('WAVELENGTH')
    sedF = dataCK.field(stringG)
    sedBlue = _regridSed(sedL, sedF, filterBlue[:,0])
    sedRed = _regridSed(sedL, sedF, filterRed[:,0])
    yBlue = filterBlue[:,0]*filterBlue[:,1]*sedBlue
    yRed = filterRed[:,0]*filterRed[:,1]*sedRed
    fluxBlue = cumtrapz(yBlue, x=filterBlue[:,0])[-1]
    fluxRed = cumtrapz(yRed, x=filterRed[:,0])[-1]
    return -2.5*np.log10(fluxBlue/fluxRed)

def plotIsochrones(fontSize=18):
    colors = ['g-r', 'r-i', 'i-z', 'z-y']
    colLimX = [(0.1, 2.0), (0.0, 1.3), (-0.05, 0.6), (-0.04 ,0.30)]
    colLimY = [(3.0, 15.0), (3.0, 14.0), (3.0, 13.0), (3.0, 12.0)]
    iReaderHaloIn = etl.IsochroneReader(stringZ='m15', stringA='p2')
    iReaderHaloOut = etl.IsochroneReader(stringZ='m20', stringA='p2')
    iReaderDiskThick = etl.IsochroneReader(stringZ='m05', stringA='p2')
    iReaderDiskThin = etl.IsochroneReader(stringZ='p00', stringA='p0')
    fig = plt.figure(figsize=(16, 12), dpi=120)
    for i, color in enumerate(colors):
        ax = fig.add_subplot(2, 2, i+1)
        ax.set_xlabel(color, fontsize=fontSize)
        ax.set_ylabel(color[0] + ' (Absolute Magnitude)', fontsize=fontSize)
        ax.set_xlim(colLimX[i])
        ax.set_ylim(colLimY[i])
        bandBlue = 'LSST_' + color[0]
        bandRed = 'LSST_' + color[2]
        ax.plot(iReaderHaloOut.isochrones[10.0][bandBlue]-iReaderHaloOut.isochrones[10.0][bandRed], iReaderHaloOut.isochrones[10.0][bandBlue],
                linestyle='-', color='black', label=r'Outer Halo')
        ax.plot(iReaderHaloIn.isochrones[10.0][bandBlue]-iReaderHaloIn.isochrones[10.0][bandRed], iReaderHaloIn.isochrones[10.0][bandBlue],
                linestyle='--', color='black', label=r'Inner Halo')
        ax.plot(iReaderDiskThick.isochrones[10.0][bandBlue]-iReaderDiskThick.isochrones[10.0][bandRed], iReaderDiskThick.isochrones[10.0][bandBlue],
                linestyle=':', color='black', label=r'Thick Disk')
        ax.plot(iReaderDiskThin.isochrones[10.0][bandBlue]-iReaderDiskThin.isochrones[10.0][bandRed], iReaderDiskThin.isochrones[10.0][bandBlue],
                linestyle='-.', color='black', label=r'Solar')
        ax.invert_yaxis()
        ax.legend(loc='upper right')
    for ax in fig.get_axes():
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontSize)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontSize)
    dirHome = os.path.expanduser('~')
    fileFig = os.path.join(dirHome, 'Desktop/isochrones.png')
    fig.savefig(fileFig, dpi=120, bbox_inches='tight')
    return fig

def plotCmdPhotoParallax(trainClfs=False, fontSize=18):
    magBins = [(18.0, 22.0), (22.0, 24.0), (24.0, 25.0), (25.0, 26.0)]
    with open('trainSet.pkl', 'rb') as f:
        trainSet = pickle.load(f)
    with open('clfsColsExt.pkl', 'rb') as f:
        clfs = pickle.load(f)
    clfXd = dGauss.XDClfs(clfs=clfs, magBins=magBins)
    mags = trainSet.getAllMags(band='i')
    magsR = trainSet.getAllMags(band='r')
    X, XErr, Y = trainSet.genColExtTrainSet(mode='all')
    posteriors = clfXd.predict_proba(X, XErr, mags)
    fig = plt.figure(dpi=120)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim((-0.10, 1.9))
    ax.set_ylim((27.5, 19.0))
    ax.set_xlabel('r-i', fontsize=fontSize)
    ax.set_ylabel('r', fontsize=fontSize)
    plt.plot([0.4, 0.4], [19.0, 24.0], color='black', linestyle='--')
    plt.plot([-0.10, 0.4], [24.0, 24.0], color='black', linestyle='--')
    sc = ax.scatter(X[:,1][Y], magsR[Y], c=posteriors[Y], marker=".", s=5, edgecolors="none")
    cb = plt.colorbar(sc)
    cb.set_label(r'$P(\mathrm{Star})$', fontsize=fontSize)
    for ax in fig.get_axes():
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontSize)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontSize)
    dirHome = os.path.expanduser('~')
    fileFig = os.path.join(dirHome, 'Desktop/cmdParallax.png')
    fig.savefig(fileFig, dpi=120, bbox_inches='tight')
    return fig

def plotCKModels(colorX='g-r', colorY = 'r-i', zs=['m25', 'p00'],
                 gs=['g30', 'g35', 'g40', 'g45', 'g50'], ts='all', markersZ=['o', 'v'],
                 labelsZ=[r'[M/H]=-2.5', r'[M/H]=0.0'], fontSize=18):
    assert isinstance(zs, list)
    assert isinstance(gs, list)
    assert isinstance(markersZ, list)
    assert len(markersZ) == len(zs)
    if ts != 'all':
        assert isinstance(ts, list)
    fig = plt.figure(dpi=120)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel(colorX, fontsize=fontSize)
    ax.set_ylabel(colorY, fontsize=fontSize)
    for ax in fig.get_axes():
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontSize)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontSize)
    for i, stringZ in enumerate(zs):
        if ts == 'all':
            fList = os.listdir('/u/garmilla/Data/castelli_kurucz/ck{0}/'.format(stringZ))
            tList = []
            for f in fList:
                match = re.match(r"ck[mp][0-9][0-9]_([0-9]*).fits", f)
                if match is not None:
                    tList.append(match.group(1))
        else:
            tList = ts
        for j, stringT in enumerate(tList):
            for k, stringG in enumerate(gs):
                cX = computeColor(stringZ, stringT, stringG, colorX)
                cY = computeColor(stringZ, stringT, stringG, colorY)
                if j == 0 and k == 0:
                    ax.plot(cX, cY, marker=markersZ[i], color='black', label=labelsZ[i])
                else:
                    ax.plot(cX, cY, marker=markersZ[i], color='black')
    ax.legend(loc='upper left', fontsize=fontSize)
    return fig

def makeIsoDensityPlot(xData, yData, xRange, yRange, bandwidth=0.1, xlabel=None, ylabel=None,
                       printMaxDens=False, levels=None):
    values = np.vstack([xData, yData]).T
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(values)
    xx, yy = np.meshgrid(np.linspace(*xRange, num=100), np.linspace(*yRange, num=100))
    positions = np.vstack([xx.ravel(), yy.ravel()]).T
    zz = np.reshape(np.exp(kde.score_samples(positions)), xx.shape)
    if printMaxDens:
        print "maxDens={0}".format(zz.max())

    fig = plt.figure(figsize=(16, 6), dpi=120)
    axData = fig.add_subplot(1, 2, 1)
    axData.scatter(xData, yData, marker='.', s=1, color='black')
    axData.set_xlim(xRange)
    axData.set_ylim(yRange)
    if xlabel is not None:
        axData.set_xlabel(xlabel)
    if ylabel is not None:
        axData.set_ylabel(ylabel)
    axContour = fig.add_subplot(1, 2, 2)
    if levels is None:
        ctr = axContour.contour(xx, yy, zz)
    else:
        ctr = axContour.contour(xx, yy, zz, levels=levels)
    if printMaxDens:
        plt.colorbar(ctr)
    axContour.set_xlim(xRange)
    axContour.set_ylim(yRange)
    if xlabel is not None:
        axContour.set_xlabel(xlabel)
    if ylabel is not None:
        axContour.set_ylabel(ylabel)
    return fig

def _makeIsoDensityPlot(ri, gr=None, iz=None, withHsc=False, paramTuple=None, minDens=None, sigma=None,
                        cutRi=None, cutGr=None, cutIz=None, fontSize=18):
    if gr is None and iz is None or\
       gr is not None and iz is not None:
        raise ValueError("You need to provide one, and only one, of these two colors: g-r, and i-z.")
    if iz is None:
        values = np.vstack([gr, ri]).T
        kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(values)
        xx, yy = np.meshgrid(np.linspace(-0.05, 1.7, num=100), np.linspace(-0.05, 2.5, num=100))
    else:
        values = np.vstack([ri, iz]).T
        kde = KernelDensity(kernel='gaussian', bandwidth=0.05).fit(values)
        xx, yy = np.meshgrid(np.linspace(-0.05, 2.5, num=100), np.linspace(-0.05, 1.2, num=100))
    positions = np.vstack([xx.ravel(), yy.ravel()]).T
    zz = np.reshape(np.exp(kde.score_samples(positions)), xx.shape)

    if cutRi is None:
       riSl = np.linspace(-0.05, 2.5, num=100)
    else:
       riSl = np.linspace(-0.05, cutRi, num=100)
    if withHsc:
        if paramTuple is None:
            assert minDens is not None
            densValues = np.exp(kde.score_samples(values))
            good = np.logical_and(True, densValues > minDens)
            if cutRi is not None:
                good = np.logical_and(good, ri < cutRi)
            if cutGr is not None:
                good = np.logical_and(good, gr < cutGr)
            if cutIz is not None:
                good = np.logical_and(good, iz < cutIz)
            if sigma is not None:
                sigma = sigma[good]
            if iz is None:
                popt, pcov = _fitGriSlHsc(gr[good], ri[good], sigma=sigma)
                popt = (0.15785242, 1.93645872)
            else:
                popt, pcov = _fitRizSlHsc(ri[good], iz[good], sigma=sigma)
                popt = (-0.0207809, 0.5644657)
            print popt
            paramTuple = popt
        if iz is None:
            grSl = _getMsGrHsc(riSl, *paramTuple)
        else:
            izSl = _getMsIzHsc(riSl, *paramTuple)
    else:
        good = np.ones(ri.shape, dtype=bool)
        if iz is None:
            grSl = _getMsGrSdss(riSl)
        else:
            pass
            #raise ValueError("I don't have a riz fit for SDSS stars. Fit to HSC stars instead.")

    fig = plt.figure(figsize=(16, 6), dpi=120)
    axData = fig.add_subplot(1, 2, 1)
    if iz is None:
        axData.scatter(gr[good], ri[good], marker='.', s=1, color='blue')
        axData.scatter(gr[np.logical_not(good)], ri[np.logical_not(good)], marker='.', s=1, color='blue')
        axData.set_xlim((-0.05, 1.7))
        axData.set_ylim((-0.05, 2.5))
        axData.set_xlabel('g-r', fontsize=fontSize)
        axData.set_ylabel('r-i', fontsize=fontSize)
        axData.plot(grSl, riSl, color='black')
        if cutRi is not None:
            axData.plot([-0.1, 2.0], [cutRi, cutRi], color='black', linestyle='--')
    else:
        axData.scatter(ri[good], iz[good], marker='.', s=1, color='blue')
        axData.scatter(ri[np.logical_not(good)], iz[np.logical_not(good)], marker='.', s=1, color='blue')
        axData.set_xlim((-0.05, 2.5))
        axData.set_ylim((-0.05, 1.2))
        axData.set_xlabel('r-i', fontsize=fontSize)
        axData.set_ylabel('i-z', fontsize=fontSize)
        try:
            axData.plot(riSl, izSl, color='black')
        except UnboundLocalError:
            pass
        if cutRi is not None and cutIz is not None:
            axData.plot([cutRi, cutRi], [-0.1, cutIz], color='black', linestyle='--')
            axData.plot([-0.1, cutRi], [cutIz, cutIz], color='black', linestyle='--')
    axContour = fig.add_subplot(1, 2, 2)
    print "Maximum contour value is {0}".format(zz.max())
    if iz is None:
        ctr = axContour.contour(xx, yy, zz, levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6])
    else:
        ctr = axContour.contour(xx, yy, zz, levels=[0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.10, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2])
    #plt.colorbar(ctr)
    if iz is None:
        axContour.set_xlim((-0.05, 1.7))
        axContour.set_ylim((-0.05, 2.5))
        axContour.plot(grSl, riSl, color='black')
        if cutRi is not None:
            axContour.plot([-0.1, 2.0], [cutRi, cutRi], color='black', linestyle='--')
        axContour.set_xlabel('g-r', fontsize=fontSize)
        axContour.set_ylabel('r-i', fontsize=fontSize)
    else:
        axContour.set_xlim((-0.05, 2.5))
        axContour.set_ylim((-0.05, 1.2))
        try:
            axContour.plot(riSl, izSl, color='black')
        except UnboundLocalError:
            pass
        if cutRi is not None and cutIz is not None:
            axContour.plot([cutRi, cutRi], [-0.1, cutIz], color='black', linestyle='--')
            axContour.plot([-0.1, cutRi], [cutIz, cutIz], color='black', linestyle='--')
        axContour.set_xlabel('r-i', fontsize=fontSize)
        axContour.set_ylabel('i-z', fontsize=fontSize)
    for ax in fig.get_axes():
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontSize)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontSize)
    return fig

def makePhotParallaxPlots(fontSize=18):
    with open('trainSet.pkl', 'rb') as f:
        trainSet = pickle.load(f)
    X, XErr, Y = trainSet.getAllSet(standardized=False)
    mags = trainSet.getAllMags(band='i')
    good = np.logical_and(Y, mags < 24.0)
    gr = X[:,0][good]
    ri = X[:,1][good]
    iz = X[:,2][good]
    figGri = _makeIsoDensityPlot(ri, gr=gr, withHsc=True, minDens=0.0, cutRi=0.4)
    figRiz = _makeIsoDensityPlot(ri, iz=iz, withHsc=True, minDens=0.0, cutIz=0.2, cutRi=0.4)
    #paramsGri = (1.30038049, -7.78059699, -0.71791215, -0.76761088, -0.19133522)
    #paramsRiz = (-0.01068287, 0.59929634, -0.19457149, 0.05357661)
    paramsGri = (0.15785242, 1.93645872)
    paramsRiz = (-0.0207809, 0.5644657)
    riSl = np.linspace(-0.05, 0.4, num=100)
    grSl = _getMsGrHsc(riSl, *paramsGri)
    izSl = _getMsIzHsc(riSl, *paramsRiz)
    grSdss, riSdss, izSdss = _fromHscToSdss(grSl, riSl, izSl, giveClosest=True)
    absMagRSdss = _getAbsoluteMagR(riSdss)
    absMagRHsc = absMagRSdss + cri[0] + cri[1]*riSdss + cri[2]*riSdss**2
    absMagGHsc = absMagRHsc + grSl
    absMagIHsc = absMagRHsc - riSl
    fig = plt.figure(figsize=(16, 6), dpi=120)
    axGr = fig.add_subplot(1, 2, 1)
    axRi = fig.add_subplot(1, 2, 2)
    axGr.plot(riSl, absMagRHsc, color='black')
    axRi.plot(izSl, absMagIHsc, color='black')
    axGr.set_xlabel('r-i', fontsize=fontSize)
    axRi.set_xlabel('i-z', fontsize=fontSize)
    axGr.set_ylabel('Absolute Magnitude HSC-R', fontsize=fontSize)
    axRi.set_ylabel('Absolute Magnitude HSC-I', fontsize=fontSize)
    axGr.set_xlim((-0.05, 0.4))
    axRi.set_xlim((-0.05, 0.2))
    axGr.xaxis.set_ticks([0.0, 0.1, 0.2, 0.3, 0.4])
    for ax in fig.get_axes():
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontSize)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontSize)
    dirHome = os.path.expanduser('~')
    fileFigGri = os.path.join(dirHome, 'Desktop/photoParallaxGri.png')
    fileFigRiz = os.path.join(dirHome, 'Desktop/photoParallaxRiz.png')
    fileFig = os.path.join(dirHome, 'Desktop/photoParallax.png')
    figGri.savefig(fileFigGri, dpi=120, bbox_inches='tight')
    figRiz.savefig(fileFigRiz, dpi=120, bbox_inches='tight')
    fig.savefig(fileFig, dpi=120, bbox_inches='tight')
    return figGri, figRiz, fig

def getParallax(gHsc, rHsc, iHsc, zHsc, projected=False):
    grHsc = gHsc - rHsc
    riHsc = rHsc - iHsc
    izHsc = iHsc - zHsc
    grSdss, riSdss, izSdss = _fromHscToSdss(grHsc, riHsc, izHsc)
    gSdss = gHsc - cgr[0] - cgr[1]*grSdss - cgr[2]*grSdss**2
    rSdss = rHsc - cri[0] - cri[1]*riSdss - cri[2]*riSdss**2
    iSdss = iHsc - ciz[0] - ciz[1]*izSdss - ciz[2]*izSdss**2
    zSdss = zHsc - czi[0] + czi[1]*izSdss - czi[2]*izSdss**2
    if projected:
        P1, P2, grProj, riProj = _getPColors(gSdss, rSdss, iSdss)
        magRAbsSdss = _getAbsoluteMagR(riProj)
        magRAbsHsc = magRAbsSdss + cri[0] + cri[1]*riProj + cri[2]*riProj**2
    else:
        riSdss = rSdss - iSdss
        magRAbsSdss = _getAbsoluteMagR(riSdss)
        magRAbsHsc = magRAbsSdss + cri[0] + cri[1]*riSdss + cri[2]*riSdss**2
    dKpc = np.power(10.0, (rHsc-magRAbsHsc)/5)/100
    return magRAbsHsc, dKpc

def plotPostMarginals(trainClfs=False, fontSize=18):
    magBins = [(18.0, 22.0), (22.0, 24.0), (24.0, 25.0), (25.0, 26.0)]
    with open('trainSet.pkl', 'rb') as f:
        trainSet = pickle.load(f)
    if trainClfs:
        fontSize = 18
        clfs = []
        gaussians = [(10, 10), (10, 10), (10, 10), (10, 10)]
        X, XErr, Y = trainSet.getTrainSet(standardized=False)
        mags = trainSet.getTrainMags()
        if mags.shape[1] > 1:
            mags = mags[:, 2]
        for i, magBin in enumerate(magBins):
            good = np.logical_and(magBin[0] < mags, mags < magBin[1])
            ngStar, ngGal = gaussians[i]
            clf = dGauss.XDClf(ngStar=ngStar, ngGal=ngGal)
            clf.fit(X[good], XErr[good], Y[good])
            clfs.append(clf)
        with open('clfsCols.pkl', 'wb') as f:
            pickle.dump(clfs, f)
    else:
        with open('clfsCols.pkl', 'rb') as f:
            clfs = pickle.load(f)
    X, XErr, Y = trainSet.getTestSet(standardized=False)
    mags = trainSet.getTestMags()
    if mags.shape[1] > 1:
        mags = mags[:, 2]
    colsList = [[0, 1], [1, 2], [2, 3]]
    cNames = ['g-r', 'r-i', 'i-z', 'z-y']
    colsLims = [[(-0.5, 2.5), (-0.5, 3.5)], [(-0.5, 3.5), (-0.5, 1.5)], [(-0.5, 1.5), (-0.5, 1.0)]]
    for j, clf in enumerate(clfs):
        fig = plt.figure(figsize=(24, 12), dpi=120)
        good = np.logical_and(magBins[j][0] < mags, mags < magBins[j][1])
        for i, cols in enumerate(colsList):
            clfMarginal = clf.getMarginalClf(cols=cols)
            axCmap = fig.add_subplot(2, 3, i+1)
            xRange = np.linspace(colsLims[i][0][0], colsLims[i][0][1], num=100)
            yRange = np.linspace(colsLims[i][1][0], colsLims[i][1][1], num=100)
            Xgrid, Ygrid = np.meshgrid(xRange, yRange)
            XInput = np.vstack((Xgrid.flatten(), Ygrid.flatten())).T
            XInputErr = np.zeros((XInput.shape + (XInput.shape[-1],)))
            Z = clfMarginal.predict_proba(XInput, XInputErr)
            Z = Z.reshape(Xgrid.shape)
            im = axCmap.imshow(Z, extent=[xRange[0], xRange[-1], yRange[0], yRange[-1]], aspect='auto', origin='lower', vmin=0.0, vmax=1.0)
            axCmap.set_xlabel(cNames[i], fontsize=fontSize)
            axCmap.set_ylabel(cNames[i+1], fontsize=fontSize)
            axScat = fig.add_subplot(2, 3, i+4)
            for k in range(len(X[good])):
                if Y[good][k]:
                    axScat.plot(X[good][k, i], X[good][k, i+1], marker='.', markersize=1, color='blue')
                else:
                    axScat.plot(X[good][k, i], X[good][k, i+1], marker='.', markersize=1, color='red')
            axScat.set_xlim(colsLims[i][0])
            axScat.set_ylim(colsLims[i][1])
            axScat.set_xlabel(cNames[i], fontsize=fontSize)
            axScat.set_ylabel(cNames[i+1], fontsize=fontSize)
        cax = fig.add_axes([0.93, 0.525, 0.015, 0.375])
        cb = plt.colorbar(im, cax=cax)
        cb.set_label(r'$P(\mathrm{Star})$', fontsize=fontSize)
        cb.ax.tick_params(labelsize=fontSize)
        for ax in fig.get_axes():
            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(fontSize)
            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(fontSize)
        magString = r'$\mathrm{Mag}_{cmodel}$ HSC-I'
        fig.suptitle('{0} < {2} < {1}'.format(magBins[j][0], magBins[j][1], magString), fontsize=fontSize)
        dirHome = os.path.expanduser('~')
        fileFig = os.path.join(dirHome, 'Desktop/xdFitVsData{0}-{1}.png'.format(*magBins[j]))
        fig.savefig(fileFig, dpi=120, bbox_inches='tight')

def makeTomPlots(dKpc, exts, magRAbsHsc, X, magRHsc, withProb=False, YProbGri=None, YProbRiz=None,
                 title='Pure Morphology Classifier', limDkpc=(0.0, 100.0)):
    if withProb:
        assert YProbGri is not None
        assert YProbRiz is not None
    fig = plt.figure(figsize=(10, 12), dpi=120)
    axExt = fig.add_subplot(3, 2, 1)
    axExt.scatter(dKpc, exts, marker='.', s=1)
    axExt.set_xlim(limDkpc)
    axExt.set_ylim((-0.01, 0.1))
    axExt.set_xlabel('d (kpc)')
    axExt.set_ylabel('Mag_psf-Mag_cmodel')
    axMagAbs = fig.add_subplot(3, 2, 2)
    axMagAbs.scatter(dKpc, magRAbsHsc, marker='.', s=1)
    axMagAbs.set_xlim(limDkpc)
    axMagAbs.set_ylim((4.0, 16.0))
    axMagAbs.set_xlabel('d (kpc)')
    axMagAbs.set_ylabel('Absolute Magnitude HSC-R')
    axCol = fig.add_subplot(3, 2, 3)
    axCol.scatter(dKpc, X[:,1], marker='.', s=1)
    axCol.set_xlim(limDkpc)
    axCol.set_ylim((-0.2, 2.0))
    axCol.set_xlabel('d (kpc)')
    axCol.set_ylabel('r-i')
    axMag = fig.add_subplot(3, 2, 4)
    axMag.scatter(dKpc, magRHsc, marker='.', s=1)
    axMag.set_xlim(limDkpc)
    axMag.set_xlabel('d (kpc)')
    axMag.set_ylabel('Apparent Magnitude HSC-R')
    axGr = fig.add_subplot(3, 2, 5)
    if withProb:
        sc = axGr.scatter(X[:,0], X[:,1], c=YProbGri, marker='.', s=2, edgecolors="none")
        cb = fig.colorbar(sc, ax=axGr)
    else:
        sc = axGr.scatter(X[:,0], X[:,1], marker='.', s=2)
    axGr.set_xlabel('g-r')
    axGr.set_ylabel('r-i')
    axGr.set_xlim((-0.5, 2.5))
    axGr.set_ylim((-0.2, 2.0))
    axRi = fig.add_subplot(3, 2, 6)
    if withProb:
        sc = axRi.scatter(X[:,1], X[:,2], c=YProbRiz,  marker='.', s=2, edgecolors="none")
        cb = fig.colorbar(sc, ax=axRi)
    else:
        sc = axRi.scatter(X[:,1], X[:,2], marker='.', s=2)
    axRi.set_xlabel('r-i')
    axRi.set_ylabel('i-z')
    axRi.set_xlim((-0.2, 2.0))
    axRi.set_ylim((-0.2, 1.0))
    fig.suptitle(title)
    return fig

def makeTomPlotsProd(dKpc, exts, magRAbsHsc, X, magRHsc, withProb=False, YProbGri=None, YProbRiz=None,
                     title='Pure Morphology Classifier', limDkpc=(8.0, 80.0)):
    if withProb:
        assert YProbGri is not None
        assert YProbRiz is not None
    fig = plt.figure(figsize=(16, 12), dpi=120)
    axMagAbs = fig.add_subplot(2, 2, 1)
    axMagAbs.scatter(dKpc, magRAbsHsc, marker='.', s=1)
    axMagAbs.set_xlim(limDkpc)
    axMagAbs.set_ylim((3.0, 7.5))
    axMagAbs.set_xlabel('d (kpc)')
    axMagAbs.set_ylabel('Absolute Magnitude HSC-R')
    axMag = fig.add_subplot(2, 2, 2)
    axMag.scatter(dKpc, magRHsc, marker='.', s=1)
    axMag.set_xlim(limDkpc)
    axMag.set_ylim((19.0, 24.5))
    axMag.set_xlabel('d (kpc)')
    axMag.set_ylabel('Apparent Magnitude HSC-R')
    axRi = fig.add_subplot(2, 2, 3)
    axRi.scatter(dKpc, X[:,1], marker='.', s=1)
    axRi.set_xlim(limDkpc)
    axRi.set_ylim((-0.1, 0.4))
    axRi.set_xlabel('d (kpc)')
    axRi.set_ylabel('r-i')
    axIz = fig.add_subplot(2, 2, 4)
    axIz.scatter(dKpc, X[:,2], marker='.', s=1)
    axIz.set_xlim(limDkpc)
    axIz.set_ylim((-0.1, 0.2))
    axIz.set_xlabel('d (kpc)')
    axIz.set_ylabel('i-z')
    #fig.suptitle(title)
    return fig

def truthStarsTom(frac=None, cutRedGr=1.2, cutRedRi=0.7):
    with open('trainSet.pkl', 'rb') as f:
        trainSet = pickle.load(f)
    X, XErr, Y = trainSet.getTestSet(standardized=False)
    mags = trainSet.getTestMags()
    exts = trainSet.getTestExts()
    good = np.logical_and(np.logical_and(Y, X[:,0] < cutRedGr), X[:,1] < cutRedRi)
    X = X[good]; mags = mags[good]; exts = exts[good]
    magRAbsHsc, dKpc = getParallax(mags[:,0], mags[:,1], mags[:,2], mags[:,3])
    if frac is not None:
        choice = np.random.choice(len(X), size=int(frac*len(X)), replace=False)
        dKpc = dKpc[choice]; exts = exts[choice]; magRAbsHsc = magRAbsHsc[choice]; X = X[choice]; mags = mags[choice]
    fig = makeTomPlots(dKpc, exts[:,1], magRAbsHsc, X, mags[:,1], title='True Stars')
    dirHome = os.path.expanduser('~')
    fileFig = os.path.join(dirHome, 'Desktop/truthStars.png')
    fig.savefig(fileFig, dpi=120, bbox_inches='tight')
    return fig

def boxStarsTom(cutRedGr=1.2, cutRedRi=0.7):
    with open('trainSet.pkl', 'rb') as f:
        trainSet = pickle.load(f)
    clf = etl.BoxClf()
    clf._setX(24.0); clf._setY(0.02)
    idxBest = np.argmax(trainSet.snrs, axis=1)
    idxArr = np.arange(len(trainSet.snrs))
    mags = trainSet.mags[idxArr, idxBest]
    exts = trainSet.exts[idxArr, idxBest]
    Xbox = np.vstack((mags, exts)).T
    Ybox = clf.predict(Xbox)
    X = trainSet.X[Ybox]
    mags = trainSet.mags[Ybox]
    exts = exts[Ybox]
    magRAbsHsc, dKpc = getParallax(mags[:,0], mags[:,1], mags[:,2], mags[:,3])
    if True:
        choice = np.random.choice(len(X), size=int(0.1*len(X)), replace=False)
        dKpc = dKpc[choice]; exts = exts[choice]; magRAbsHsc = magRAbsHsc[choice]; X = X[choice]; mags = mags[choice]
    fig = makeTomPlots(dKpc, exts, magRAbsHsc, X, mags[:,1])
    dirHome = os.path.expanduser('~')
    fileFig = os.path.join(dirHome, 'Desktop/boxStars.png')
    fig.savefig(fileFig, dpi=120, bbox_inches='tight')
    return fig

def colExtStarsTom(trainClfs=False, cutRedRi=0.4, cutRedIz=0.2, b=42.10264796, l=236.81366468,
                   fontSize=18, computePosteriors=False):
    with open('trainSet.pkl', 'rb') as f:
        trainSet = pickle.load(f)
    magBins = [(18.0, 22.0), (22.0, 24.0), (24.0, 25.0), (25.0, 26.0)]
    if trainClfs:
        gaussians = [(10, 10), (10, 10), (10, 10), (10, 10)]
        X, XErr, Y = trainSet.genColExtTrainSet(mode='train')
        mags = trainSet.getTrainMags(band='i')
        clfs = []
        for i, magBin in enumerate(magBins):
            good = np.logical_and(magBin[0] < mags, mags < magBin[1])
            ngStar, ngGal = gaussians[i]
            clf = dGauss.XDClf(ngStar=ngStar, ngGal=ngGal)
            clf.fit(X[good], XErr[good], Y[good])
            clfs.append(clf)
        with open('clfsColsExt.pkl', 'wb') as f:
            pickle.dump(clfs, f)
    else:
        with open('clfsColsExt.pkl', 'rb') as f:
            clfs = pickle.load(f)
    X, XErr, Y = trainSet.genColExtTrainSet(mode='all')
    mags = trainSet.getAllMags(band='i')
    exts = trainSet.getAllExts(band='i')
    clfXd = dGauss.XDClfs(clfs=clfs, magBins=magBins)
    #posteriors = clfXd.predict_proba(X, XErr, mags)
    if computePosteriors:
        YProb = np.zeros(Y.shape)
        YProbGri = np.zeros(Y.shape)
        YProbRiz = np.zeros(Y.shape)
        for i, magBin in enumerate(magBins):
            clfMarginalGri = clfs[i].getMarginalClf(cols=[0, 1])
            clfMarginalRiz = clfs[i].getMarginalClf(cols=[1, 2])
            magCut = np.logical_and(magBin[0] < mags, mags < magBin[1])
            YProb[magCut] = clfs[i].predict_proba(X[magCut], XErr[magCut])
            rowsV, colsV = np.meshgrid([0, 1], [0, 1], indexing='ij')
            YProbGri[magCut] = clfMarginalGri.predict_proba(X[magCut][:, [0, 1]], XErr[magCut][:, rowsV, colsV])
            rowsV, colsV = np.meshgrid([1, 2], [1, 2], indexing='ij')
            YProbRiz[magCut] = clfMarginalRiz.predict_proba(X[magCut][:, [1, 2]], XErr[magCut][:, rowsV, colsV])
        with open('cosmosTomPosteriors.pkl', 'wb') as f:
            pickle.dump((YProb, YProbGri, YProbRiz), f)
    else:
        with open('cosmosTomPosteriors.pkl', 'rb') as f:
            YProb, YProbGri, YProbRiz = pickle.load(f)
    good = np.logical_and(YProb > 0.8, mags < 24.0)
    good = np.logical_and(np.logical_and(good, X[:,1] < cutRedRi), X[:,2] < cutRedIz)
    mags = trainSet.getAllMags()
    magRAbsHsc, dKpc = getParallax(mags[good,0], mags[good,1], mags[good,2], mags[good,3])
    b = np.radians(b)
    l = np.radians(l)
    dKpcGal = np.sqrt(8.0**2 + dKpc**2 - 2*8.0*dKpc*np.cos(b)*np.cos(l))
    sinBStar = dKpc*np.sin(b)/dKpcGal
    cosBStar = np.sqrt(1.0 - sinBStar**2)
    RStar = dKpcGal*cosBStar
    ZStar = dKpcGal*sinBStar
    exts = exts[good]
    mags = mags[good]
    X = X[good]
    fig = makeTomPlotsProd(dKpcGal, exts, magRAbsHsc, X, mags[:,1], withProb=True,
                           YProbGri=YProbGri[good], YProbRiz=YProbRiz[good],
                           title='Morphology+Colors')
    figStruct = plt.figure()
    ax = figStruct.add_subplot(1, 1, 1)
    magRAbsHscBins = [(4.5, 5.0), (5.0, 5.5), (5.5, 6.0), (6.0, 6.5), (6.5, 7.0)]
    colors = ['blue', 'red', 'green', 'cyan', 'black']
    for i, magAbsBin in enumerate(magRAbsHscBins):
        magAbsCut = np.logical_and(magRAbsHsc > magAbsBin[0], magRAbsHsc < magAbsBin[1])
        histData = dKpcGal[magAbsCut]
        hist, bins = np.histogram(histData, bins=15, range=(20.0, 80.0))
        binCenters = 0.5*(bins[:-1] + bins[1:])
        hist = hist*1.0/binCenters**2
        hist = hist*1.0/hist[0]
        barWidth = binCenters[1] - binCenters[0]
        ax.bar(bins[:-1], hist, barWidth, edgecolor=colors[i], fill=False)
    #ax.scatter(RStar, ZStar, marker='.', s=5, color='black')
    #ax.set_xlabel('R (kpc)', fontsize=fontSize)
    #ax.set_ylabel('Z (kpc)', fontsize=fontSize)
    #for ax in fig.get_axes():
    #    for tick in ax.xaxis.get_major_ticks():
    #        tick.label.set_fontsize(16)
    #    for tick in ax.yaxis.get_major_ticks():
    #        tick.label.set_fontsize(16)
    dirHome = os.path.expanduser('~')
    fileFig = os.path.join(dirHome, 'Desktop/colExtStars.png')
    fig.savefig(fileFig, dpi=120, bbox_inches='tight')
    return fig

def highPostStarsShape(trainClfs=False, withBox=False):
    with open('trainSet.pkl', 'rb') as f:
        trainSet = pickle.load(f)
    fontSize = 18
    magBins = [(18.0, 22.0), (22.0, 24.0), (24.0, 25.0), (25.0, 26.0)]
    clfs = []
    if trainClfs:
        gaussians = [(10, 10), (10, 10), (10, 10), (10, 10)]
        X, XErr, Y = trainSet.getTrainSet(standardized=False)
        mags = trainSet.getTrainMags()
        for i, magBin in enumerate(magBins):
            good = np.logical_and(magBin[0] < mags, mags < magBin[1])
            ngStar, ngGal = gaussians[i]
            clf = dGauss.XDClf(ngStar=ngStar, ngGal=ngGal)
            clf.fit(X[good], XErr[good], Y[good])
            clfs.append(clf)
        with open('clfsCols.pkl', 'wb') as f:
            pickle.dump(clfs, f)
    else:
        with open('clfsCols.pkl', 'rb') as f:
            clfs = pickle.load(f)
    X, XErr, Y = trainSet.getTestSet(standardized=False)
    mags = trainSet.getTestMags()
    exts = trainSet.getTestExts()
    YProb = np.zeros(Y.shape)
    YProbGri = np.zeros(Y.shape)
    YProbRiz = np.zeros(Y.shape)
    for i, magBin in enumerate(magBins):
        clfMarginalGri = clfs[i].getMarginalClf(cols=[0, 1])
        clfMarginalRiz = clfs[i].getMarginalClf(cols=[1, 2])
        magCut = np.logical_and(magBin[0] < mags, mags < magBin[1])
        YProb[magCut] = clfs[i].predict_proba(X[magCut], XErr[magCut])
        rowsV, colsV = np.meshgrid([0, 1], [0, 1], indexing='ij')
        YProbGri[magCut] = clfMarginalGri.predict_proba(X[magCut][:, [0, 1]], XErr[magCut][:, rowsV, colsV])
        rowsV, colsV = np.meshgrid([1, 2], [1, 2], indexing='ij')
        YProbRiz[magCut] = clfMarginalRiz.predict_proba(X[magCut][:, [1, 2]], XErr[magCut][:, rowsV, colsV])
    good = np.logical_and(YProb > 0.9, mags < 26.0)
    grSdss, riSdss, izSdss = _fromHscToSdss(X[good,0], X[good,1], X[good,2])
    YProbGri = YProbGri[good]
    YProbRiz = YProbRiz[good]
    magIHsc = mags[good]
    magRHsc = magIHsc + X[good, 1]
    magGHsc = magRHsc + X[good, 0]
    magZHsc = magIHsc - X[good, 2]
    magGSdss = magGHsc - cgr[0] - cgr[1]*grSdss - cgr[2]*grSdss**2
    magRSdss = magRHsc - cri[0] - cri[1]*riSdss - cri[2]*riSdss**2
    magISdss = magIHsc - ciz[0] - ciz[1]*izSdss - ciz[2]*izSdss**2
    magZSdss = magZHsc - czi[0] + czi[1]*izSdss - czi[2]*izSdss**2
    P1, P2, grProj, riProj = _getPColors(magGSdss, magRSdss, magISdss)
    #magRAbsSdss = _getAbsoluteMagR(riSdss)
    magRAbsSdss = _getAbsoluteMagR(riProj)
    #fig = plt.figure()
    #plt.scatter(magRAbsSdss, magRAbsProj, marker='.', s=1)
    #for i in range(len(grSdss)):
    #    plt.plot([riSdss[i], riProj[i]], [magRAbsSdss[i], magRAbsProj[i]])
    #plt.show()
    #magRAbsHsc = magRAbsSdss + cri[0] + cri[1]*riSdss + cri[2]*riSdss**2
    magRAbsHsc = magRAbsSdss + cri[0] + cri[1]*riProj + cri[2]*riProj**2
    dKpc = np.power(10.0, (magRSdss-magRAbsSdss)/5)/100
    fig = plt.figure(figsize=(10, 12), dpi=120)
    #fig = plt.figure(figsize=(24, 12), dpi=120)
    axExt = fig.add_subplot(3, 2, 1)
    axExt.scatter(dKpc, exts[good], marker='.', s=1)
    axExt.set_xlim((0.0, 50.0))
    axExt.set_ylim((-0.01, 0.1))
    axExt.set_xlabel('d (kpc)')
    axExt.set_ylabel('Mag_psf-Mag_cmodel')
    axMagAbs = fig.add_subplot(3, 2, 2)
    axMagAbs.scatter(dKpc, magRAbsHsc, marker='.', s=1)
    axMagAbs.set_xlim((0.0, 50.0))
    axMagAbs.set_ylim((4.0, 16.0))
    axMagAbs.set_xlabel('d (kpc)')
    axMagAbs.set_ylabel('Absolute Magnitude HSC-R')
    axCol = fig.add_subplot(3, 2, 3)
    axCol.scatter(dKpc, X[good,1], marker='.', s=1)
    axCol.set_xlim((0.0, 50.0))
    axCol.set_ylim((-0.2, 2.0))
    axCol.set_xlabel('d (kpc)')
    axCol.set_ylabel('r-i')
    axMag = fig.add_subplot(3, 2, 4)
    axMag.scatter(dKpc, magRHsc, marker='.', s=1)
    axMag.set_xlim((0.0, 50.0))
    #axMag.set_ylim((-0.2, 2.0))
    axMag.set_xlabel('d (kpc)')
    axMag.set_ylabel('Apparent Magnitude HSC-R')
    #redColor = np.logical_and(X[good, 0] < 1.0, X[good, 1] > 0.5*X[good, 0])
    #redSample = np.random.choice(np.sum(redColor))
    #sampleGr = X[good][redColor][redSample][0]
    #sampleRi = X[good][redColor][redSample][1]
    #sampleIz = X[good][redColor][redSample][2]
    #sampleZy = X[good][redColor][redSample][3]
    #sampleMag = mags[good][redColor][redSample]
    #for idxMagBin, magBin in enumerate(magBins):
    #    if np.logical_and(sampleMag > magBin[0], sampleMag < magBin[1]):
    #        break
    #sampleClf = clfs[idxMagBin]
    #sampleXErr = XErr[good][redColor][redSample]
    #print sampleXErr
    #circGri = plt.Circle((sampleGr, sampleRi), radius=0.02, color='magenta', fill=False)
    #circGriMarg = plt.Circle((sampleGr, sampleRi), radius=0.02, color='magenta', fill=False)
    #circRiz = plt.Circle((sampleRi, sampleIz), radius=0.02, color='magenta', fill=False)
    #circRizMarg = plt.Circle((sampleRi, sampleIz), radius=0.02, color='magenta', fill=False)
    #circIzy = plt.Circle((sampleIz, sampleZy), radius=0.02, color='magenta', fill=False)
    #circIzyMarg = plt.Circle((sampleIz, sampleZy), radius=0.02, color='magenta', fill=False)
    #blackColor = np.logical_not(redColor)
    axGr = fig.add_subplot(3, 2, 5)
    #axGr = fig.add_subplot(2, 3, 1)
    #sc = axGr.scatter(X[good][blackColor,0], X[good][blackColor,1], c='black', marker='.', s=5, edgecolors="none")
    #sc = axGr.scatter(X[good][redColor,0], X[good][redColor,1], c='red', marker='.', s=5, edgecolors="none")
    #axGr.add_patch(circGri)
    sc = axGr.scatter(X[good,0], X[good,1], c=YProbGri, marker='.', s=2, edgecolors="none")
    cb = fig.colorbar(sc, ax=axGr)
    axGr.set_xlabel('g-r')
    axGr.set_ylabel('r-i')
    axGr.set_xlim((-0.5, 2.5))
    axGr.set_ylim((-0.2, 2.0))
    #axGrMarg = fig.add_subplot(2, 3, 4)
    #grLim = axGr.get_xlim()
    #riLim = axGr.get_ylim()
    #xRange = np.linspace(grLim[0], grLim[1], num=100)
    #yRange = np.linspace(riLim[0], riLim[1], num=100)
    #Xgrid, Ygrid = np.meshgrid(xRange, yRange)
    #XInput = np.vstack((Xgrid.flatten(), Ygrid.flatten())).T
    #XInputErr = np.zeros((XInput.shape + (XInput.shape[-1],)))
    #rowsV, colsV = np.meshgrid([0, 1], [0, 1], indexing='ij')
    #XInputErr[:] = sampleXErr[rowsV, colsV]
    #clfMarginal = sampleClf.getMarginalClf(cols=[0, 1])
    #Z = clfMarginal.predict_proba(XInput, XInputErr)
    #Z = Z.reshape(Xgrid.shape)
    #im = axGrMarg.imshow(Z, extent=[xRange[0], xRange[-1], yRange[0], yRange[-1]], aspect='auto', origin='lower')
    #cb = plt.colorbar(im)
    #axGrMarg.add_patch(circGriMarg)
    #axGrMarg.set_xlabel('g-r')
    #axGrMarg.set_ylabel('r-i')
    #axGrMarg.set_xlim(grLim)
    #axGrMarg.set_ylim(riLim)
    axRi = fig.add_subplot(3, 2, 6)
    #axRi = fig.add_subplot(2, 3, 2)
    #sc = axRi.scatter(X[good][blackColor,1], X[good][blackColor,2], c='black', marker='.', s=5, edgecolors="none")
    #sc = axRi.scatter(X[good][redColor,1], X[good][redColor,2], c='red', marker='.', s=5, edgecolors="none")
    #axRi.add_patch(circRiz)
    sc = axRi.scatter(X[good,1], X[good,2], c=YProbRiz,  marker='.', s=2, edgecolors="none")
    cb = fig.colorbar(sc, ax=axRi)
    axRi.set_xlabel('r-i')
    axRi.set_ylabel('i-z')
    axRi.set_xlim((-0.2, 2.0))
    axRi.set_ylim((-0.2, 1.5))
    #axRiMarg = fig.add_subplot(2, 3, 5)
    #riLim = axRi.get_xlim()
    #izLim = axRi.get_ylim()
    #xRange = np.linspace(riLim[0], riLim[1], num=100)
    #yRange = np.linspace(izLim[0], izLim[1], num=100)
    #Xgrid, Ygrid = np.meshgrid(xRange, yRange)
    #XInput = np.vstack((Xgrid.flatten(), Ygrid.flatten())).T
    #XInputErr = np.zeros((XInput.shape + (XInput.shape[-1],)))
    #rowsV, colsV = np.meshgrid([1, 2], [1, 2], indexing='ij')
    #XInputErr[:] = sampleXErr[rowsV, colsV]
    #clfMarginal = sampleClf.getMarginalClf(cols=[1, 2])
    #Z = clfMarginal.predict_proba(XInput, XInputErr)
    #Z = Z.reshape(Xgrid.shape)
    #im = axRiMarg.imshow(Z, extent=[xRange[0], xRange[-1], yRange[0], yRange[-1]], aspect='auto', origin='lower')
    #cb = plt.colorbar(im)
    #axRiMarg.add_patch(circRizMarg)
    #axRiMarg.set_xlabel('r-i')
    #axRiMarg.set_ylabel('i-z')
    #axRiMarg.set_xlim(riLim)
    #axRiMarg.set_ylim(izLim)
    #axIz = fig.add_subplot(2, 3, 3)
    #sc = axIz.scatter(X[good][blackColor,2], X[good][blackColor,3], c='black', marker='.', s=5, edgecolors="none")
    #sc = axIz.scatter(X[good][redColor,2], X[good][redColor,3], c='red', marker='.', s=5, edgecolors="none")
    #axIz.add_patch(circIzy)
    #axIz.set_xlabel('i-z')
    #axIz.set_ylabel('z-y')
    #axIz.set_xlim((-0.2, 1.5))
    #axIz.set_ylim((-1.0, 1.0))
    #axIzMarg = fig.add_subplot(2, 3, 6)
    #izLim = axIz.get_xlim()
    #zyLim = axIz.get_ylim()
    #xRange = np.linspace(izLim[0], izLim[1], num=100)
    #yRange = np.linspace(zyLim[0], zyLim[1], num=100)
    #Xgrid, Ygrid = np.meshgrid(xRange, yRange)
    #XInput = np.vstack((Xgrid.flatten(), Ygrid.flatten())).T
    #XInputErr = np.zeros((XInput.shape + (XInput.shape[-1],)))
    #rowsV, colsV = np.meshgrid([2, 3], [2, 3], indexing='ij')
    #XInputErr[:] = sampleXErr[rowsV, colsV]
    #clfMarginal = sampleClf.getMarginalClf(cols=[2, 3])
    #Z = clfMarginal.predict_proba(XInput, XInputErr)
    #Z = Z.reshape(Xgrid.shape)
    #im = axIzMarg.imshow(Z, extent=[xRange[0], xRange[-1], yRange[0], yRange[-1]], aspect='auto', origin='lower')
    #cb = plt.colorbar(im)
    #axIzMarg.add_patch(circIzyMarg)
    #axIzMarg.set_xlabel('i-z')
    #axIzMarg.set_ylabel('z-y')
    #axIzMarg.set_xlim(izLim)
    #axIzMarg.set_ylim(zyLim)
    fig.suptitle('Objects with P(Star|Colors)>0.9')
    dirHome = os.path.expanduser('~')
    fileFig = os.path.join(dirHome, 'Desktop/colorStarsShapes.png')
    fig.savefig(fileFig, dpi=120, bbox_inches='tight')
    plt.show()

def morphStarsPlotsSingle(train=False, featuresCuts={1:(None, 0.2)}, ylim=(-0.5, 1.0), xRange=(23.5, 24.5),
                          yRange=(-400000, 400000), ylabel='rTrace', asLogX=False, xlim=(18.0, 27.0), xlabel='Magnitude'):
    trains = {}
    for band in ['g', 'r', 'i', 'z', 'y']:
        with open('trainSetHsmSingleBand{0}.pkl'.format(band.upper()), 'rb') as f:
            trainSet = pickle.load(f)
        clf = dGauss.logisticFit(trainSet, featuresCuts=featuresCuts, n_jobs=1, doCV=False, C=0.01, class_weight='auto')
        train = etl.Training(trainSet, clf)
        figPMap = train.plotPMap(xlim, ylim, 200, 200, xlabel=xlabel+band.upper(), ylabel=ylabel+band.upper(), asLogX=asLogX, cbLabel='pStar')
        #figBdy = train.plotBoundary(0, 1, xRange=xRange, overPlotData=True, ylim=ylim, asLogX=asLogX, xlim=xlim, yRange=yRange,
        #                            xlabel=xlabel+band.upper(), ylabel=ylabel+band.upper(), frac=0.05)
        trains[band] = train
    plt.show()
    return trains

def morphStarsPlotsMulti(train=False, featuresCuts={1:(None, 0.2)}, ylim=(-0.4, 1), xRange=(50.0, 3000.0),
                         yRange=(-4, 4), ylabel='rTrace', asLogX=True, xlim=(5.0, 3000), xlabel='S/N'):
    with open('trainSetHsmMultiBand.pkl', 'rb') as f:
        trainSet = pickle.load(f)
    clf = dGauss.logisticFit(trainSet, featuresCuts=featuresCuts, n_jobs=1, doCV=False, C=100.0)
    train = etl.Training(trainSet, clf)
    figPMap = train.plotPMap(xlim, ylim, 200, 200, xlabel=xlabel, ylabel=ylabel, asLogX=asLogX, cbLabel='pStar')

    figPMap.savefig('/u/garmilla/Desktop/pMapHsmMulti.png', dpi=120, bbox_inches='tight')
    train.printPolynomial(['snrPsf', 'rDet'])
    figBdy = train.plotBoundary(0, 1, xRange=xRange, overPlotData=True, ylim=ylim, asLogX=asLogX, xlim=xlim, yRange=yRange,
                                xlabel=xlabel, ylabel=ylabel, frac=0.006)
    figBdy.savefig('/u/garmilla/Desktop/boundaryHscMulti.png', dpi=120, bbox_inches='tight')
    mpl.rcParams['figure.figsize'] = 12, 6
    figScores = train.plotScores(magRange=(18.0, 26.0))
    figScores.savefig('/u/garmilla/Desktop/scoresHsmMulti.png', dpi=120, bbox_inches='tight')
    return train

def rcPlots(rerun='Cosmos1', polyOrder=3, snrType='snrPsf', extType='extHsmDeconv', ylim=(-2, 5), xRange=(10.0, 3000.0),
            yRange=(-20, 20), ylabel='rTrace', featuresCuts={1:(None, 1.0)}, asLogX=True, xlim=(5.0, 3000), xlabel='S/N',
            singleBand=False, band='i'):
    if rerun == 'Cosmos1':
        cat = afwTable.SimpleCatalog.readFits('/scr/depot0/garmilla/HSC/matchDeepCoaddMeas-137520151126Cosmos1GRIZY.fits')
    elif rerun == 'Cosmos2':
        cat = afwTable.SimpleCatalog.readFits('/scr/depot0/garmilla/HSC/matchDeepCoaddMeas-137520151126Cosmos2GRIZY.fits')
    elif rerun == 'Cosmos':
        cat = afwTable.SimpleCatalog.readFits('/scr/depot0/garmilla/HSC/matchDeepCoaddMeas-137520151126CosmosGRIZY.fits')

    if rerun in ['Cosmos1', 'Cosmos2']:
        trainSet = etl.extractTrainSet(cat, inputs=[snrType, extType], polyOrder=polyOrder, bands=['i'])
    elif rerun == 'Cosmos':
        if singleBand:
            trainSet = etl.extractTrainSet(cat, inputs=[snrType, extType], bands=[band], polyOrder=polyOrder)
        else:
            trainSet = etl.extractTrainSet(cat, inputs=[snrType, extType], bands=['g', 'r', 'i', 'z', 'y'], polyOrder=polyOrder)
            with open('trainSetHsmMultiBand.pkl', 'wb') as f:
                pickle.dump(trainSet, f)

    clf = dGauss.logisticFit(trainSet, featuresCuts=featuresCuts, n_jobs=1)
    train = etl.Training(trainSet, clf)
    figPMap = train.plotPMap(xlim, ylim, 200, 200, xlabel=xlabel, ylabel=ylabel, asLogX=asLogX, cbLabel='pStar')

    figPMap.savefig('/u/garmilla/Desktop/pMap{0}.png'.format(rerun), dpi=120, bbox_inches='tight')
    train.printPolynomial(['snr', 'magDiff'])
    if rerun in ['Cosmos1', 'Cosmos2']:
        figBdy = train.plotBoundary(0, 1, xRange=xRange, overPlotData=True, ylim=ylim, asLogX=asLogX, xlim=xlim, yRange=yRange,
                                    xlabel=xlabel, ylabel=ylabel, frac=0.06)
    elif rerun == 'Cosmos':
        figBdy = train.plotBoundary(0, 1, xRange=xRange, overPlotData=True, ylim=ylim, asLogX=asLogX, xlim=xlim, yRange=yRange,
                                    xlabel=xlabel, ylabel=ylabel, frac=0.006)
    figBdy.savefig('/u/garmilla/Desktop/boundary{0}.png'.format(rerun), dpi=120, bbox_inches='tight')
    mpl.rcParams['figure.figsize'] = 12, 6
    figScores = train.plotScores(magRange=(18.0, 26.0))
    figScores.savefig('/u/garmilla/Desktop/scores{0}.png'.format(rerun), dpi=120, bbox_inches='tight')

def magExtPlots(rerun='Cosmos1'):
    if rerun == 'Cosmos1':
        cat = afwTable.SimpleCatalog.readFits('/scr/depot0/garmilla/HSC/matchDeepCoaddMeas-137520151126Cosmos1GRIZY.fits')
    elif rerun == 'Cosmos2':
        cat = afwTable.SimpleCatalog.readFits('/scr/depot0/garmilla/HSC/matchDeepCoaddMeas-137520151126Cosmos2GRIZY.fits')
    elif rerun == 'Cosmos':
        cat = afwTable.SimpleCatalog.readFits('/scr/depot0/garmilla/HSC/matchDeepCoaddMeas-137520151126CosmosGRIZY.fits')

    #fig = utils.makeMagExPlot(cat, 'i', withLabels=True, trueSample=True, frac=0.04)
    fig = utils.makeExtHist(cat, 'i', withLabels=True, magCuts=[(18.0, 22.0), (22.0, 24.0), (24.0, 25.0), (25.0, 26.0)], xlim=(-0.01, 0.5))

    #plt.savefig('/u/garmilla/Desktop/magExtDist.png', bbox_inches='tight')

def extCutRoc(rerun='Cosmos1', extType='ext', snrCut=(10, 30), nConnect=20):
    if rerun == 'Cosmos1':
        cat = afwTable.SimpleCatalog.readFits('/scr/depot0/garmilla/HSC/matchDeepCoaddMeas-137520151126Cosmos1GRIZY.fits')
    elif rerun == 'Cosmos2':
        cat = afwTable.SimpleCatalog.readFits('/scr/depot0/garmilla/HSC/matchDeepCoaddMeas-137520151126Cosmos2GRIZY.fits')
    elif rerun == 'Cosmos':
        cat = afwTable.SimpleCatalog.readFits('/scr/depot0/garmilla/HSC/matchDeepCoaddMeas-137520151126CosmosGRIZY.fits')

    if rerun in ['Cosmos1', 'Cosmos2']:
        trainSet = etl.extractTrainSet(cat, inputs=[extType], polyOrder=1)
    elif rerun == 'Cosmos':
        trainSet = etl.extractTrainSet(cat, inputs=[extType], bands=['g', 'r', 'i', 'z', 'y'], polyOrder=1)

    clf = LinearSVC()
    X, Y = trainSet.getAllSet(standardized=False)
    snrs = trainSet.snrs
    inSnrCut = np.logical_and(snrs > snrCut[0], snrs < snrCut[1])
    clf.fit(X, Y)

    cutRange = np.linspace(-0.02, 2.0, num=500)
    xxStars = np.zeros(cutRange.shape)
    yyStars = np.zeros(cutRange.shape)
    xxGals = np.zeros(cutRange.shape)
    yyGals = np.zeros(cutRange.shape)

    Xcut = X[inSnrCut]; Ycut = Y[inSnrCut]
    for i, cut in enumerate(cutRange):
        clf.coef_[0][0] = -1.0
        clf.intercept_[0] = cut
        Ypred = clf.predict(Xcut)
        goodStars = np.logical_and(Ypred, Ycut)
        goodGals = np.logical_and(np.logical_not(Ypred), np.logical_not(Ycut))
        if np.sum(Ycut) > 0:
            xxStars[i] = float(np.sum(goodStars))/np.sum(Ycut)
        if np.sum(Ypred) > 0:
            yyStars[i] = float(np.sum(goodStars))/np.sum(Ypred)
        if len(Ycut) - np.sum(Ycut) > 0:
            xxGals[i] = float(np.sum(goodGals))/(len(Ycut) - np.sum(Ycut))
        if len(Ypred) - np.sum(Ypred) > 0:
            yyGals[i] = float(np.sum(goodGals))/(len(Ypred) - np.sum(Ypred))

    fig = plt.figure(figsize=(16, 6))
    fig.suptitle('{0} < S/N < {1}'.format(*snrCut), fontsize=18)
    axHist = fig.add_subplot(1, 2, 1)
    axRoc = fig.add_subplot(1, 2, 2)

    hist, bins = np.histogram(Xcut[:,0], bins=50, range=(-0.05, 0.5))
    dataStars = Xcut[:,0][Ycut]
    dataGals = Xcut[:,0][np.logical_not(Ycut)]
    axHist.hist(dataStars, bins=bins, histtype='step', color='blue', label='Stars', linewidth=2)
    axHist.hist(dataGals, bins=bins, histtype='step', color='red', label='Galaxies', linewidth=2)
    axHist.set_xlabel('mag_psf-mag_cmodel', fontsize=16)
    axHist.set_ylabel('counts', fontsize=16)

    axRoc.plot(xxStars, yyStars, color='blue', linewidth=3)
    axRoc.plot(xxGals, yyGals, color='red', linewidth=3)
    for i in range(nConnect):
        idx = i*len(xxStars)/(nConnect-1) - 1
        plt.plot([xxStars[idx], xxGals[idx]], [yyStars[idx], yyGals[idx]], color='black', linestyle='--')
    axRoc.set_xlabel('Completeness', fontsize=16)
    axRoc.set_ylabel('Purity', fontsize=16)

    for ax in fig.get_axes():
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(16)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(16)

    plt.show()

def hstVsHscSize(snrCut=(10, 30)):
    cat = afwTable.SimpleCatalog.readFits('/scr/depot0/garmilla/HSC/matchDeepCoaddMeas-137520151126Cosmos1Iiphot.fits')
    extHsc = -2.5*np.log10(cat.get('flux.psf.i')/cat.get('cmodel.flux.i'))
    extHst = cat.get('mu.max')-cat.get('mag.auto')
    snr = cat.get('flux.psf.i')/cat.get('flux.psf.err.i')
    good = np.logical_and(snr > snrCut[0], snr < snrCut[1])

    fig = plt.figure()
    plt.scatter(extHst[good], extHsc[good], marker='.', s=1)
    plt.title('{0} < S/N < {1}'.format(snrCut[0], snrCut[1]))
    plt.xlabel('mu_max-mag_auto (HST)')
    plt.ylabel('mag_psf-mag_cmodel (HSC)')
    plt.xlim((-5.0, 1.0))
    plt.ylim((-0.1, 2.0))

    fig.savefig('/u/garmilla/Desktop/hstVsHscSizeSnr{0}-{1}.png'.format(snrCut[0], snrCut[1]), bbox_inches='tight')

    plt.show()

def xdColorFitScores(trainClfs=False, fontSize=18, cuts=[0.1, 0.5, 0.9], style = ['--', '-', ':']):
    with open('trainSet.pkl', 'rb') as f:
        trainSet = pickle.load(f)
    magBins = [(18.0, 22.0), (22.0, 24.0), (24.0, 25.0), (25.0, 26.0)]
    mags = trainSet.getAllMags(band='i')
    if trainClfs:
        gaussians = [(10, 10), (10, 10), (10, 10), (10, 10)]
        X, XErr, Y = trainSet.getTrainSet(standardized=False)
        trainIdxs = trainSet.trainIndexes
        mags = mags[trainIdxs]
        clfs = []
        for i, magBin in enumerate(magBins):
            good = np.logical_and(magBin[0] < mags, mags < magBin[1])
            ngStar, ngGal = gaussians[i]
            clf = dGauss.XDClf(ngStar=ngStar, ngGal=ngGal)
            clf.fit(X[good], XErr[good], Y[good])
            clfs.append(clf)
        with open('clfsCols.pkl', 'wb') as f:
            pickle.dump(clfs, f)
    else:
        with open('clfsCols.pkl', 'rb') as f:
            clfs = pickle.load(f)
    X, XErr, Y = trainSet.getTestSet(standardized=False)
    mags = trainSet.getTestMags(band='i')
    clfXd = dGauss.XDClfs(clfs=clfs, magBins=magBins)
    posteriors = clfXd.predict_proba(X, XErr, mags)
    figPosts = plt.figure(figsize=(16, 6), dpi=120)
    axPostT = figPosts.add_subplot(1, 2, 1)
    axPostV = figPosts.add_subplot(1, 2, 2)
    axPostT.set_xlabel('P(Star|Colors)', fontsize=fontSize)
    axPostT.set_ylabel('Star Fraction', fontsize=fontSize)
    axPostV.set_xlabel(r'$\mathrm{Mag}_{cmodel}$ HSC-I', fontsize=fontSize)
    axPostV.set_ylabel('P(Star|Colors)', fontsize=fontSize)
    posteriorBins = np.linspace(0.0, 1.0, num=20)
    starFracs = np.zeros((len(posteriorBins)-1,))
    barWidth = posteriorBins[1] - posteriorBins[0]
    for i in range(len(posteriorBins)-1):
        pBin = (posteriorBins[i], posteriorBins[i+1])
        good = np.logical_and(posteriors > pBin[0], posteriors < pBin[1])
        starFracs[i] = np.sum(Y[good])*1.0/np.sum(good)
    axPostT.bar(posteriorBins[:-1], starFracs, barWidth, color='black', fill=False)
    axPostT.plot(posteriorBins, posteriorBins, linestyle='--', color='black')
    axPostT.set_xlim((0.0, 1.0))
    choice = np.random.choice(len(X), size=len(X), replace=False)
    for idx in choice:
        if Y[idx]:
            axPostV.plot(mags[idx], posteriors[idx], marker='.', markersize=2, color='blue')
        else:
            axPostV.plot(mags[idx], posteriors[idx], marker='.', markersize=2, color='red')
    axPostV.set_xlim((18.0, 25.0))
    for i, cut in enumerate(cuts):
        axPostV.plot([18.0, 25.0], [cut, cut], linestyle=style[i], color='black')
    for ax in figPosts.get_axes():
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontSize)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontSize)
    figPosts.savefig('/u/garmilla/Desktop/xdColsOnlyPosts.png', dpi=120, bbox_inches='tight')
    train = etl.Training(trainSet, clfXd)
    for i, cut in enumerate(cuts):
        if i == 0:
            figScores = train.plotScores(sType='test', xlabel=r'$\mathrm{Mag}_{cmodel}$ HSC-I full depth', linestyle=style[i],
                                         legendLabel=r'P(Star)={0}'.format(cut), standardized=False, magRange=(18.5, 25.0),
                                         suptitle=r'P(Star | Colors) full depth', kargsPred={'threshold': cut})
        else:
            figScores = train.plotScores(sType='test', fig=figScores, xlabel=r'$\mathrm{Mag}_{cmodel}$ HSC-I full depth', linestyle=style[i],
                                         legendLabel=r'P(Star)={0}'.format(cut), standardized=False, magRange=(18.5, 25.0),
                                         kargsPred={'threshold': cut})
    figScores.savefig('/u/garmilla/Desktop/xdColsOnlyScores.png', dpi=120, bbox_inches='tight')
    figBias = plt.figure(figsize=(24, 18), dpi=120)
    magString = r'$\mathrm{Mag}_{cmodel}$ HSC-I'
    colNames = ['g-r', 'r-i', 'i-z', 'z-y']
    colLims = [(0.0, 1.5), (-0.2, 2.0), (-0.2, 1.0), (-0.2, 0.4)]
    for i in range(3):
        good = np.logical_and(Y, np.logical_and(mags > magBins[i][0], mags < magBins[i][1]))
        for j in range(i*3+1, i*3+4):
            ax = figBias.add_subplot(3, 3, j)
            ax.set_title('{0} < {1} < {2}'.format(magBins[i][0], magString, magBins[i][1]), fontsize=fontSize)
            ax.set_xlabel(colNames[j-i*3-1], fontsize=fontSize)
            ax.set_ylabel(colNames[j-i*3], fontsize=fontSize)
            ax.set_xlim(colLims[j-i*3-1])
            ax.set_ylim(colLims[j-i*3])
            im = ax.scatter(X[:, j-i*3-1][good], X[:, j-i*3][good], marker='.', s=10, c=posteriors[good], vmin=0.0, vmax=1.0,
                            edgecolors='none')
        bounds = ax.get_position().bounds
        cax = figBias.add_axes([0.93, bounds[1], 0.015, bounds[3]])
        cb = plt.colorbar(im, cax=cax)
        cb.set_label(r'P(Star|Colors)', fontsize=fontSize)
        cb.ax.tick_params(labelsize=fontSize)
    for ax in figBias.get_axes():
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontSize)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontSize)
    figBias.savefig('/u/garmilla/Desktop/xdColsOnlyBias.png', dpi=120, bbox_inches='tight')

def xdColExtFitScores(trainClfs=False, fontSize=18, cuts=[0.1, 0.5, 0.9], style = ['--', '-', ':']):
    with open('trainSet.pkl', 'rb') as f:
        trainSet = pickle.load(f)
    magBins = [(18.0, 22.0), (22.0, 24.0), (24.0, 25.0), (25.0, 26.0)]
    if trainClfs:
        gaussians = [(10, 10), (10, 10), (10, 10), (10, 10)]
        X, XErr, Y = trainSet.genColExtTrainSet(mode='train')
        mags = trainSet.getTrainMags(band='i')
        clfs = []
        for i, magBin in enumerate(magBins):
            good = np.logical_and(magBin[0] < mags, mags < magBin[1])
            ngStar, ngGal = gaussians[i]
            clf = dGauss.XDClf(ngStar=ngStar, ngGal=ngGal)
            clf.fit(X[good], XErr[good], Y[good])
            clfs.append(clf)
        print "Finished training, dumping resutls in clsColsExt.pkl"
        with open('clfsColsExt.pkl', 'wb') as f:
            pickle.dump(clfs, f)
    else:
        with open('clfsColsExt.pkl', 'rb') as f:
            clfs = pickle.load(f)
    X, XErr, Y = trainSet.genColExtTrainSet(mode='test')
    mags = trainSet.getTestMags(band='i')
    clfXd = dGauss.XDClfs(clfs=clfs, magBins=magBins)
    posteriors = clfXd.predict_proba(X, XErr, mags)
    figPosts = plt.figure(figsize=(16, 6), dpi=120)
    axPostT = figPosts.add_subplot(1, 2, 1)
    axPostV = figPosts.add_subplot(1, 2, 2)
    axPostT.set_xlabel('P(Star|Colors+Extendedness)', fontsize=fontSize)
    axPostT.set_ylabel('Star Fraction', fontsize=fontSize)
    axPostV.set_xlabel(r'$\mathrm{Mag}_{cmodel}$ HSC-I', fontsize=fontSize)
    axPostV.set_ylabel('P(Star|Colors+Extendedness)', fontsize=fontSize)
    posteriorBins = np.linspace(0.0, 1.0, num=20)
    starFracs = np.zeros((len(posteriorBins)-1,))
    barWidth = posteriorBins[1] - posteriorBins[0]
    for i in range(len(posteriorBins)-1):
        pBin = (posteriorBins[i], posteriorBins[i+1])
        good = np.logical_and(posteriors > pBin[0], posteriors < pBin[1])
        starFracs[i] = np.sum(Y[good])*1.0/np.sum(good)
    axPostT.bar(posteriorBins[:-1], starFracs, barWidth, color='black', fill=False)
    axPostT.plot(posteriorBins, posteriorBins, linestyle='--', color='black')
    axPostT.set_xlim((0.0, 1.0))
    choice = np.random.choice(len(X), size=len(X), replace=False)
    for idx in choice:
        if Y[idx]:
            axPostV.plot(mags[idx], posteriors[idx], marker='.', markersize=2, color='blue')
        else:
            axPostV.plot(mags[idx], posteriors[idx], marker='.', markersize=2, color='red')
    axPostV.set_xlim((18.0, 25.0))
    for i, cut in enumerate(cuts):
        axPostV.plot([18.0, 25.0], [cut, cut], linestyle=style[i], color='black')
    for ax in figPosts.get_axes():
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontSize)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontSize)
    dirHome = os.path.expanduser('~')
    figPosts.savefig(os.path.join(dirHome, 'Desktop/xdColExtPosts.png'), dpi=120, bbox_inches='tight')
    train = etl.Training(trainSet, clfXd)
    for i, cut in enumerate(cuts):
        if i == 0:
            figScores = train.plotScores(sType='test', xlabel=r'$\mathrm{Mag}_{cmodel}$ HSC-I full depth', linestyle=style[i],
                                         legendLabel=r'P(Star)={0}'.format(cut), standardized=False, magRange=(18.5, 25.0),
                                         suptitle=r'P(Star|Colors+Extendedness) full depth', kargsPred={'threshold': cut}, colExt=True)
        else:
            figScores = train.plotScores(sType='test', fig=figScores, xlabel=r'$\mathrm{Mag}_{cmodel}$ HSC-I full depth', linestyle=style[i],
                                         legendLabel=r'P(Star)={0}'.format(cut), standardized=False, magRange=(18.5, 25.0),
                                         kargsPred={'threshold': cut}, colExt=True)
    figScores.savefig(os.path.join(dirHome, 'Desktop/xdColExtScores.png'), dpi=120, bbox_inches='tight')
    figBias = plt.figure(figsize=(24, 18), dpi=120)
    magString = r'$\mathrm{Mag}_{cmodel}$ HSC-I'
    colNames = ['g-r', 'r-i', 'i-z', 'z-y']
    colLims = [(0.0, 1.5), (-0.2, 2.0), (-0.2, 1.0), (-0.2, 0.4)]
    for i in range(3):
        good = np.logical_and(Y, np.logical_and(mags > magBins[i][0], mags < magBins[i][1]))
        for j in range(i*3+1, i*3+4):
            ax = figBias.add_subplot(3, 3, j)
            ax.set_title('{0} < {1} < {2}'.format(magBins[i][0], magString, magBins[i][1]), fontsize=fontSize)
            ax.set_xlabel(colNames[j-i*3-1], fontsize=fontSize)
            ax.set_ylabel(colNames[j-i*3], fontsize=fontSize)
            ax.set_xlim(colLims[j-i*3-1])
            ax.set_ylim(colLims[j-i*3])
            im = ax.scatter(X[:, j-i*3-1][good], X[:, j-i*3][good], marker='.', s=10, c=posteriors[good], vmin=0.0, vmax=1.0,
                            edgecolors='none')
        bounds = ax.get_position().bounds
        cax = figBias.add_axes([0.93, bounds[1], 0.015, bounds[3]])
        cb = plt.colorbar(im, cax=cax)
        cb.set_label(r'P(Star|Colors+Extendedness)', fontsize=fontSize)
        cb.ax.tick_params(labelsize=fontSize)
    for ax in figBias.get_axes():
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontSize)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontSize)
    figBias.savefig(os.path.join(dirHome, 'Desktop/xdColExtBias.png'), dpi=120, bbox_inches='tight')

def xdColExtSvmScores(trainXd=False, trainSvm=False, fontSize=18):
    with open('trainSet.pkl', 'rb') as f:
        trainSet = pickle.load(f)
    magBins = [(18.0, 22.0), (22.0, 24.0), (24.0, 25.0), (25.0, 26.0)]
    if trainXd:
        gaussians = [(10, 10), (10, 10), (10, 10), (10, 10)]
        X, XErr, Y = trainSet.genColExtTrainSet(mode='train')
        mags = trainSet.getTrainMags(band='i')
        clfs = []
        for i, magBin in enumerate(magBins):
            good = np.logical_and(magBin[0] < mags, mags < magBin[1])
            ngStar, ngGal = gaussians[i]
            clf = dGauss.XDClf(ngStar=ngStar, ngGal=ngGal)
            clf.fit(X[good], XErr[good], Y[good])
            clfs.append(clf)
        with open('clfsColsExt.pkl', 'wb') as f:
            pickle.dump(clfs, f)
    else:
        with open('clfsColsExt.pkl', 'rb') as f:
            clfs = pickle.load(f)
    if trainSvm:
        X, XErr, Y = trainSet.genColExtTrainSet(mode='train', standardized=True)
        #param_grid = {'C':[0.1, 1.0, 10.0], 'gamma': [0.1, 1.0, 10.0]}
        #clfSvm = GridSearchCV(SVC(), param_grid=param_grid)
        clfSvm = SVC(C=10.0, gamma=0.1)
        clfSvm.fit(X, Y)
        #print "Best parameters:"
        #print clfSvm.best_params_
        with open('clfSvm.pkl', 'wb') as f:
            #pickle.dump(clfSvm.best_estimator_, f)
            pickle.dump(clfSvm, f)
    else:
        with open('clfSvm.pkl', 'rb') as f:
            clfSvm = pickle.load(f)
    X, XErr, Y = trainSet.genColExtTrainSet(mode='test')
    clfXd = dGauss.XDClfs(clfs=clfs, magBins=magBins)
    train = etl.Training(trainSet, clfXd)
    figScores = train.plotScores(sType='test', xlabel=r'$\mathrm{Mag}_{cmodel}$ HSC-I full depth', linestyle='-',
                                 legendLabel=r'XD P(Star)=0.5 Cut', standardized=False, magRange=(18.5, 25.0),
                                 suptitle=r'XD vs SVM', kargsPred={'threshold': 0.5}, colExt=True)
    train = etl.Training(trainSet, clfSvm)
    figScores = train.plotScores(sType='test', fig=figScores, xlabel=r'$\mathrm{Mag}_{cmodel}$ HSC-I full depth', linestyle='--',
                                 legendLabel=r'SVM', standardized=False, magRange=(18.5, 25.0), svm=True)
    plt.show()
    figScores.savefig('/u/garmilla/Desktop/xdColExtSvmScores.png', dpi=120, bbox_inches='tight')

def xdFitEllipsePlots(trainClfs=False, fontSize=18):
    with open('trainSet.pkl', 'rb') as f:
        trainSet = pickle.load(f)
    magBins = [(18.0, 22.0), (22.0, 24.0), (24.0, 25.0), (25.0, 26.0)]
    mags = trainSet.getAllMags(band='i')
    if trainClfs:
        gaussians = [(10, 10), (10, 10), (10, 10), (10, 10)]
        X, XErr, Y = trainSet.getTrainSet(standardized=False)
        trainIdxs = trainSet.trainIndexes
        mags = mags[trainIdxs]
        clfs = []
        for i, magBin in enumerate(magBins):
            good = np.logical_and(magBin[0] < mags, mags < magBin[1])
            ngStar, ngGal = gaussians[i]
            clf = dGauss.XDClf(ngStar=ngStar, ngGal=ngGal)
            clf.fit(X[good], XErr[good], Y[good])
            clfs.append(clf)
        with open('clfsCols.pkl', 'wb') as f:
            pickle.dump(clfs, f)
    else:
        with open('clfsCols.pkl', 'rb') as f:
            clfs = pickle.load(f)
    _colLabels = ['g-r', 'r-i', 'i-z', 'z-y']
    _colsLims = [[(-0.5, 2.5), (-0.5, 3.5)], [(-0.5, 3.5), (-0.5, 1.5)], [(-0.5, 1.5), (-0.5, 1.0)]]
    magString = r'$\mathrm{Mag}_{cmodel}$ HSC-I'
    for i, magBin in enumerate(magBins):
        fig = plt.figure(figsize=(24, 12), dpi=120)
        clf = clfs[i]
        for j in range(3):
            arrGen = [j, j+1]
            subArray = (arrGen,)
            xxSub, yySub = np.meshgrid(arrGen, arrGen, indexing='ij')
            clfStar = clf.clfStar; clfGal = clf.clfGal
            axStar = fig.add_subplot(2, 3, j+1)
            axGal = fig.add_subplot(2, 3, j+4)
            for k in range(clfStar.n_components):
                draw_ellipse(clfStar.mu[k][subArray], clfStar.V[k][xxSub, yySub],
                             scales=[2], ax=axStar, ec='k', fc='blue', alpha=clfStar.alpha[k])
            for k in range(clfGal.n_components):
                draw_ellipse(clfGal.mu[k][subArray], clfGal.V[k][xxSub, yySub],
                             scales=[2], ax=axGal, ec='k', fc='red', alpha=clfGal.alpha[k])
            axStar.set_xlabel(_colLabels[j], fontsize=fontSize)
            axStar.set_ylabel(_colLabels[j+1], fontsize=fontSize)
            axGal.set_xlabel(_colLabels[j], fontsize=fontSize)
            axGal.set_ylabel(_colLabels[j+1], fontsize=fontSize)
            axStar.set_xlim(_colsLims[j][0]); axStar.set_ylim(_colsLims[j][1])
            axGal.set_xlim(_colsLims[j][0]); axGal.set_ylim(_colsLims[j][1])
            if j == 1:
                axStar.set_title(r'Stars {0} < {1} < {2}'.format(magBin[0], magString, magBin[1]), fontsize=fontSize)
                axGal.set_title(r'Galaxies {0} < {1} < {2}'.format(magBin[0], magString, magBin[1]), fontsize=fontSize)
            for ax in fig.get_axes():
                for tick in ax.xaxis.get_major_ticks():
                    tick.label.set_fontsize(fontSize)
                for tick in ax.yaxis.get_major_ticks():
                    tick.label.set_fontsize(fontSize)
        fig.savefig('/u/garmilla/Desktop/xdFitEllipses{0}-{1}.png'.format(*magBin), dpi=120, bbox_inches='tight')

def extCorrPlot(time=1200.0, gal=1000, star=100, real=100, fontSize=18):
    cat = afwTable.SimpleCatalog.readFits('/u/garmilla/Source/sgsim/examples/output/sgExExpTime{0}FWHM0.5nGal{1}nStar{2}nReal{3}.fits'.format(time,
                                          gal, star, real))
    assert len(cat) % real == 0
    mags = cat.get('cmodel.flux')
    magsPsf = cat.get('flux.psf')
    extsTrue = -2.5*np.log10(cat.get('true.flux.psf')/cat.get('true.flux.cmodel'))
    stellar = cat.get('stellar')
    assert np.sum(stellar) % real == 0
    assert (len(cat)-np.sum(stellar)) % real == 0
    x = np.vstack((mags, magsPsf))
    corrCoeffs = np.zeros((len(cat)/real,))
    extsTrueScat = np.zeros(corrCoeffs.shape)
    corrCoeffsStar = np.zeros((np.sum(stellar)/real,))
    corrCoeffsGal = np.zeros(((len(cat)-np.sum(stellar))/real,))
    countStar = 0; countGal = 0
    for i in range(len(cat)/real):
        xCut = x[:, real*i:real*(i+1)]
        good = np.logical_and(np.isfinite(xCut[0,:]), np.isfinite(xCut[1,:]))
        corrMatrix = np.corrcoef(xCut[:, good])
        corrCoeffs[i] = corrMatrix[0, 1]
        try:
            assert np.all(extsTrue[real*i:real*(i+1)] == extsTrue[real*i])
        except AssertionError:
            try:
                assert np.all(np.isnan(extsTrue[real*i:real*(i+1)]))
            except AssertionError:
                import ipdb; ipdb.set_trace()
        extsTrueScat[i] = extsTrue[real*i]
        if stellar[real*i]:
            assert np.all(stellar[real*i:real*(i+1)])
            corrCoeffsStar[countStar] = corrMatrix[0, 1]
            countStar += 1
        else:
            assert np.all(np.logical_not(stellar[real*i:real*(i+1)]))
            corrCoeffsGal[countGal] = corrMatrix[0, 1]
            countGal += 1
    rangeHist = (corrCoeffs.min(), corrCoeffs.max())
    fig = plt.figure(figsize=(16, 6), dpi=120)
    axHist = fig.add_subplot(1, 2, 1)
    axHist.set_xlabel(r'$\mathrm{Corr}[\mathrm{Mag}_{psf}, \mathrm{Mag}_{cmodel}]$', fontsize=fontSize)
    axHist.set_ylabel('Counts', fontsize=fontSize)
    axHist.hist(corrCoeffsStar, histtype='step', color='blue', bins=20, range=rangeHist, label='Stars')
    axHist.hist(corrCoeffsGal, histtype='step', color='red', bins=20, range=rangeHist, label='Galaxies')
    axHist.hist(corrCoeffs, histtype='step', color='black', bins=20, range=rangeHist, label='Total')
    axHist.set_xlim((-0.2, 1.0))
    axHist.legend(loc='upper left', fontsize=fontSize)
    axScatter = fig.add_subplot(1, 2, 2)
    axScatter.set_xlabel(r'$\mathrm{Mag}_{psf} - \mathrm{Mag}_{cmodel}$ Without Noise', fontsize=fontSize)
    axScatter.set_ylabel(r'$\mathrm{Corr}[\mathrm{Mag}_{psf}, \mathrm{Mag}_{cmodel}]$', fontsize=fontSize)
    choice = np.random.choice(len(corrCoeffs), size=len(corrCoeffs), replace=False)
    for idx in choice:
        if stellar[real*idx]:
            axScatter.plot(extsTrueScat[idx], corrCoeffs[idx], marker='.', markersize=3, color='blue')
        else:
            axScatter.plot(extsTrueScat[idx], corrCoeffs[idx], marker='.', markersize=3, color='red')
    axScatter.set_xlim((-0.01, 1.0))
    axScatter.set_ylim((-0.2, 1.0))
    for ax in fig.get_axes():
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontSize)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontSize)
    fig.savefig('/u/garmilla/Desktop/magPsfMagCmodelCorr.png', bbox_inches='tight')

def peterPlot(trainClfs=False, fontSize=16):
    with open('trainSet.pkl', 'rb') as f:
        trainSet = pickle.load(f)
    magBins = [(18.0, 22.0), (22.0, 24.0), (24.0, 25.0), (25.0, 26.0)]
    idxBest = np.argmax(trainSet.snrs, axis=1)
    idxArr = np.arange(len(trainSet.snrs))
    mags = trainSet.mags[idxArr, idxBest]
    exts = trainSet.exts[idxArr, idxBest]
    assert np.all(exts == trainSet.getAllExts(band='best')[0])
    extsErr = 1.0/trainSet.snrs[idxArr, idxBest]
    assert np.all(extsErr == trainSet.getAllExts(band='best')[1])
    if trainClfs:
        gaussians = [(10, 10), (10, 10), (10, 10), (10, 10)]
        XSub, XErrSub, Y = trainSet.getTrainSet(standardized=False)
        trainIdxs = trainSet.trainIndexes
        X = np.concatenate((XSub, exts[trainIdxs][:, None]), axis=1)
        covShapeSub = XErrSub.shape
        dimSub = covShapeSub[1]
        assert dimSub == covShapeSub[2]
        covShape = (covShapeSub[0], dimSub+1, dimSub+1)
        XErr = np.zeros(covShape)
        xxSub, yySub = np.meshgrid(np.arange(dimSub), np.arange(dimSub), indexing='ij')
        XErr[:, xxSub, yySub] = XErrSub
        XErr[:, dimSub, dimSub] = extsErr[trainIdxs]
        assert np.all(X == trainSet.genColExtTrainSet(mode='train')[0])
        assert np.all(XErr == trainSet.genColExtTrainSet(mode='train')[1])
        print "Assertions passed!"
        mags = mags[trainIdxs]
        clfs = []
        for i, magBin in enumerate(magBins):
            good = np.logical_and(magBin[0] < mags, mags < magBin[1])
            ngStar, ngGal = gaussians[i]
            clf = dGauss.XDClf(ngStar=ngStar, ngGal=ngGal)
            clf.fit(X[good], XErr[good], Y[good])
            clfs.append(clf)
        with open('clfsColsExt.pkl', 'wb') as f:
            pickle.dump(clfs, f)
    else:
        with open('clfsColsExt.pkl', 'rb') as f:
            clfs = pickle.load(f)
    XSub, XErrSub, Y = trainSet.getTestSet(standardized=False)
    testIdxs = trainSet.testIndexes
    mags = mags[testIdxs]
    exts = exts[testIdxs]
    X = np.concatenate((XSub, exts[:, None]), axis=1)
    figScat = plt.figure(figsize=(16, 12), dpi=120)
    figGauss = plt.figure(figsize=(16, 12), dpi=120)
    subArray = ([1, 4],)
    xxSub, yySub = np.meshgrid([1, 4], [1, 4], indexing='ij')
    for i, magBin in enumerate(magBins):
        clf = clfs[i]
        clfStar = clf.clfStar; clfGal = clf.clfGal
        axGauss = figGauss.add_subplot(2, 2, i+1)
        for j in range(clfStar.n_components):
            draw_ellipse(clfStar.mu[j][subArray], clfStar.V[j][xxSub, yySub],
                         scales=[2], ax=axGauss, ec='k', fc='blue', alpha=clfStar.alpha[j])
        for j in range(clfGal.n_components):
            draw_ellipse(clfGal.mu[j][subArray], clfGal.V[j][xxSub, yySub],
                         scales=[2], ax=axGauss, ec='k', fc='red', alpha=clfGal.alpha[j])
        magCut = np.logical_and(mags > magBin[0], mags < magBin[1])
        choice = np.random.choice(np.sum(magCut), size=4000, replace=False)
        axScat = figScat.add_subplot(2, 2, i+1)
        for j in choice:
            if Y[magCut][j]:
                axScat.scatter(X[:, 1][magCut][j], exts[magCut][j], marker='.', s=1, color='blue')
            else:
                axScat.scatter(X[:, 1][magCut][j], exts[magCut][j], marker='.', s=1, color='red')
        magString = r'$\mathrm{Mag}_{cmodel}$ HSC-I'
        axGauss.set_xlabel('r-i', fontsize=fontSize)
        axGauss.set_ylabel(r'$\mathrm{Mag}_{psf}-\mathrm{Mag}_{cmodel}$', fontsize=fontSize)
        axGauss.set_xlim((-0.5, 3.0))
        axGauss.set_ylim((-0.05, 0.5))
        axGauss.set_title('{0} < {1} < {2}'.format(magBin[0], magString, magBin[1]))
        axScat.set_xlabel('r-i', fontsize=fontSize)
        axScat.set_ylabel(r'$\mathrm{Mag}_{psf}-\mathrm{Mag}_{cmodel}$', fontsize=fontSize)
        axScat.set_xlim((-0.5, 3.0))
        axScat.set_ylim((-0.05, 0.5))
        axScat.set_title('{0} < {1} < {2}'.format(magBin[0], magString, magBin[1]))
    for fig in [figScat, figGauss]:
        for ax in fig.get_axes():
            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(fontSize)
            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(fontSize)
    figScat.savefig('/u/garmilla/Desktop/colVsExtScatter.png', bbox_inches='tight', dpi=120)
    figGauss.savefig('/u/garmilla/Desktop/colVsExtGaussians.png', bbox_inches='tight', dpi=120)
    return figScat, figGauss

def _getTrainWide(wideCat=1):
    if wideCat == 1:
        catWide = afwTable.SimpleCatalog.readFits('/scr/depot0/garmilla/HSC/matchDeepCoaddMeas-137520151126Cosmos1GRIZY.fits')
    elif wideCat == 2:
        catWide = afwTable.SimpleCatalog.readFits('/scr/depot0/garmilla/HSC/matchDeepCoaddMeas-137520151126Cosmos2GRIZY.fits')
    else:
        raise ValueError('I only have wide cats 1 and 2')
    trainSetWide = etl.extractTrainSet(catWide, inputs=['mag', 'ext', 'extHsmDeconvLinear'])
    return trainSetWide

def _getTrainDeep():
    catDeep = afwTable.SimpleCatalog.readFits('/scr/depot0/garmilla/HSC/matchDeepCoaddMeas-137520151126CosmosGRIZY.fits')
    trainSetDeep = etl.extractTrainSet(catDeep, inputs=['mag', 'ext', 'extHsmDeconvLinear'])
    return trainSetDeep

def _extMomentsCompHists(trainSetWide, trainSetDeep=None, wideCat=1, withDeepCat=True, fontSize=16,
                         nBins=50, magCut=(24.0, 25.0), rangeExt=(-0.02, 0.3), rangeMom=(-0.04, 0.15),
                         cutsExt=[0.001, 0.01, 0.02], cutsMom=[0.0005, 0.005, 0.01],
                         style = ['--', '-', ':']):
    if withDeepCat and trainSetDeep is None:
        raise ValueError('You have to specify a deep train set if withDeepCat is True')

    if withDeepCat:
        fig = plt.figure(figsize=(16, 12), dpi=120)
        axExtDeep = fig.add_subplot(2, 2, 1)
        axExtWide = fig.add_subplot(2, 2, 2)
        axMomDeep = fig.add_subplot(2, 2, 3)
        axMomWide = fig.add_subplot(2, 2, 4)
        axExtDeep.set_title('Full Depth', fontsize=fontSize)
        axExtWide.set_title('Wide Depth', fontsize=fontSize)
        goodDeep = np.logical_and(trainSetDeep.mags > magCut[0], trainSetDeep.mags < magCut[1])
        starsDeep = np.logical_and(trainSetDeep.Y, goodDeep)
        galsDeep = np.logical_and(np.logical_not(trainSetDeep.Y), goodDeep)
        p, binsExtDeep = np.histogram(trainSetDeep.X[:,1], bins=nBins, range=rangeExt)
        p, binsMomDeep = np.histogram(0.16*trainSetDeep.X[:,2], bins=nBins, range=rangeMom)
        axExtDeep.hist(trainSetDeep.X[starsDeep][:,1], binsExtDeep, histtype='step', color='blue')
        axExtDeep.hist(trainSetDeep.X[galsDeep][:,1], binsExtDeep, histtype='step', color='red')
        axMomDeep.hist(0.16*trainSetDeep.X[starsDeep][:,2], binsMomDeep, histtype='step', color='blue')
        axMomDeep.hist(0.16*trainSetDeep.X[galsDeep][:,2], binsMomDeep, histtype='step', color='red')
        axExtDeep.set_xlim(rangeExt)
        axMomDeep.set_xlim(rangeMom)
        axExtDeep.set_xlabel(r'$\mathrm{Mag}_{psf}-\mathrm{Mag}_{cmodel}$ HSC-I', fontsize=fontSize)
        axMomDeep.set_xlabel(r'$r_{tr}-(r_{tr})_{PSF}$ HSC-I', fontsize=fontSize)
    else:
        fig = plt.figure(figsize=(8, 12), dpi=120)
        axExtWide = fig.add_subplot(2, 1, 1)
        axMomWide = fig.add_subplot(2, 1, 2)
    goodWide = np.logical_and(trainSetWide.mags > magCut[0], trainSetWide.mags < magCut[1])
    starsWide = np.logical_and(trainSetWide.Y, goodWide)
    galsWide = np.logical_and(np.logical_not(trainSetWide.Y), goodWide)
    p, binsExtWide = np.histogram(trainSetWide.X[:,1], bins=nBins, range=rangeExt)
    p, binsMomWide = np.histogram(0.16*trainSetWide.X[:,2], bins=nBins, range=rangeMom)
    axExtWide.hist(trainSetWide.X[starsWide][:,1], binsExtWide, histtype='step', color='blue')
    axExtWide.hist(trainSetWide.X[galsWide][:,1], binsExtWide, histtype='step', color='red')
    axMomWide.hist(0.16*trainSetWide.X[starsWide][:,2], binsMomWide, histtype='step', color='blue')
    axMomWide.hist(0.16*trainSetWide.X[galsWide][:,2], binsMomWide, histtype='step', color='red')
    axExtWide.set_xlim(rangeExt)
    axMomWide.set_xlim(rangeMom)
    for i in range(len(cutsExt)):
        if withDeepCat:
            ylim = axExtDeep.get_ylim()
            axExtDeep.plot([cutsExt[i], cutsExt[i]], [ylim[0], ylim[1]], color='black', linestyle=style[i])
            ylim = axMomDeep.get_ylim()
            axMomDeep.plot([cutsMom[i], cutsMom[i]], [ylim[0], ylim[1]], color='black', linestyle=style[i])
        ylim = axExtWide.get_ylim()
        axExtWide.plot([cutsExt[i], cutsExt[i]], [ylim[0], ylim[1]], color='black', linestyle=style[i])
        ylim = axMomWide.get_ylim()
        axMomWide.plot([cutsMom[i], cutsMom[i]], [ylim[0], ylim[1]], color='black', linestyle=style[i])
    axExtWide.set_xlabel(r'$\mathrm{Mag}_{psf}-\mathrm{Mag}_{cmodel}$ HSC-I', fontsize=fontSize)
    axMomWide.set_xlabel(r'$r_{tr}-(r_{tr})_{PSF}$ HSC-I', fontsize=fontSize)
    for ax in fig.get_axes():
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontSize)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontSize)
    return fig

def extMomentsCompPlots(wideCat=1, withDeepCat=True, choiceSize=50000, fontSize=16,
                        cutsExt=[0.001, 0.01, 0.02], cutsMom=[0.001, 0.005, 0.01],
                        style = ['--', '-', ':']):
    trainSetWide = _getTrainWide(wideCat=wideCat)

    if withDeepCat:
        trainSetDeep = _getTrainDeep()

    choice = np.random.choice(len(trainSetWide.X), size=choiceSize, replace=False)

    if withDeepCat:
        figScat = plt.figure(figsize=(16, 12), dpi=120)
        axExtDeep = figScat.add_subplot(2, 2, 1)
        axExtDeep.set_title('Full Depth', fontsize=fontSize)
        axMomDeep = figScat.add_subplot(2, 2, 3)
        axExt = figScat.add_subplot(2, 2, 2)
        axExt.set_title('Wide Depth', fontsize=fontSize)
        axMom = figScat.add_subplot(2, 2, 4)
        axExtDeep.set_xlabel(r'$\mathrm{Mag}_{cmodel}$ HSC-I', fontsize=fontSize)
        axExtDeep.set_ylabel(r'$\mathrm{Mag}_{psf}-\mathrm{Mag}_{cmodel}$ HSC-I', fontsize=fontSize)
        axMomDeep.set_xlabel(r'$\mathrm{Mag}_{cmodel}$ HSC-I', fontsize=fontSize)
        axMomDeep.set_ylabel(r'$r_{tr}-(r_{tr})_{PSF}$ HSC-I', fontsize=fontSize)
        axExtDeep.set_xlim((19.0, 27.0))
        axExtDeep.set_ylim((-0.02, 0.2))
        axMomDeep.set_xlim((19.0, 27.0))
        axMomDeep.set_ylim((-0.04, 0.1))
        X = trainSetDeep.X; mag = trainSetDeep.mags
        for i in choice:
            if trainSetDeep.Y[i]:
                axExtDeep.plot(mag[i], X[i, 1], marker='.', markersize=1, color='blue')
                axMomDeep.plot(mag[i], 0.16*X[i, 2], marker='.', markersize=1, color='blue')
            else:
                axExtDeep.plot(mag[i], X[i, 1], marker='.', markersize=1, color='red')
                axMomDeep.plot(mag[i], 0.16*X[i, 2], marker='.', markersize=1, color='red')
    else:
        figScat = plt.figure(figsize=(8, 12), dpi=120)
        axExt = figScat.add_subplot(2, 1, 1)
        axMom = figScat.add_subplot(2, 1, 2)
    axExt.set_xlabel(r'$\mathrm{Mag}_{cmodel}$ HSC-I', fontsize=fontSize)
    axExt.set_ylabel(r'$\mathrm{Mag}_{psf}-\mathrm{Mag}_{cmodel}$ HSC-I', fontsize=fontSize)
    axMom.set_xlabel(r'$\mathrm{Mag}_{cmodel}$ HSC-I', fontsize=fontSize)
    axMom.set_ylabel(r'$r_{tr}-(r_{tr})_{PSF}$ HSC-I', fontsize=fontSize)
    X = trainSetWide.X; mag = trainSetWide.mags
    for i in choice:
        if trainSetWide.Y[i]:
            axExt.plot(mag[i], X[i, 1], marker='.', markersize=1, color='blue')
            axMom.plot(mag[i], 0.16*X[i, 2], marker='.', markersize=1, color='blue')
        else:
            axExt.plot(mag[i], X[i, 1], marker='.', markersize=1, color='red')
            axMom.plot(mag[i], 0.16*X[i, 2], marker='.', markersize=1, color='red')
    axExt.set_xlim((19.0, 27.0))
    axExt.set_ylim((-0.02, 0.2))
    axMom.set_xlim((19.0, 27.0))
    axMom.set_ylim((-0.04, 0.1))
    for i in range(len(cutsExt)):
        if withDeepCat:
            xlim = axExtDeep.get_xlim()
            axExtDeep.plot([xlim[0], xlim[1]], [cutsExt[i], cutsExt[i]], color='black', linestyle=style[i])
            xlim = axMomDeep.get_xlim()
            axMomDeep.plot([xlim[0], xlim[1]], [cutsMom[i], cutsMom[i]], color='black', linestyle=style[i])
        xlim = axExt.get_xlim()
        axExt.plot([xlim[0], xlim[1]], [cutsExt[i], cutsExt[i]], color='black', linestyle=style[i])
        xlim = axMom.get_xlim()
        axMom.plot([xlim[0], xlim[1]], [cutsMom[i], cutsMom[i]], color='black', linestyle=style[i])

    dirHome = os.path.expanduser('~')
    for ax in figScat.get_axes():
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontSize)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontSize)
    figScat.savefig(os.path.join(dirHome, 'Desktop/extMomComp.png'), dpi=120, bbox_inches='tight')
    if withDeepCat:
        figHist = _extMomentsCompHists(trainSetWide, trainSetDeep=trainSetDeep, wideCat=wideCat, withDeepCat=withDeepCat,
                                       fontSize=fontSize, cutsExt=cutsExt, cutsMom=cutsMom,
                                       style = ['--', '-', ':'])
    else:
        figHist = _extMomentsCompHists(trainSetWide, wideCat=wideCat, withDeepCat=withDeepCat, fontSize=fontSize,
                                       cutsExt=cutsExt, cutsMom=cutsMom,
                                       style = ['--', '-', ':'])
    figHist.savefig(os.path.join(dirHome, 'Desktop/extMomCompHists.png'), dpi=120, bbox_inches='tight')

    clf = etl.BoxClf()
    clf._setX(27.0)
    for i in range(len(cutsExt)):
        clf._setY(cutsExt[i])
        trainWideSubExt = trainSetWide.genTrainSubset(cols=[0, 1])
        train = etl.Training(trainWideSubExt, clf)
        string = r'$\Delta\mathrm{Mag}=$'
        if i == 0:
            figScoresExtWide = train.plotScores(sType='all', xlabel=r'$\mathrm{Mag}_{cmodel}$ HSC-I wide', linestyle=style[i],
                                                legendLabel=string + r'${0}$'.format(cutsExt[i]), standardized=False,
                                                suptitle=r'$\Delta\mathrm{Mag}$ HSC-I wide')
        else:
            figScoresExtWide = train.plotScores(sType='all', fig=figScoresExtWide, xlabel=r'$\mathrm{Mag}_{cmodel}$ HSC-I wide', linestyle=style[i],
                                                legendLabel=string + r'${0}$'.format(cutsExt[i]), standardized=False)
    for ax in figScoresExtWide.get_axes():
        ax.set_xlim((19.0, 26.0))
    figScoresExtWide.savefig(os.path.join(dirHome, 'Desktop/extMomCompScoresWideExt.png'), dpi=120, bbox_inches='tight')
    for i in range(len(cutsMom)):
        clf._setY(cutsMom[i]/0.16)
        trainWideSubMom = trainSetWide.genTrainSubset(cols=[0, 2])
        train = etl.Training(trainWideSubMom, clf)
        string = r'$r_{tr}-(r_{tr})_{PSF}=$'
        if i == 0:
            figScoresMomWide = train.plotScores(sType='all', xlabel=r'$\mathrm{Mag}_{cmodel}$ HSC-I wide', linestyle=style[i],
                                                legendLabel=string + r'${0}$'.format(cutsMom[i]), standardized=False,
                                                suptitle=r'$r_{tr}-(r_{tr})_{PSF}$ HSC-I wide')
        else:
            figScoresMomWide = train.plotScores(sType='all', fig=figScoresMomWide, xlabel=r'$\mathrm{Mag}_{cmodel}$ HSC-I wide', linestyle=style[i],
                                                legendLabel=string + r'${0}$'.format(cutsMom[i]), standardized=False)
    for ax in figScoresMomWide.get_axes():
        ax.set_xlim((19.0, 26.0))
    figScoresMomWide.savefig(os.path.join(dirHome, 'Desktop/extMomCompScoresWideMom.png'), dpi=120, bbox_inches='tight')
    if withDeepCat:
        for i in range(len(cutsExt)):
            clf._setY(cutsExt[i])
            trainDeepSubExt = trainSetDeep.genTrainSubset(cols=[0, 1])
            train = etl.Training(trainDeepSubExt, clf)
            string = r'$\Delta\mathrm{Mag}=$'
            if i == 0:
                figScoresExtDeep = train.plotScores(sType='all', xlabel=r'$\mathrm{Mag}_{cmodel}$ HSC-I full depth', linestyle=style[i],
                                                    legendLabel=string + r'${0}$'.format(cutsExt[i]), standardized=False,
                                                    suptitle=r'$\Delta\mathrm{Mag}$ HSC-I full depth')
            else:
                figScoresExtDeep = train.plotScores(sType='all', fig=figScoresExtDeep, xlabel=r'$\mathrm{Mag}_{cmodel}$ HSC-I full depth', linestyle=style[i],
                                                    legendLabel=string + r'${0}$'.format(cutsExt[i]), standardized=False)
        for ax in figScoresExtDeep.get_axes():
            ax.set_xlim((19.0, 26.0))
        figScoresExtDeep.savefig(os.path.join(dirHome, 'Desktop/extMomCompScoresDeepExt.png'), dpi=120, bbox_inches='tight')
        for i in range(len(cutsMom)):
            clf._setY(cutsMom[i]/0.16)
            trainDeepSubMom = trainSetDeep.genTrainSubset(cols=[0, 2])
            train = etl.Training(trainDeepSubMom, clf)
            string = r'$r_{tr}-(r_{tr})_{PSF}=$'
            if i == 0:
                figScoresMomDeep = train.plotScores(sType='all', xlabel=r'$\mathrm{Mag}_{cmodel}$ HSC-I full depth', linestyle=style[i],
                                                    legendLabel=string + r'${0}$'.format(cutsMom[i]), standardized=False,
                                                    suptitle=r'$r_{tr}-(r_{tr})_{PSF}$ HSC-I full depth')
            else:
                figScoresMomDeep = train.plotScores(sType='all', fig=figScoresMomDeep, xlabel=r'$\mathrm{Mag}_{cmodel}$ HSC-I full depth', linestyle=style[i],
                                                    legendLabel=string + r'${0}$'.format(cutsMom[i]), standardized=False)
        for ax in figScoresMomDeep.get_axes():
            ax.set_xlim((19.0, 26.0))
        figScoresMomDeep.savefig(os.path.join(dirHome, 'Desktop/extMomCompScoresDeepMom.png'), dpi=120, bbox_inches='tight')

    return figScat, figHist, figScoresExtWide, figScoresMomWide, figScoresExtDeep, figScoresMomDeep

def makeCosmosWidePlots(band='i', size=100000, fontSize=18):
    _names = ['Best', 'Median', 'Worst']
    _strMag = r'$\mathrm{Mag}_{cmodel}$'
    _strMagPsf = r'$\mathrm{Mag}_{psf}$'
    figPhot = plt.figure(figsize=(24, 12), dpi=120)
    for i, name in enumerate(_names):
        try:
            catName = '/scr/depot0/garmilla/HSC/matchS15BWide{0}.fits'.format(name)
            cat = afwTable.SimpleCatalog.readFits(catName)
        except LsstCppException:
            catName = '/home/jose/Data/matchS15BWide{0}.fits'.format(name)
            cat = afwTable.SimpleCatalog.readFits(catName)
        axScatter = figPhot.add_subplot(2, 3, i+1)
        axScatter.set_xlabel(r'{0} HSC-{1}'.format(_strMag, band.upper()), fontsize=fontSize)
        axScatter.set_ylabel(r'{0}-{1} HSC-{2}'.format(_strMagPsf, _strMag, band.upper()), fontsize=fontSize)
        axScatter.set_xlim((18.0, 26.0))
        axScatter.set_ylim((-0.01, 0.2))
        axScatter.set_title(name, fontsize=fontSize)
        ext = cat.get('{0}ext'.format(band))
        mag = cat.get('{0}mag'.format(band))
        stellar = cat.get('mu.class') == 2
        choice = np.random.choice(len(ext), size=size, replace=False)
        for j in choice:
            if ext[j] >= -0.01 and ext[j] <= 0.2 and mag[j] >= 18.0 and mag[j] <= 26.0:
                if stellar[j]:
                    axScatter.plot(mag[j], ext[j], marker='.', markersize=1, color='blue')
                else:
                    axScatter.plot(mag[j], ext[j], marker='.', markersize=1, color='red')
        good = np.logical_and(mag > 24.0, mag < 25.0)
        dataStars = ext[np.logical_and(good, stellar)]
        dataGals = ext[np.logical_and(good, np.logical_not(stellar))]
        axHist = figPhot.add_subplot(2, 3, i+4)
        axHist.set_xlabel(r'{0}-{1} HSC-{2}'.format(_strMagPsf, _strMag, band.upper()), fontsize=fontSize)
        axHist.set_ylabel('Counts', fontsize=fontSize)
        hist, bins = np.histogram(ext, bins=50, range=(-0.05, 0.3))
        axHist.hist(dataStars, bins=bins, histtype='step', color='blue', label='Stars', linewidth=2)
        axHist.hist(dataGals, bins=bins, histtype='step', color='red', label='Galaxies', linewidth=2)
    for ax in figPhot.get_axes():
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontSize)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontSize)
    dirHome = os.path.expanduser('~')
    figPhot.savefig(os.path.join(dirHome, 'Desktop/cosmosWideTruth.png'), dpi=120, bbox_inches='tight')

def makeCosmosWideScoresPlot(fontSize=18, cuts=[0.1, 0.5, 0.9], style = ['--', '-', ':']):
    _strMag = r'$\mathrm{Mag}_{cmodel}$'
    with open('clfsColsExt.pkl', 'rb') as f:
        clfs = pickle.load(f)
    magBins = [(18.0, 22.0), (22.0, 24.0), (24.0, 25.0), (25.0, 26.0)]
    _names = ['Best', 'Median', 'Worst']
    for i, name in enumerate(_names):
        with open('trainSet{0}.pkl'.format(name), 'rb') as f:
            trainSet = pickle.load(f)
        X, XErr, Y = trainSet.genColExtTrainSet(mode='test')
        mags = trainSet.getTestMags(band='i')
        clfXd = dGauss.XDClfs(clfs=clfs, magBins=magBins)
        posteriors = clfXd.predict_proba(X, XErr, mags)
        figPosts = plt.figure(figsize=(16, 6), dpi=120)
        axPostT = figPosts.add_subplot(1, 2, 1)
        axPostV = figPosts.add_subplot(1, 2, 2)
        figPosts.suptitle(name, fontsize=fontSize)
        axPostT.set_xlabel('P(Star|Colors+Extendedness)', fontsize=fontSize)
        axPostT.set_ylabel('Star Fraction', fontsize=fontSize)
        axPostV.set_xlabel(r'$\mathrm{Mag}_{cmodel}$ HSC-I', fontsize=fontSize)
        axPostV.set_ylabel('P(Star|Colors+Extendedness)', fontsize=fontSize)
        posteriorBins = np.linspace(0.0, 1.0, num=20)
        starFracs = np.zeros((len(posteriorBins)-1,))
        barWidth = posteriorBins[1] - posteriorBins[0]
        for i in range(len(posteriorBins)-1):
            pBin = (posteriorBins[i], posteriorBins[i+1])
            good = np.logical_and(posteriors > pBin[0], posteriors < pBin[1])
            starFracs[i] = np.sum(Y[good])*1.0/np.sum(good)
        axPostT.bar(posteriorBins[:-1], starFracs, barWidth, color='black', fill=False)
        axPostT.plot(posteriorBins, posteriorBins, linestyle='--', color='black')
        axPostT.set_xlim((0.0, 1.0))
        choice = np.random.choice(len(X), size=len(X), replace=False)
        for idx in choice:
            if Y[idx]:
                axPostV.plot(mags[idx], posteriors[idx], marker='.', markersize=2, color='blue')
            else:
                axPostV.plot(mags[idx], posteriors[idx], marker='.', markersize=2, color='red')
        axPostV.set_xlim((18.0, 25.0))
        for i, cut in enumerate(cuts):
            axPostV.plot([18.0, 25.0], [cut, cut], linestyle=style[i], color='black')
        for ax in figPosts.get_axes():
            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(fontSize)
            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(fontSize)
        dirHome = os.path.expanduser('~')
        figPosts.savefig(os.path.join(dirHome, 'Desktop/cosmosWidePosteriors{0}.png'.format(name)), dpi=120, bbox_inches='tight')
        train = etl.Training(trainSet, clfXd)
        for i, cut in enumerate(cuts):
            if i == 0:
                figScores = train.plotScores(sType='test',
                                             xlabel=r'{0} HSC-I Wide {1}'.format(_strMag, name),
                                             linestyle=style[i], legendLabel=r'P(Star)={0}'.format(cut),
                                             standardized=False, magRange=(18.5, 25.0),
                                             suptitle=r'P(Star|Colors+Extendedness) Wide {0}'.format(name),
                                             kargsPred={'threshold': cut}, colExt=True)
            else:
                figScores = train.plotScores(sType='test', fig=figScores,
                                             xlabel=r'{0} HSC-I Wide {1}'.format(_strMag, name),
                                             linestyle=style[i], legendLabel=r'P(Star)={0}'.format(cut),
                                             standardized=False, magRange=(18.5, 25.0),
                                             kargsPred={'threshold': cut}, colExt=True)
        figScores.savefig(os.path.join(dirHome, 'Desktop/cosmosWideScores{0}.png'.format(name)), dpi=120, bbox_inches='tight')

def cosmosWideSvmScores(trainXd=False, trainSvm=False, fontSize=18):
    _strMag = r'$\mathrm{Mag}_{cmodel}$'
    magBins = [(18.0, 22.0), (22.0, 24.0), (24.0, 25.0), (25.0, 26.0)]
    with open('clfsColsExt.pkl', 'rb') as f:
        clfs = pickle.load(f)
    with open('clfSvm.pkl', 'rb') as f:
        clfSvm = pickle.load(f)
    clfXd = dGauss.XDClfs(clfs=clfs, magBins=magBins)
    _names = ['Best', 'Median', 'Worst']
    for name in _names:
        with open('trainSet{0}.pkl'.format(name), 'rb') as f:
            trainSet = pickle.load(f)
        train = etl.Training(trainSet, clfXd)
        figScores = train.plotScores(sType='test', xlabel=r'{0} HSC-I Wide {1}'.format(_strMag, name),
                                     linestyle='-', legendLabel=r'XD P(Star)=0.5 Cut', standardized=False,
                                     magRange=(18.5, 25.0), suptitle=r'XD vs SVM {0} Seeing'.format(name),
                                     kargsPred={'threshold': 0.5}, colExt=True)
        train = etl.Training(trainSet, clfSvm)
        figScores = train.plotScores(sType='test', fig=figScores,
                                     xlabel=r'{0} HSC-I Wide {1}'.format(_strMag, name), linestyle='--',
                                     legendLabel=r'SVM', standardized=False, magRange=(18.5, 25.0), svm=True)
        dirHome = os.path.expanduser('~')
        figScores.savefig(os.path.join(dirHome, 'Desktop/cosmosWideSvmScores{0}.png').format(name), dpi=120, bbox_inches='tight')

if __name__ == '__main__':
    #cutsPlots()
    #colExPlots()
    #rcPlots(rerun='Cosmos1')
    #rcPlots(rerun='Cosmos1', snrType='mag', xRange=(23.0, 24.5), asLogX=False, xlim=(18.0, 26.0), xlabel='magnitude', polyOrder=3,
    #        featuresCuts={1:(None, None)})
    #rcPlots(rerun='Cosmos1', polyOrder=3, extType='ext', ylim=(-0.02, 0.1), xRange=(25.0, 2000.0), yRange=(-0.1, 0.50),
    #         ylabel='Mag_psf-Mag_cmodel', featuresCuts={1:(None, 0.1)})
    #rcPlots(rerun='Cosmos1', polyOrder=3, extType='ext', snrType='mag', ylim=(-0.02, 0.1), xRange=(18.0, 25.2), yRange=(-0.1, 0.50),
    #        ylabel='mag_psf-mag_cmodel (HSC-I deep)', featuresCuts={1:(None, 0.1)}, asLogX=False, xlim=(18.0, 27.0),
    #        xlabel='mag_cmodel (HSC-I deep)')
    #magExtPlots()
    #extCutRoc()
    #highPostStarsShape(trainClfs=False)
    #colExtStarsTom()
    #plt.show()
    #hstVsHscSize()
    #extMomentsCompPlots()
    #xdFitEllipsePlots()
    #plotPostMarginals()
    #xdColorFitScores()
    #extCorrPlot()
    #peterPlot()
    #xdColExtFitScores()
    #xdColExtSvmScores(trainSvm=True)
    makeCosmosWidePlots()
    #makeCosmosWideScoresPlot()
    #cosmosWideSvmScores()
