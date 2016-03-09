import os
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.svm import LinearSVC
from sklearn.neighbors.kde import KernelDensity
from astroML.plotting.tools import draw_ellipse

import supervisedEtl as etl
import dGauss

import lsst.afw.table as afwTable

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

def _getMsGrHsc(ri, A, B, C, D, E):
    return A*(1.0 - np.exp(B*ri**3 + C*ri**2 + D*ri + E))

def _getMsIzHsc(ri, A, B, C, D):
    return A + B*ri + C*ri**2 + D*ri**3

def _fitGriSlHsc(gr, ri, sigma=None):
    popt, pcov = curve_fit(_getMsGrHsc, ri, gr, p0=(1.39, -4.9, -2.45, -1.68, -0.050), sigma=sigma)
    return popt, pcov

def _fitRizSlHsc(ri, iz, sigma=None):
    popt, pcov = curve_fit(_getMsIzHsc, ri, iz, p0=(0.0, 0.5, 0.0, 0.0), sigma=sigma)
    return popt, pcov

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

def _makeIsoDensityPlot(ri, gr=None, iz=None, withHsc=False, paramTuple=None, minDens=None, sigma=None):
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

    riSl = np.linspace(-0.05, 2.5, num=100)
    if withHsc:
        if paramTuple is None:
            assert minDens is not None
            densValues = np.exp(kde.score_samples(values))
            good = np.logical_and(True, densValues > minDens)
            if sigma is not None:
                sigma = sigma[good]
            if iz is None:
                popt, pcov = _fitGriSlHsc(gr[good], ri[good], sigma=sigma)
            else:
                popt, pcov = _fitRizSlHsc(ri[good], iz[good], sigma=sigma)
            print popt
            paramTuple = popt
        if iz is None:
            grSl = _getMsGrHsc(riSl, *paramTuple)
        else:
            izSl = _getMsIzHsc(riSl, *paramTuple)
    else:
        if iz is None:
            grSl = _getMsGrSdss(riSl)
        else:
            raise ValueError("I don't have a riz fir for SDSS stars. Fit to HSC stars instead.")

    fig = plt.figure(figsize=(16, 6), dpi=120)
    axData = fig.add_subplot(1, 2, 1)
    if iz is None:
        axData.scatter(gr[good], ri[good], marker='.', s=1, color='blue')
        axData.scatter(gr[np.logical_not(good)], ri[np.logical_not(good)], marker='.', s=1, color='red')
        axData.set_xlim((-0.05, 1.7))
        axData.set_ylim((-0.05, 2.5))
        axData.set_xlabel('g-r')
        axData.set_ylabel('r-i')
        axData.plot(grSl, riSl, color='black')
    else:
        axData.scatter(ri[good], iz[good], marker='.', s=1, color='blue')
        axData.scatter(ri[np.logical_not(good)], iz[np.logical_not(good)], marker='.', s=1, color='red')
        axData.set_xlim((-0.05, 2.5))
        axData.set_ylim((-0.05, 1.2))
        axData.set_xlabel('r-i')
        axData.set_ylabel('i-z')
        axData.plot(riSl, izSl, color='black')
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
        axContour.set_xlabel('g-r')
        axContour.set_ylabel('r-i')
    else:
        axContour.set_xlim((-0.05, 2.5))
        axContour.set_ylim((-0.05, 1.2))
        axContour.plot(riSl, izSl, color='black')
        axContour.set_xlabel('r-i')
        axContour.set_ylabel('i-z')
    return fig

def makePhotParallaxPlot():
    paramsGri = (1.30038049, -7.78059699, -0.71791215, -0.76761088, -0.19133522)
    paramsRiz = (-0.01068287, 0.59929634,-0.19457149, 0.05357661)
    riSl = np.linspace(-0.05, 2.5, num=100)
    grSl = _getMsGrHsc(riSl, *paramsGri)
    izSl = _getMsIzHsc(riSl, *paramsRiz)
    grSdss, riSdss, izSdss = _fromHscToSdss(grSl, riSl, izSl, giveClosest=True)
    absMagRSdss = _getAbsoluteMagR(riSdss)
    absMagRHsc = absMagRSdss + cri[0] + cri[1]*riSdss + cri[2]*riSdss**2
    absMagGHsc = absMagRHsc + grSl

    fig = plt.figure(figsize=(16, 6), dpi=120)
    axGr = fig.add_subplot(1, 2, 1)
    axRi = fig.add_subplot(1, 2, 2)
    axGr.plot(grSl, absMagGHsc, color='black')
    axRi.plot(riSl, absMagRHsc, color='black')
    axGr.set_xlabel('g-r')
    axRi.set_xlabel('r-i')
    axGr.set_ylabel('Absolute Magnitude HSC-G')
    axRi.set_ylabel('Absolute Magnitude HSC-R')
    return fig

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

def plotPostMarginals(trainClfs=False):
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
        fig = plt.figure(figsize=(20, 12), dpi=120)
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
            im = axCmap.imshow(Z, extent=[xRange[0], xRange[-1], yRange[0], yRange[-1]], aspect='auto', origin='lower')
            cb = plt.colorbar(im)
            axCmap.set_xlabel(cNames[i])
            axCmap.set_ylabel(cNames[i+1])
            axScat = fig.add_subplot(2, 3, i+4)
            for k in range(len(X[good])):
                if Y[good][k]:
                    axScat.plot(X[good][k, i], X[good][k, i+1], marker='.', markersize=1, color='blue')
                else:
                    axScat.plot(X[good][k, i], X[good][k, i+1], marker='.', markersize=1, color='red')
            axScat.set_xlim(colsLims[i][0])
            axScat.set_ylim(colsLims[i][1])
            axScat.set_xlabel(cNames[i])
            axScat.set_ylabel(cNames[i+1])
        fig.suptitle('{0} < Mag HSC-I < {1}'.format(*magBins[j]))
        dirHome = os.path.expanduser('~')
        fileFig = os.path.join(dirHome, 'Desktop/xdFitVsData{0}-{1}.png'.format(*magBins[j]))
        fig.savefig(fileFig, dpi=120, bbox_inches='tight')

def makeTomPlots(dKpc, exts, magRAbsHsc, X, magRHsc, withProb=False, YProbGri=None, YProbRiz=None,
                 title='Pure Morphology Classifier'):
    if withProb:
        assert YProbGri is not None
        assert YProbRiz is not None
    fig = plt.figure(figsize=(10, 12), dpi=120)
    axExt = fig.add_subplot(3, 2, 1)
    axExt.scatter(dKpc, exts, marker='.', s=1)
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
    axCol.scatter(dKpc, X[:,1], marker='.', s=1)
    axCol.set_xlim((0.0, 50.0))
    axCol.set_ylim((-0.2, 2.0))
    axCol.set_xlabel('d (kpc)')
    axCol.set_ylabel('r-i')
    axMag = fig.add_subplot(3, 2, 4)
    axMag.scatter(dKpc, magRHsc, marker='.', s=1)
    axMag.set_xlim((0.0, 50.0))
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

def truthStarsTom(frac=None):
    with open('trainSet.pkl', 'rb') as f:
        trainSet = pickle.load(f)
    X, XErr, Y = trainSet.getTestSet(standardized=False)
    mags = trainSet.getTestMags()
    exts = trainSet.getTestExts()
    X = X[Y]; mags = mags[Y]; exts = exts[Y]
    magRAbsHsc, dKpc = getParallax(mags[:,0], mags[:,1], mags[:,2], mags[:,3])
    if frac is not None:
        choice = np.random.choice(len(X), size=int(frac*len(X)), replace=False)
        dKpc = dKpc[choice]; exts = exts[choice]; magRAbsHsc = magRAbsHsc[choice]; X = X[choice]; mags = mags[choice]
    fig = makeTomPlots(dKpc, exts[:,1], magRAbsHsc, X, mags[:,1], title='True Stars')
    dirHome = os.path.expanduser('~')
    fileFig = os.path.join(dirHome, 'Desktop/truthStars.png')
    fig.savefig(fileFig, dpi=120, bbox_inches='tight')
    return fig

def boxStarsTom():
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

def colExtStarsTom(trainClfs=False):
    with open('trainSet.pkl', 'rb') as f:
        trainSet = pickle.load(f)
    magBins = [(18.0, 22.0), (22.0, 24.0), (24.0, 25.0), (25.0, 26.0)]
    idxBest = np.argmax(trainSet.snrs, axis=1)
    idxArr = np.arange(len(trainSet.snrs))
    mags = trainSet.mags[idxArr, idxBest]
    exts = trainSet.exts[idxArr, idxBest]
    extsErr = 1.0/trainSet.snrs[idxArr, idxBest]
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
    X = np.concatenate((XSub, exts[testIdxs][:, None]), axis=1)
    covShapeSub = XErrSub.shape
    dimSub = covShapeSub[1]
    assert dimSub == covShapeSub[2]
    covShape = (covShapeSub[0], dimSub+1, dimSub+1)
    XErr = np.zeros(covShape)
    xxSub, yySub = np.meshgrid(np.arange(dimSub), np.arange(dimSub), indexing='ij')
    XErr[:, xxSub, yySub] = XErrSub
    XErr[:, dimSub, dimSub] = extsErr[testIdxs]
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
    mags = trainSet.getTestMags()
    magRAbsHsc, dKpc = getParallax(mags[good,0], mags[good,1], mags[good,2], mags[good,3])
    exts = exts[testIdxs][good]
    mags = mags[good]
    X = X[good]
    fig = makeTomPlots(dKpc, exts, magRAbsHsc, X, mags[:,1], withProb=True,
                       YProbGri=YProbGri[good], YProbRiz=YProbRiz[good],
                       title='Morphology+Colors')
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

def peterPlot(trainClfs=False):
    with open('trainSet.pkl', 'rb') as f:
        trainSet = pickle.load(f)
    magBins = [(18.0, 22.0), (22.0, 24.0), (24.0, 25.0), (25.0, 26.0)]
    idxBest = np.argmax(trainSet.snrs, axis=1)
    idxArr = np.arange(len(trainSet.snrs))
    mags = trainSet.mags[idxArr, idxBest]
    exts = trainSet.exts[idxArr, idxBest]
    extsErr = 1.0/trainSet.snrs[idxArr, idxBest]
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
        mags = mags[trainIdxs]
        clfs = []
        for i, magBin in enumerate(magBins):
            good = np.logical_and(magBin[0] < mags, mags < magBin[1])
            ngStar, ngGal = gaussians[i]
            clf = dGauss.XDClf(ngStar=ngStar, ngGal=ngGal)
            clf.fit(X[good], XErr[good], Y[good])
            clfs.append(clf)
        with open('clfsColsExtPeter.pkl', 'wb') as f:
            pickle.dump(clfs, f)
    else:
        with open('clfsColsExtPeter.pkl', 'rb') as f:
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
        choice = np.random.choice(np.sum(magCut), size=2000, replace=False)
        axScat = figScat.add_subplot(2, 2, i+1)
        for j in choice:
            if Y[magCut][j]:
                axScat.scatter(X[:, 1][magCut][j], exts[magCut][j], marker='.', s=1, color='blue')
            else:
                axScat.scatter(X[:, 1][magCut][j], exts[magCut][j], marker='.', s=1, color='red')
        axGauss.set_xlabel('r-i')
        axGauss.set_ylabel('Mag_psf-Mag_cmodel')
        axGauss.set_xlim((-0.5, 3.0))
        axGauss.set_ylim((-0.05, 1.0))
        axGauss.set_title('{0} < Mag_cmodel_i < {1}'.format(*magBin))
        axScat.set_xlabel('r-i')
        axScat.set_ylabel('Mag_psf-Mag_cmodel')
        axScat.set_xlim((-0.5, 3.0))
        axScat.set_ylim((-0.05, 1.0))
        axScat.set_title('{0} < Mag_cmodel_i < {1}'.format(*magBin))
    figScat.savefig('/u/garmilla/Desktop/colVsExtScatter.png', bbox_inches='tight')
    figGauss.savefig('/u/garmilla/Desktop/colVsExtGaussians.png', bbox_inches='tight')
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
                         nBins=50, magCut=(24.0, 25.0), rangeExt=(-0.02, 0.3), rangeMom=(-0.2, 0.3),
                         cutsExt = [0.001, 0.01, 0.02], cutsMom = [0.0005, 0.005, 0.01],
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
        p, binsExtDeep = np.histogram(trainSetDeep.X[:,0], bins=nBins, range=rangeExt)
        p, binsMomDeep = np.histogram(trainSetDeep.X[:,1], bins=nBins, range=rangeMom)
        axExtDeep.hist(trainSetDeep.X[starsDeep][:,0], binsExtDeep, histtype='step', color='blue')
        axExtDeep.hist(trainSetDeep.X[galsDeep][:,0], binsExtDeep, histtype='step', color='red')
        axMomDeep.hist(trainSetDeep.X[starsDeep][:,1], binsMomDeep, histtype='step', color='blue')
        axMomDeep.hist(trainSetDeep.X[galsDeep][:,1], binsMomDeep, histtype='step', color='red')
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
    p, binsMomWide = np.histogram(trainSetWide.X[:,2], bins=nBins, range=rangeMom)
    axExtWide.hist(trainSetWide.X[starsWide][:,1], binsExtWide, histtype='step', color='blue')
    axExtWide.hist(trainSetWide.X[galsWide][:,1], binsExtWide, histtype='step', color='red')
    axMomWide.hist(trainSetWide.X[starsWide][:,2], binsMomWide, histtype='step', color='blue')
    axMomWide.hist(trainSetWide.X[galsWide][:,2], binsMomWide, histtype='step', color='red')
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

def extMomentsCompPlots(wideCat=1, withDeepCat=False, choiceSize=50000, fontSize=16,
                        cutsExt = [0.001, 0.01, 0.02], cutsMom = [0.01, 0.02, 0.03],
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
                axMomDeep.plot(mag[i], 0.16*X[i, 1], marker='.', markersize=1, color='blue')
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
                                       fontSize=fontSize, cutsExt = [0.001, 0.01, 0.02], cutsMom = [0.0005, 0.005, 0.01],
                                       style = ['--', '-', ':'])
    else:
        figHist = _extMomentsCompHists(trainSetWide, wideCat=wideCat, withDeepCat=withDeepCat, fontSize=fontSize,
                                       cutsExt = [0.001, 0.01, 0.02], cutsMom = [0.0005, 0.005, 0.01],
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
                                                legendLabel=string + r'${0}$'.format(cutsExt[i]), standardized=False)
        else:
            figScoresExtWide = train.plotScores(sType='all', fig=figScoresExtWide, xlabel=r'$\mathrm{Mag}_{cmodel}$ HSC-I wide', linestyle=style[i],
                                                legendLabel=string + r'${0}$'.format(cutsExt[i]), standardized=False)
    for ax in figScoresExtWide.get_axes():
        ax.set_xlim((19.0, 26.0))
    figScoresExtWide.savefig(os.path.join(dirHome, 'Desktop/extMomCompScoresWideExt.png'), dpi=120, bbox_inches='tight')
    for i in range(len(cutsMom)):
        clf._setY(cutsMom[i])
        trainWideSubMom = trainSetWide.genTrainSubset(cols=[0, 2])
        train = etl.Training(trainWideSubMom, clf)
        string = r'$r_{tr}-(r_{tr})_{PSF}=$'
        if i == 0:
            figScoresMomWide = train.plotScores(sType='all', xlabel=r'$\mathrm{Mag}_{cmodel}$ HSC-I wide', linestyle=style[i],
                                                legendLabel=string + r'${0}$'.format(cutsMom[i]), standardized=False)
        else:
            figScoresMomWide = train.plotScores(sType='all', fig=figScoresMomWide, xlabel=r'$\mathrm{Mag}_{cmodel}$ HSC-I wide', linestyle=style[i],
                                                legendLabel=string + r'${0}$'.format(cutsMom[i]), standardized=False)
    for ax in figScoresMomWide.get_axes():
        ax.set_xlim((19.0, 26.0))
    figScoresMomWide.savefig(os.path.join(dirHome, 'Desktop/extMomCompScoresWideMom.png'), dpi=120, bbox_inches='tight')
    return figScat, figHist, figScoresExtWide

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
    extMomentsCompPlots()
