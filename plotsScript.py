import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC

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
    magsBins = etl.np.linspace(18.0, 26.0, num=30)
    magsCenters = 0.5*(magsBins[:-1] + magsBins[1:])
    complStars = etl.np.zeros(magsCenters.shape)
    purityStars = etl.np.zeros(magsCenters.shape)
    complGals = etl.np.zeros(magsCenters.shape)
    purityGals = etl.np.zeros(magsCenters.shape)
    YPred = etl.np.logical_not(YProb < threshold)
    for i in range(len(magsCenters)):
        magCut = etl.np.logical_and(mags > magsBins[i], mags < magsBins[i+1])
        predCut = YPred[magCut]; truthCut = Y[magCut]
        goodStars = etl.np.logical_and(predCut, truthCut)
        goodGals = etl.np.logical_and(etl.np.logical_not(predCut), etl.np.logical_not(truthCut))
        if etl.np.sum(truthCut) > 0:
            complStars[i] = float(etl.np.sum(goodStars))/etl.np.sum(truthCut)
        if etl.np.sum(predCut) > 0:
            purityStars[i] = float(etl.np.sum(goodStars))/etl.np.sum(predCut)
        if len(truthCut) - etl.np.sum(truthCut) > 0:
            complGals[i] = float(etl.np.sum(goodGals))/(len(truthCut) - etl.np.sum(truthCut))
        if len(predCut) - etl.np.sum(predCut) > 0:
            purityGals[i] = float(etl.np.sum(goodGals))/(len(predCut) - etl.np.sum(predCut))

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
        good = etl.np.logical_and(magBin[0] < mags, mags < magBin[1])
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
    YProb = etl.np.zeros(Y.shape)
    YPred = etl.np.zeros(Y.shape, dtype=bool)
    for i, magBin in enumerate(magBins):
        good = etl.np.logical_and(magBin[0] < mags, mags < magBin[1])
        YProb[good] = clfs[i].predict_proba(X[good], XErr[good])
        YPred[good] = clfs[i].predict(X[good], XErr[good])
        mpl.rcParams['figure.figsize'] = 16, 10
        plotPosterior(X[good][Y[good]], YPred[good][Y[good]], magBin=magBin)
        #plt.tight_layout()
        #plotPosterior(X[good][etl.np.logical_not(Y[good])], YPred[good][etl.np.logical_not(Y[good])])
        plt.savefig('/u/garmilla/Desktop/colorColorStars{0}-{1}.png'.format(magBin[0], magBin[1]), bbox_inches='tight')
        #plt.savefig('/u/garmilla/Desktop/colorColorGalaxies{0}-{1}.png'.format(magBin[0], magBin[1]), bbox_inches='tight')
    print "Score={0}".format(etl.np.sum(YPred == Y)*1.0/len(Y))


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

cri = (0.00231810,  0.01284177, -0.03068248)
ciz = (0.00130204, -0.16922042, -0.01374245)
czi = (-0.00680620,  0.01353969,  0.01479369)
Aiz = ciz[2] - czi[2] 
Biz = 1.0 + ciz[1] + czi[1]
Ari = cri[2]
Bri = 1.0 + cri[1]

def _fromHscToSdss(rHsc, iHsc, zHsc):
    izHsc = iHsc - zHsc
    riHsc = rHsc - iHsc
    Ciz = ciz[0] - czi[0] - izHsc
    izSdss1 = (-Biz + np.sqrt(Biz**2-4*Aiz*Ciz))/2/Aiz
    izSdss2 = (-Biz - np.sqrt(Biz**2-4*Aiz*Ciz))/2/Aiz
    Cri1 = cri[0] - ciz[0] - ciz[1]*izSdss1 - ciz[2]*izSdss1**2 - riHsc
    Cri2 = cri[0] - ciz[0] - ciz[1]*izSdss2 - ciz[2]*izSdss2**2 - riHsc
    riSdss1 = (-Bri + np.sqrt(Bri**2-4*Ari*Cri1))/2/Ari
    riSdss2 = (-Bri - np.sqrt(Bri**2-4*Ari*Cri1))/2/Ari
    riSdss3 = (-Bri + np.sqrt(Bri**2-4*Ari*Cri2))/2/Ari
    riSdss4 = (-Bri - np.sqrt(Bri**2-4*Ari*Cri2))/2/Ari
    sols = [(riSdss1, izSdss1), (riSdss2, izSdss1), (riSdss3, izSdss2), (riSdss4, izSdss2)]
    return sols

def _fromSdssToHsc(rSdss, iSdss, zSdss):
    riSdss = rSdss - iSdss 
    izSdss = iSdss - zSdss 
    ziSdss = zSdss - iSdss 
    rHsc = rSdss + cri[0] + cri[1]*riSdss + cri[2]*riSdss**2
    iHsc = iSdss + ciz[0] + ciz[1]*izSdss + ciz[2]*izSdss**2
    zHsc = zSdss + czi[0] + czi[1]*ziSdss + czi[2]*ziSdss**2
    return rHsc, iHsc, zHsc

def highPostStarsShape():
    with open('trainSet.pkl', 'rb') as f:
        trainSet = pickle.load(f)

    fontSize = 18
    magBins = [(18.0, 22.0), (22.0, 24.0), (24.0, 25.0), (25.0, 26.0)]
    gaussians = [(10, 10), (10, 10), (10, 10), (10, 10)]
    X, XErr, Y = trainSet.getTrainSet(standardized=False)
    mags = trainSet.getTrainMags()
    #X = X[:5000]; XErr[:5000]; Y = Y[:5000]; mags = mags[:5000]
    clfs = []
    for i, magBin in enumerate(magBins):
        good = etl.np.logical_and(magBin[0] < mags, mags < magBin[1])
        ngStar, ngGal = gaussians[i]
        clf = dGauss.XDClf(ngStar=ngStar, ngGal=ngGal)
        clf.fit(X[good], XErr[good], Y[good])
        clfs.append(clf)
    X, XErr, Y = trainSet.getTestSet(standardized=False)
    mags = trainSet.getTestMags()
    exts = trainSet.getTestExts()
    YProb = etl.np.zeros(Y.shape)
    for i, magBin in enumerate(magBins):
        good = etl.np.logical_and(magBin[0] < mags, mags < magBin[1])
        YProb[good] = clfs[i].predict_proba(X[good], XErr[good])
    good = YProb > 0.9
    fig = plt.figure()
    plt.scatter(mags[good], exts[good], marker='.', s=1)
    plt.xlabel('Mag_cmodel')
    plt.ylabel('Mag_psf-Mag_cmodel')
    plt.title('Objects with P(Star|Colors)>0.9')
    fig.savefig('/u/garmilla/Desktop/colorStarsShapes.png', bbox_inches='tight')
    plt.show()

def rc1Plots(rerun='Cosmos1', polyOrder=3, extType='extHsmDeconv', ylim=(-2, 5), xRange=(10.0, 3000.0), yRange=(-20, 20), ylabel='rTrace',
             featuresCuts={1:(None, 1.0)}):
    if rerun == 'Cosmos1':
        cat = afwTable.SimpleCatalog.readFits('/scr/depot0/garmilla/HSC/matchDeepCoaddMeas-136120151104Cosmos1GRIZY.fits')
    elif rerun == 'Cosmos2':
        cat = afwTable.SimpleCatalog.readFits('/scr/depot0/garmilla/HSC/matchDeepCoaddMeas-136120151104Cosmos2GRIZY.fits')
    elif rerun == 'Cosmos':
        cat = afwTable.SimpleCatalog.readFits('/scr/depot0/garmilla/HSC/matchDeepCoaddMeas-136120151104CosmosGRIZY.fits')

    if rerun in ['Cosmos1', 'Cosmos2']:
        trainSet = etl.extractTrainSet(cat, inputs=['snrPsf', extType], polyOrder=polyOrder)
    elif rerun == 'Cosmos':
        trainSet = etl.extractTrainSet(cat, inputs=['snrPsf', extType], bands=['g', 'r', 'i', 'z', 'y'], polyOrder=polyOrder)

    clf = dGauss.logisticFit(trainSet, featuresCuts=featuresCuts)
    train = etl.Training(trainSet, clf)
    figPMap = train.plotPMap((5, 3000), ylim, 200, 200, xlabel='S/N', ylabel=ylabel, asLogX=True, cbLabel='pStar')

    figPMap.savefig('/u/garmilla/pMap{0}.png'.format(rerun))
    train.printPolynomial(['snr', 'magDiff'])
    if rerun in ['Cosmos1', 'Cosmos2']:
        figBdy = train.plotBoundary(0, xRange=xRange, overPlotData=True, ylim=ylim, asLogX=True, xlim=(5.0, 3000), yRange=yRange,
                                    xlabel='S/N', ylabel=ylabel)
    elif rerun == 'Cosmos':
        figBdy = train.plotBoundary(0, xRange=xRange, overPlotData=True, ylim=ylim, asLogX=True, xlim=(5.0, 3000), yRange=yRange,
                                    xlabel='S/N', ylabel=ylabel, frac=0.006)
    figBdy.savefig('/u/garmilla/boundary{0}.png'.format(rerun))
    mpl.rcParams['figure.figsize'] = 12, 6
    figScores = train.plotScores(magRange=(18.0, 26.0))
    figScores.savefig('/u/garmilla/scores{0}.png'.format(rerun))

def magExtPlots(rerun='Cosmos1'):
    if rerun == 'Cosmos1':
        cat = afwTable.SimpleCatalog.readFits('/scr/depot0/garmilla/HSC/matchDeepCoaddMeas-136120151104Cosmos1GRIZY.fits')
    elif rerun == 'Cosmos2':
        cat = afwTable.SimpleCatalog.readFits('/scr/depot0/garmilla/HSC/matchDeepCoaddMeas-136120151104Cosmos2GRIZY.fits')
    elif rerun == 'Cosmos':
        cat = afwTable.SimpleCatalog.readFits('/scr/depot0/garmilla/HSC/matchDeepCoaddMeas-136120151104CosmosGRIZY.fits')

    #fig = utils.makeMagExPlot(cat, 'i', withLabels=True, trueSample=True, frac=0.04)
    fig = utils.makeExtHist(cat, 'i', withLabels=True, magCuts=[(18.0, 22.0), (22.0, 24.0), (24.0, 25.0), (25.0, 26.0)], xlim=(-0.01, 0.5))

    #plt.savefig('/u/garmilla/Desktop/magExtDist.png', bbox_inches='tight')

def extCutRoc(rerun='Cosmos1', extType='ext', snrCut=(10, 30), nConnect=20):
    if rerun == 'Cosmos1':
        cat = afwTable.SimpleCatalog.readFits('/scr/depot0/garmilla/HSC/matchDeepCoaddMeas-136120151104Cosmos1GRIZY.fits')
    elif rerun == 'Cosmos2':
        cat = afwTable.SimpleCatalog.readFits('/scr/depot0/garmilla/HSC/matchDeepCoaddMeas-136120151104Cosmos2GRIZY.fits')
    elif rerun == 'Cosmos':
        cat = afwTable.SimpleCatalog.readFits('/scr/depot0/garmilla/HSC/matchDeepCoaddMeas-136120151104CosmosGRIZY.fits')

    if rerun in ['Cosmos1', 'Cosmos2']:
        trainSet = etl.extractTrainSet(cat, inputs=[extType], polyOrder=1)
    elif rerun == 'Cosmos':
        trainSet = etl.extractTrainSet(cat, inputs=[extType], bands=['g', 'r', 'i', 'z', 'y'], polyOrder=1)

    clf = LinearSVC()
    X, Y = trainSet.getAllSet(standardized=False)
    snrs = trainSet.snrs
    inSnrCut = etl.np.logical_and(snrs > snrCut[0], snrs < snrCut[1])
    clf.fit(X, Y)

    cutRange = etl.np.linspace(-0.02, 2.0, num=500)
    xxStars = etl.np.zeros(cutRange.shape)
    yyStars = etl.np.zeros(cutRange.shape)
    xxGals = etl.np.zeros(cutRange.shape)
    yyGals = etl.np.zeros(cutRange.shape)

    Xcut = X[inSnrCut]; Ycut = Y[inSnrCut]
    for i, cut in enumerate(cutRange):
        clf.coef_[0][0] = -1.0
        clf.intercept_[0] = cut
        Ypred = clf.predict(Xcut)
        goodStars = etl.np.logical_and(Ypred, Ycut)
        goodGals = etl.np.logical_and(etl.np.logical_not(Ypred), etl.np.logical_not(Ycut))
        if etl.np.sum(Ycut) > 0:
            xxStars[i] = float(etl.np.sum(goodStars))/etl.np.sum(Ycut)
        if etl.np.sum(Ypred) > 0:
            yyStars[i] = float(etl.np.sum(goodStars))/etl.np.sum(Ypred)
        if len(Ycut) - etl.np.sum(Ycut) > 0:
            xxGals[i] = float(etl.np.sum(goodGals))/(len(Ycut) - etl.np.sum(Ycut))
        if len(Ypred) - etl.np.sum(Ypred) > 0:
            yyGals[i] = float(etl.np.sum(goodGals))/(len(Ypred) - etl.np.sum(Ypred))

    fig = plt.figure(figsize=(16, 6))
    fig.suptitle('{0} < S/N < {1}'.format(*snrCut), fontsize=18)
    axHist = fig.add_subplot(1, 2, 1)
    axRoc = fig.add_subplot(1, 2, 2)

    hist, bins = etl.np.histogram(Xcut[:,0], bins=50, range=(-0.05, 0.5))
    dataStars = Xcut[:,0][Ycut]
    dataGals = Xcut[:,0][etl.np.logical_not(Ycut)]
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
    cat = afwTable.SimpleCatalog.readFits('/scr/depot0/garmilla/HSC/matchDeepCoaddMeas-136120151104Cosmos1Iiphot.fits')
    extHsc = -2.5*etl.np.log10(cat.get('flux.psf.i')/cat.get('cmodel.flux.i'))
    extHst = cat.get('mu.max')-cat.get('mag.auto')
    snr = cat.get('flux.psf.i')/cat.get('flux.psf.err.i')
    good = etl.np.logical_and(snr > snrCut[0], snr < snrCut[1])

    fig = plt.figure()
    plt.scatter(extHst[good], extHsc[good], marker='.', s=1)
    plt.title('{0} < S/N < {1}'.format(snrCut[0], snrCut[1]))
    plt.xlabel('mu_max-mag_auto (HST)')
    plt.ylabel('mag_psf-mag_cmodel (HSC)')
    plt.xlim((-5.0, 1.0))
    plt.ylim((-0.1, 2.0))

    fig.savefig('/u/garmilla/Desktop/hstVsHscSizeSnr{0}-{1}.png'.format(snrCut[0], snrCut[1]), bbox_inches='tight')

    plt.show()

if __name__ == '__main__':
    #cutsPlots()
    #colExPlots()
    #rc1Plots(rerun='Cosmos')
    #rc1Plots(rerun='Cosmos1', polyOrder=3, extType='ext', ylim=(-0.02, 0.1), xRange=(25.0, 2000.0), yRange=(-0.1, 0.50),
    #         ylabel='Mag_psf-Mag_cmodel', featuresCuts={1:(None, 0.1)})
    #magExtPlots()
    #extCutRoc()
    highPostStarsShape()
    #hstVsHscSize()
