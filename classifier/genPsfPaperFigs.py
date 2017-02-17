import os
import pickle
import numpy as np
from scipy.optimize import brentq
if os.environ.get("DISPLAY") is None:
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
else:
    import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.svm import LinearSVC, SVC
from sklearn.grid_search import GridSearchCV

import supervisedEtl as etl

def trainSVC(snrType='snrPsf', extType='ext'):
    with open('trainSetGRIZY.pkl', 'rb') as f:
        trainSet = pickle.load(f)
    clf = SVC()
    for i, band in enumerate(['g', 'r', 'i', 'z', 'y']):
        idxSnr = trainSet.names.index(snrType + '_' + band)
        idxExt = trainSet.names.index(extType + '_' + band)
        trainSetBand = trainSet.genTrainSubset(cuts={idxExt:(None, 0.2)}, cols=[idxSnr, idxExt])
        Xtrain, Y = trainSetBand.getAllSet()
        clf.fit(Xtrain, Y)
        with open('svc{0}.pkl'.format(band.upper()), 'wb') as f:
            pickle.dump(clf, f)

def trainSVCWMag(magType='i', extType='ext'):
    with open('trainSetGRIZY.pkl', 'rb') as f:
        trainSet = pickle.load(f)
    clf = SVC()
    for i, band in enumerate(['g', 'r', 'i', 'z', 'y']):
        idxMag = trainSet.names.index('mag' + '_' + magType)
        idxExt = trainSet.names.index(extType + '_' + band)
        trainSetBand = trainSet.genTrainSubset(cuts={idxExt:(None, 0.2)}, cols=[idxMag, idxExt])
        Xtrain, Y = trainSetBand.getAllSet()
        clf.fit(Xtrain, Y)
        with open('svcWMag{0}.pkl'.format(band.upper()), 'wb') as f:
            pickle.dump(clf, f)

def trainSVCWSeeing(snrType='snrPsf', extType='ext', concatBands=False):
    if concatBands:
        with open('trainSetConcatGRIZY.pkl', 'rb') as f:
            trainSet = pickle.load(f)
    else:
        with open('trainSetGRIZY.pkl', 'rb') as f:
            trainSet = pickle.load(f)
    clf = SVC()
    if concatBands:
        idxSnr = trainSet.names.index(snrType)
        idxExt = trainSet.names.index(extType)
        idxSeeing = trainSet.names.index('seeing')
        trainSetBand = trainSet.genTrainSubset(cuts={idxExt:(None, 0.2)}, cols=[idxSnr, idxExt, idxSeeing])
        Xtrain, Y = trainSetBand.getAllSet()
        clf.fit(Xtrain, Y)
        with open('svcWSeeingConcat.pkl', 'wb') as f:
            pickle.dump(clf, f)
    else:
        for i, band in enumerate(['g', 'r', 'i', 'z', 'y']):
            idxSnr = trainSet.names.index(snrType + '_' + band)
            idxExt = trainSet.names.index(extType + '_' + band)
            idxSeeing = trainSet.names.index('seeing' + '_' + band)
            trainSetBand = trainSet.genTrainSubset(cuts={idxExt:(None, 0.2)}, cols=[idxSnr, idxExt, idxSeeing])
            Xtrain, Y = trainSetBand.getAllSet()
            clf.fit(Xtrain, Y)
            with open('svcWSeeing{0}.pkl'.format(band.upper()), 'wb') as f:
                pickle.dump(clf, f)

def trainSVCWSeeingWMag(magType='i', extType='ext', concatBands=False, bands=['g', 'r', 'i', 'z', 'y'], getScore=False,
                        withDGauss=False, withSnr=False, snrType='snrPsf'):
    with open('trainSetGRIZY.pkl', 'rb') as f:
        trainSet = pickle.load(f)
    clf = SVC()
    if concatBands:
        trainSetConcat = trainSet.getConcatSet(keep=['mag'])
        cols = []
        idxMag = trainSetConcat.names.index('mag' + '_' + magType)
        cols.append(idxMag)
        idxExt = trainSetConcat.names.index(extType)
        cols.append(idxExt)
        if withSnr:
            idxSnr = trainSetConcat.names.index(snrType)
            cols.append(idxSnr)
        if withDGauss:
            idxRadInner = trainSetConcat.names.index('dGaussRadInner')
            cols.append(idxRadInner)
            idxAmpRat = trainSetConcat.names.index('dGaussAmpRat')
            cols.append(idxAmpRat)
        else:
            idxSeeing = trainSetConcat.names.index('seeing')
            cols.append(idxSeeing)
        trainSetConcatSub = trainSet.genTrainSubset(cuts={idxExt:(None, 0.2)}, cols=cols)
        if getScore:
            Xtrain, Y = trainSetConcatSub.getTrainSet()
            clf.fit(Xtrain, Y)
            Xtest, Y = trainSetConcatSub.getTestSet()
            weightStar = 1.0/np.sum(Y)
            weightGal = 1.0/(len(Y)-np.sum(Y))
            weightStar /= weightStar + weightGal
            weightGal /= weightStar + weightGal
            weights = np.zeros(Y.shape)
            weights[Y] = weightStar; weights[np.logical_not(Y)] = weightGal
            print "Concatenation score=", clf.score(Xtest, Y, sample_weight=weights)
        else:
            Xtrain, Y = trainSetConcatSub.getAllSet()
            clf.fit(Xtrain, Y)
        with open('svcWSeeingWMagConcat.pkl', 'wb') as f:
            pickle.dump(clf, f)
    else:
        for i, band in enumerate(['g', 'r', 'i', 'z', 'y']):
            cols = []
            idxMag = trainSet.names.index('mag' + '_' + magType)
            cols.append(idxMag)
            idxExt = trainSet.names.index(extType + '_' + band)
            cols.append(idxExt)
            if withSnr:
                idxSnr = trainSet.names.index(snrType + '_' + band)
                cols.append(idxSnr)
            if withDGauss:
                idxRadInner = trainSet.names.index('dGaussRadInner' + '_' + band)
                cols.append(idxRadInner)
                idxAmpRat = trainSet.names.index('dGaussAmpRat' + '_' + band)
                cols.append(idxAmpRat)
            else:
                idxSeeing = trainSet.names.index('seeing' + '_' + band)
                cols.append(idxSeeing)
            trainSetBand = trainSet.genTrainSubset(cuts={idxExt:(None, 0.2)}, cols=cols)
            if getScore:
                Xtrain, Y = trainSetBand.getTrainSet()
                clf.fit(Xtrain, Y)
                Xtest, Y = trainSetBand.getTestSet()
                weightStar = 1.0/np.sum(Y)
                weightGal = 1.0/(len(Y)-np.sum(Y))
                weightStar /= weightStar + weightGal
                weightGal /= weightStar + weightGal
                weights = np.zeros(Y.shape)
                weights[Y] = weightStar; weights[np.logical_not(Y)] = weightGal
                print "Band {0} score=".format(band), clf.score(Xtest, Y, sample_weight=weights)
            else:
                Xtrain, Y = trainSetBand.getAllSet()
                clf.fit(Xtrain, Y)
            with open('svcWSeeingWMag{0}.pkl'.format(band.upper()), 'wb') as f:
                pickle.dump(clf, f)

def trainSVCWDGauss(snrType='snrPsf', extType='ext'):
    with open('trainSetGRIZY.pkl', 'rb') as f:
        trainSet = pickle.load(f)
    clf = SVC()
    for i, band in enumerate(['g', 'r', 'i', 'z', 'y']):
        idxSnr = trainSet.names.index(snrType + '_' + band)
        idxExt = trainSet.names.index(extType + '_' + band)
        idxRadIn = trainSet.names.index('dGaussRadInner' + '_' + band)
        idxAmpRat = trainSet.names.index('dGaussAmpRat' + '_' + band)
        trainSetBand = trainSet.genTrainSubset(cuts={idxExt:(None, 0.2)}, cols=[idxSnr, idxExt, idxRadIn, idxAmpRat])
        Xtrain, Y = trainSetBand.getAllSet()
        clf.fit(Xtrain, Y)
        with open('svcWDGauss{0}.pkl'.format(band.upper()), 'wb') as f:
            pickle.dump(clf, f)

def plotDecFuncCMap(snrType='snrPsf', extType='ext', xRange=(5.0, 1000.0), yRange=(-0.1, 0.5),
                    xN=150, yN=150, ylim=None, fontSize=14, xlabel=None, ylabel=None, wSeeing=False,
                    seeingVal=None, fName='decFuncCMap.png', asLogX=True, concatBands=False):

    if wSeeing and seeingVal is None:
        raise ValueError("You need to specify a seeing value if wSeeing is set to `True`")
        
    with open('trainSetGRIZY.pkl', 'rb') as f:
        trainSet = pickle.load(f)

    if asLogX:
        xx, yy = np.meshgrid(np.linspace(np.log10(xRange[0]), np.log10(xRange[1]), num=xN),
                             np.linspace(yRange[0], yRange[1], num=yN))
        xx = np.power(10.0, xx)
    else:
        xx, yy = np.meshgrid(np.linspace(xRange[0], xRange[1], num=xN),
                             np.linspace(yRange[0], yRange[1], num=yN))
    if wSeeing:
        seeings = np.ones(xx.shape)*seeingVal/0.16/2.35
        Xphys = np.vstack((xx.flatten(), yy.flatten(), seeings.flatten())).T
    else:
        Xphys = np.vstack((xx.flatten(), yy.flatten())).T

    if concatBands:
        nColumn = 1; nRow = 1
    else:
        nColumn = 3; nRow = 2

    fig = plt.figure(figsize=(nColumn*8, nRow*6), dpi=120)
    if concatBands:
        if snrType == 'magI':
            idxSnr = trainSet.names.index('mag_i')
        else:
            idxSnr = trainSet.names.index(snrType)
        idxExt = trainSet.names.index(extType)
        if wSeeing:
            idxSeeing = trainSet.names.index('seeing')
            trainSetBand = trainSet.genTrainSubset(cuts={idxExt:(None, 0.2)}, cols=[idxSnr, idxExt, idxSeeing])
        else:
            trainSetBand = trainSet.genTrainSubset(cuts={idxExt:(None, 0.2)}, cols=[idxSnr, idxExt])
        X = trainSetBand.applyPostTestTransform(Xphys)
        if wSeeing and snrType == 'snrPsf':
            with open('svcWSeeingConcat.pkl', 'rb') as f:
                clf = pickle.load(f)
        elif wSeeing and snrType == 'magI':
            with open('svcWSeeingWMagConcat.pkl', 'rb') as f:
                clf = pickle.load(f)
        elif not wSeeing and snrType == 'magI':
            with open('svcWMagConcat.pkl', 'rb') as f:
                clf = pickle.load(f)
        else:
            with open('svcConcat.pkl', 'rb') as f:
                clf = pickle.load(f)
        Z = clf.decision_function(X)
        cap = np.logical_and(True, Z > 1.0)
        Z[cap] = 1.0
        cap = np.logical_and(True, Z < -1.0)
        Z[cap] = -1.0
        zz = Z.reshape(xx.shape)
        ax = fig.add_subplot(nRow, nColumn, i+1)
        mappable = ax.pcolor(xx, yy, zz)
        if snrType == 'snrPsf':
            ax.set_xscale('log')
        ax.set_xlim(xRange)
        ax.set_ylim(yRange)
        cb = plt.colorbar(mappable, ax=ax)
    else:
        for i, band in enumerate(['g', 'r', 'i', 'z', 'y']):
            if snrType == 'magI':
                idxSnr = trainSet.names.index('mag_i')
            else:
                idxSnr = trainSet.names.index(snrType + '_' + band)
            idxExt = trainSet.names.index(extType + '_' + band)
            if wSeeing:
                idxSeeing = trainSet.names.index('seeing' + '_' + band)
                trainSetBand = trainSet.genTrainSubset(cuts={idxExt:(None, 0.2)}, cols=[idxSnr, idxExt, idxSeeing])
            else:
                trainSetBand = trainSet.genTrainSubset(cuts={idxExt:(None, 0.2)}, cols=[idxSnr, idxExt])
            X = trainSetBand.applyPostTestTransform(Xphys)
            if wSeeing and snrType == 'snrPsf':
                with open('svcWSeeing{0}.pkl'.format(band.upper()), 'rb') as f:
                    clf = pickle.load(f)
            elif wSeeing and snrType == 'magI':
                with open('svcWSeeingWMag{0}.pkl'.format(band.upper()), 'rb') as f:
                    clf = pickle.load(f)
            elif not wSeeing and snrType == 'magI':
                with open('svcWMag{0}.pkl'.format(band.upper()), 'rb') as f:
                    clf = pickle.load(f)
            else:
                with open('svc{0}.pkl'.format(band.upper()), 'rb') as f:
                    clf = pickle.load(f)
            Z = clf.decision_function(X)
            cap = np.logical_and(True, Z > 1.0)
            Z[cap] = 1.0
            cap = np.logical_and(True, Z < -1.0)
            Z[cap] = -1.0
            zz = Z.reshape(xx.shape)
            ax = fig.add_subplot(nRow, nColumn, i+1)
            mappable = ax.pcolor(xx, yy, zz)
            if snrType == 'snrPsf':
                ax.set_xscale('log')
            ax.set_xlim(xRange)
            ax.set_ylim(yRange)
            cb = plt.colorbar(mappable, ax=ax)
    fig.savefig('psfPaperFigs/{0}'.format(fName), dpi=120, bbox_inches='tight')
    return fig

def seeingDistribs(dType='seeing', fontSize=16, xlabel='FWHM "', ylabel='PDF', histRange=(0.5, 0.9),
                   xlim=(0.5, 0.9), ylim=(0.0, 50.0), nTicks=5, fName='seeingDistrib.png', textRight=['r', 'i', 'z'],
                   textLeft=['g', 'y']):
    with open('trainSetGRIZY.pkl', 'rb') as f:
        trainSet = pickle.load(f)
    nRow = 5; nColumn = 1
    fig = plt.figure(figsize=(nColumn*8, nRow*3), dpi=120)
    for i, band in enumerate(['g', 'r', 'i', 'z', 'y']):
        idxSeeing = trainSet.names.index(dType + '_' + band)
        ax = fig.add_subplot(nRow, nColumn, i+1)
        if dType == 'seeing':
            ax.hist(trainSet.X[:, idxSeeing]*0.16*2.35, bins=75, range=histRange, color='black', histtype='step', normed=True)
        else:
            ax.hist(trainSet.X[:, idxSeeing], bins=75, range=histRange, color='black', histtype='step', normed=True)
        ax.set_xlabel(xlabel, fontsize=fontSize)
        ax.set_ylabel(ylabel, fontsize=fontSize)
        if band in textRight:
            ax.text(0.75, 0.5, 'HSC-{0}'.format(band.upper()), transform=ax.transAxes, fontsize=fontSize+2,
                    verticalalignment='top')
        elif band in textLeft:
            ax.text(0.25, 0.5, 'HSC-{0}'.format(band.upper()), transform=ax.transAxes, fontsize=fontSize+2,
                    verticalalignment='top')
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontSize)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontSize)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        if i > 0:
            plt.yticks(np.arange(ylim[0] , ylim[1], (ylim[1]-ylim[0])*1.0/nTicks))
        #    ax.yaxis.set_major_locator(MaxNLocator(prune='upper'))
        #ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useOffset=False)
    fig.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
    fig.savefig('psfPaperFigs/{0}'.format(fName), dpi=120, bbox_inches='tight')
    return fig

def equalShapeDifferentBands(snrType='snrPsf', extType='ext', ylim=None, underSample=None, size=1, fontSize=14,
                             xlabel=None, ylabel=None, doTrain=False, asLogX=True):
    with open('trainSetGRIZY.pkl', 'rb') as f:
        trainSet = pickle.load(f)
    nColumn = 3; nRow = 2
    if underSample is not None:
        nSample = int(underSample*trainSet.X.shape[0])
        idxSample = np.random.choice(trainSet.X.shape[0], nSample, replace=False)
    else:
        idxSample = range(trainSet.X.shape[0])
    #estimator = LinearSVC()
    #param_grid = {'C':[0.1, 1.0, 10.0, 100.0]}
    #clf = GridSearchCV(estimator, param_grid, n_jobs=4)
    #clf = LinearSVC()
    if doTrain:
        clf = SVC()
    fig = plt.figure(figsize=(nColumn*8, nRow*6), dpi=120)
    figBdy = plt.figure(dpi=120)
    axBdy = figBdy.add_subplot(1, 1, 1)
    for i, band in enumerate(['g', 'r', 'i', 'z', 'y']):
        if snrType == 'magI':
            idxSnr = trainSet.names.index('mag_i')
        else:
            idxSnr = trainSet.names.index(snrType + '_' + band)
        idxExt = trainSet.names.index(extType + '_' + band)
        trainSetBand = trainSet.genTrainSubset(cuts={idxExt:(None, 0.2)}, cols=[idxSnr, idxExt])
        Xtrain, Y = trainSetBand.getAllSet()
        if doTrain:
            clf.fit(Xtrain, Y)
        else:
            if snrType == 'snrPsf' and extType == 'ext':
                with open('svc{0}.pkl'.format(band.upper()), 'rb') as f:
                    clf = pickle.load(f)
            elif snrType == 'magI' and extType == 'ext':
                with open('svcWMag{0}.pkl'.format(band.upper()), 'rb') as f:
                    clf = pickle.load(f)
        train = etl.Training(trainSetBand, clf, preFit=False)
        print "band {0}".format(band)
        train.printPhysicalFit(normalized=True)
        ax = fig.add_subplot(nRow, nColumn, i+1)
        snr = trainSet.X[:, idxSnr]
        ext = trainSet.X[:, idxExt]
        for i in idxSample: 
            if trainSet.Y[i]:
                ax.plot(snr[i], ext[i], marker='.', markersize=size, color='blue')
            else:
                ax.plot(snr[i], ext[i], marker='.', markersize=size, color='red')
        if snrType == 'snrPsf' and extType == 'ext':
            if band == 'g':
                xRange=(50.0, 1000.0)
                yRange=(-0.05, 0.175)
                fallbackRange = (0.0, 0.175)
            elif band == 'r':
                xRange=(38.2, 1000.0)
                yRange=(-0.05, 0.175)
                fallbackRange = (0.0, 0.175)
            elif band == 'i':
                xRange=(36.0, 1000.0)
                yRange=(-0.05, 0.175)
                fallbackRange = (0.0, 0.175)
            elif band == 'z':
                xRange=(30.0, 1000.0)
                yRange=(-0.04, 0.175)
                fallbackRange = (0.0, 0.175)
            elif band == 'y':
                xRange=(30.0, 1000.0)
                yRange=(-0.05, 0.175)
                fallbackRange = (0.0, 0.175)
        elif snrType == 'magI' and extType == 'ext':
            if band == 'g':
                xRange=(18.0, 23.75)
                yRange=(-0.01, 0.3)
                fallbackRange = (-0.05, 0.175)
            elif band == 'r':
                xRange=(18, 24.75)
                yRange=(-0.01, 0.3)
                fallbackRange = (-0.05, 0.175)
            elif band == 'i':
                xRange=(18.0, 25.2)
                yRange=(-0.01, 0.3)
                fallbackRange = (-0.05, 0.175)
            elif band == 'z':
                xRange=(18.0, 24.75)
                yRange=(-0.01, 0.3)
                fallbackRange = (-0.05, 0.175)
            elif band == 'y':
                xRange=(18.0, 24.0)
                yRange=(-0.01, 0.3)
                fallbackRange = (-0.05, 0.175)
        xGrid, yGrid = train.getDecBoundary(0, 1, xRange=xRange, nPoints=100, yRange=yRange,
                                            asLogX=asLogX, fallbackRange=fallbackRange)
        axBdy.plot(xGrid, yGrid, label='HSC-{0}'.format(band.upper()))
        ax.plot(xGrid, yGrid, color='black', linestyle='--')
        if snrType == 'snrPsf':
            ax.set_xlim((5.0, 1000.0))
        elif snrType == 'magI':
            ax.set_xlim((19.0, 27.0))
        if ylim is not None:
            ax.set_ylim(ylim)
        if snrType == 'snrPsf':
            ax.set_xscale('log')
        if xlabel is not None:
            ax.set_xlabel(xlabel, fontsize=fontSize)
            axBdy.set_xlabel(xlabel, fontsize=fontSize)
        if ylabel is not None:
            ax.set_ylabel(ylabel, fontsize=fontSize)
            axBdy.set_ylabel(ylabel, fontsize=fontSize)
        ax.text(0.75, 0.17, 'HSC-{0}'.format(band.upper()), transform=ax.transAxes, fontsize=fontSize+2,
                verticalalignment='top')
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontSize)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontSize)
        for tick in axBdy.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontSize)
        for tick in axBdy.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontSize)
    axBdy.set_ylim(ylim)
    if snrType == 'snrPsf':
        axBdy.set_xlim((5.0, 1000.0))
        axBdy.legend(loc='upper left')
        axBdy.set_xscale('log')
        fig.savefig('psfPaperFigs/equalShapeDiffBand{0}.png'.format(extType), dpi=120, bbox_inches='tight')
        figBdy.savefig('psfPaperFigs/equalShapeDiffBandBdy{0}.png'.format(extType), dpi=120, bbox_inches='tight')
    elif snrType == 'magI':
        axBdy.set_xlim((18.0, 26.0))
        axBdy.legend(loc='upper right')
        fig.savefig('psfPaperFigs/equalShapeDiffBandWMag{0}.png'.format(extType), dpi=120, bbox_inches='tight')
        figBdy.savefig('psfPaperFigs/equalShapeDiffBandBdyWMag{0}.png'.format(extType), dpi=120, bbox_inches='tight')
    return fig

def equalShapeDifferentBandsWSeeing(snrType='snrPsf', extType='ext', ylim=None, underSample=None, size=1, fontSize=14,
                                    xlabel=None, ylabel=None, doTrain=False, asLogX=False, concatBands=False):
    with open('trainSetGRIZY.pkl', 'rb') as f:
        trainSet = pickle.load(f)
    nColumn = 3; nRow = 2
    if underSample is not None:
        nSample = int(underSample*trainSet.X.shape[0])
        idxSample = np.random.choice(trainSet.X.shape[0], nSample, replace=False)
    else:
        idxSample = range(trainSet.X.shape[0])
    if doTrain:
        clf = SVC()
    figBdy = plt.figure(dpi=120)
    axBdy = figBdy.add_subplot(1, 1, 1)
    for i, band in enumerate(['g', 'r', 'i', 'z', 'y']):
        if snrType == 'magI':
            idxSnr = trainSet.names.index('mag_i')
        else:
            idxSnr = trainSet.names.index(snrType + '_' + band)
        idxExt = trainSet.names.index(extType + '_' + band)
        idxSeeing = trainSet.names.index('seeing' + '_' + band)
        trainSetBand = trainSet.genTrainSubset(cuts={idxExt:(None, 0.2)}, cols=[idxSnr, idxExt, idxSeeing])
        Xtrain, Y = trainSetBand.getAllSet()
        if doTrain:
            clf.fit(Xtrain, Y)
        else:
            if snrType == 'snrPsf' and extType == 'ext' and not concatBands:
                with open('svcWSeeing{0}.pkl'.format(band.upper()), 'rb') as f:
                    clf = pickle.load(f)
            elif snrType == 'magI' and extType == 'ext' and not concatBands:
                with open('svcWSeeingWMag{0}.pkl'.format(band.upper()), 'rb') as f:
                    clf = pickle.load(f)
            elif snrType == 'magI' and extType == 'ext' and concatBands:
                with open('svcWSeeingWMagConcat.pkl', 'rb') as f:
                    clf = pickle.load(f)
        train = etl.Training(trainSetBand, clf, preFit=False)
        if snrType == 'snrPsf' and extType == 'ext':
            if band == 'g':
                xRange=(50.0, 1000.0)
                yRange=(-0.06, 0.175)
                fallbackRange = (0.0, 0.175)
                seeingVal = 0.78/0.16/2.35
            elif band == 'r':
                xRange=(38.2, 1000.0)
                yRange=(-0.05, 0.175)
                fallbackRange = (0.0, 0.175)
                seeingVal = 0.55/0.16/2.35
            elif band == 'i':
                xRange=(36.0, 1000.0)
                yRange=(-0.05, 0.175)
                fallbackRange = (0.0, 0.175)
                seeingVal = 0.61/0.16/2.35
            elif band == 'z':
                xRange=(31.0, 1000.0)
                yRange=(-0.04, 0.175)
                fallbackRange = (0.0, 0.175)
                seeingVal = 0.55/0.16/2.35
            elif band == 'y':
                xRange=(30.0, 1000.0)
                yRange=(-0.05, 0.175)
                fallbackRange = (0.0, 0.175)
                seeingVal = 0.78/0.16/2.35
        elif snrType == 'magI' and extType == 'ext':
            if band == 'g':
                xRange=(19.0, 23.75)
                yRange=(-0.02, 0.3)
                fallbackRange = (-0.05, 0.175)
                seeingVal = 0.78/0.16/2.35
            elif band == 'r':
                xRange=(19, 24.75)
                yRange=(-0.025, 0.3)
                fallbackRange = (-0.05, 0.175)
                seeingVal = 0.55/0.16/2.35
            elif band == 'i':
                xRange=(19.0, 25.2)
                yRange=(-0.085, 0.3)
                fallbackRange = (-0.08, 0.175)
                seeingVal = 0.6/0.16/2.35
            elif band == 'z':
                xRange=(19.0, 24.75)
                yRange=(-0.01, 0.3)
                fallbackRange = (-0.05, 0.175)
                seeingVal = 0.55/0.16/2.35
            elif band == 'y':
                xRange=(19.0, 24.0)
                yRange=(-0.03, 0.3)
                fallbackRange = (-0.05, 0.175)
                seeingVal = 0.78/0.16/2.35

        xGrid, yGrid = train.getDecBoundary(0, 1, xRange=xRange, nPoints=100, yRange=yRange,
                                            asLogX=asLogX, fallbackRange=(0.0, 0.175), fixedIndexes=[2], fixedVals=[seeingVal])
        axBdy.plot(xGrid, yGrid, label='HSC-{0} (FWHM={1}")'.format(band.upper(), seeingVal*0.16*2.35))
        if xlabel is not None:
            axBdy.set_xlabel(xlabel, fontsize=fontSize)
        if ylabel is not None:
            axBdy.set_ylabel(ylabel, fontsize=fontSize)

        for tick in axBdy.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontSize)
        for tick in axBdy.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontSize)

    axBdy.set_ylim(ylim)
    if snrType == 'snrPsf':
        axBdy.set_xlim((5.0, 1000.0))
        axBdy.legend(loc='upper left')
        axBdy.set_xscale('log')
        figBdy.savefig('psfPaperFigs/equalShapeDiffBandBdyWSeeing{0}.png'.format(extType), dpi=120, bbox_inches='tight')
    elif snrType == 'magI':
        axBdy.set_xlim((19.0, 27.0))
        axBdy.legend(loc='upper right')
        figBdy.savefig('psfPaperFigs/equalShapeDiffBandBdyWSeeingWMag{0}.png'.format(extType), dpi=120, bbox_inches='tight')

    return figBdy

if __name__ == '__main__':
    #equalShapeDifferentBands(underSample=0.05, extType='ext', ylim=(-0.075, 0.2), xlabel='S/N', ylabel=r'$mag_{psf}-mag_{cmodel}$')
    #equalShapeDifferentBandsWSeeing(extType='ext', ylim=(-0.075, 0.2), xlabel='S/N', ylabel=r'$mag_{psf}-mag_{cmodel}$')
    #equalShapeDifferentBandsWSeeing(snrType='magI', extType='ext', ylim=(-0.075, 0.2), xlabel=r'$mag_{cmodel}$ HSC-I', ylabel=r'$mag_{psf}-mag_{cmodel}$')
    #equalShapeDifferentBandsWSeeing(snrType='magI', extType='ext', ylim=(-0.075, 0.2), xlabel=r'$mag_{cmodel}$ HSC-I',
    #                                ylabel=r'$mag_{psf}-mag_{cmodel}$', concatBands=True)
    #trainSVC()
    #trainSVCWMag()
    #equalShapeDifferentBands(underSample=0.05, snrType='magI', extType='ext', ylim=(-0.075, 0.2), xlabel=r'$mag_{cmodel}$ HSC-I', ylabel=r'$mag_{psf}-mag_{cmodel}$', asLogX=False)
    #trainSVCWSeeing()
    #trainSVCWSeeingWMag()
    trainSVCWSeeingWMag(concatBands=False, getScore=True, withDGauss=False, withSnr=True, snrType='snrAp')
    trainSVCWSeeingWMag(concatBands=True, getScore=True, withDGauss=False, withSnr=True, snrType='snrAp')
    #trainSVCWDGauss()
    #plotDecFuncCMap()
    #plotDecFuncCMap(snrType='magI', xRange=(18.0, 26.0), asLogX=False)
    #plotDecFuncCMap(wSeeing=True, seeingVal=0.55)
    #plotDecFuncCMap(snrType='magI', xRange=(19.0, 27.0), wSeeing=True, seeingVal=0.78, asLogX=False)
    #seeingDistribs()
    #plotDecFuncCMap(wSeeing=True, seeingVal=0.775)
    #seeingDistribs(dType='dGaussRadInner', histRange=(1.0, 2.5), xlim=(1.0, 2.5), ylim=(0.0, 15.0), fName='dGaussRadInnerDistrib.png', xlabel=r'$\sigma_{in}$')
    #seeingDistribs(dType='dGaussRadRat', histRange=(1.95, 2.05), xlim=(1.95, 2.05), ylim=(0.0, 15.0), fName='dGaussRadRatDistrib.png', xlabel=r'$\sigma_{out}/\sigma_{in}$')
    #seeingDistribs(dType='dGaussAmpRat', histRange=(0.15, 0.35), xlim=(0.15, 0.35), ylim=(0.0, 65.0), fName='dGaussAmpRatDistrib.png', xlabel=r'$peak_{out}/peak_{in}$', textRight=['g', 'r', 'i', 'z'], textLeft=['y'])
    #plt.show()
