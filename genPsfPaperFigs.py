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

def trainSVCWSeeing(snrType='snrPsf', extType='ext'):
    with open('trainSetGRIZY.pkl', 'rb') as f:
        trainSet = pickle.load(f)
    clf = SVC()
    for i, band in enumerate(['g', 'r', 'i', 'z', 'y']):
        idxSnr = trainSet.names.index(snrType + '_' + band)
        idxExt = trainSet.names.index(extType + '_' + band)
        idxSeeing = trainSet.names.index('seeing' + '_' + band)
        trainSetBand = trainSet.genTrainSubset(cuts={idxExt:(None, 0.2)}, cols=[idxSnr, idxExt, idxSeeing])
        Xtrain, Y = trainSetBand.getAllSet()
        clf.fit(Xtrain, Y)
        with open('svcWSeeing{0}.pkl'.format(band.upper()), 'wb') as f:
            pickle.dump(clf, f)

def plotDecFuncCMap(snrType='snrPsf', extType='ext', xRange=(5.0, 1000.0), yRange=(-0.1, 0.5),
                    xN=150, yN=150, ylim=None, fontSize=14, xlabel=None, ylabel=None, wSeeing=False,
                    seeingVal=None):

    if wSeeing and seeingVal is None:
        raise ValueError("You need to specify a seeing value if wSeeing is set to `True`")
        
    with open('trainSetGRIZY.pkl', 'rb') as f:
        trainSet = pickle.load(f)

    xx, yy = np.meshgrid(np.linspace(np.log10(xRange[0]), np.log10(xRange[1]), num=xN),
                         np.linspace(yRange[0], yRange[1], num=yN))
    xx = np.power(10.0, xx)
    if wSeeing:
        seeings = np.ones(xx.shape)*seeingVal/0.16/2.35
        Xphys = np.vstack((xx.flatten(), yy.flatten(), seeings.flatten())).T
    else:
        Xphys = np.vstack((xx.flatten(), yy.flatten())).T

    nColumn = 3; nRow = 2
    fig = plt.figure(figsize=(nColumn*8, nRow*6), dpi=120)
    for i, band in enumerate(['g', 'r', 'i', 'z', 'y']):
        idxSnr = trainSet.names.index(snrType + '_' + band)
        idxExt = trainSet.names.index(extType + '_' + band)
        if wSeeing:
            idxSeeing = trainSet.names.index('seeing' + '_' + band)
            trainSetBand = trainSet.genTrainSubset(cuts={idxExt:(None, 0.2)}, cols=[idxSnr, idxExt, idxSeeing])
        else:
            trainSetBand = trainSet.genTrainSubset(cuts={idxExt:(None, 0.2)}, cols=[idxSnr, idxExt])
        X = trainSetBand.applyPostTestTransform(Xphys)
        if wSeeing:
            with open('svcWSeeing{0}.pkl'.format(band.upper()), 'rb') as f:
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
        ax.set_xscale('log')
        ax.set_xlim(xRange)
        ax.set_ylim(yRange)
        cb = plt.colorbar(mappable, ax=ax)
    return fig

def seeingDistribs(fontSize=16):
    with open('trainSetGRIZY.pkl', 'rb') as f:
        trainSet = pickle.load(f)
    nRow = 5; nColumn = 1
    fig = plt.figure(figsize=(nColumn*8, nRow*3), dpi=120)
    for i, band in enumerate(['g', 'r', 'i', 'z', 'y']):
        idxSeeing = trainSet.names.index('seeing' + '_' + band)
        ax = fig.add_subplot(nRow, nColumn, i+1)
        ax.hist(trainSet.X[:, idxSeeing]*0.16*2.35, bins=75, range=(0.5, 0.9), color='black', histtype='step', normed=True)
        ax.set_xlabel('FWHM "', fontsize=fontSize)
        ax.set_ylabel('PDF', fontsize=fontSize)
        if band in ['r', 'i', 'z']:
            ax.text(0.75, 0.5, 'HSC-{0}'.format(band.upper()), transform=ax.transAxes, fontsize=fontSize+2,
                    verticalalignment='top')
        elif band in ['g', 'y']:
            ax.text(0.25, 0.5, 'HSC-{0}'.format(band.upper()), transform=ax.transAxes, fontsize=fontSize+2,
                    verticalalignment='top')
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontSize)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontSize)
        ax.set_xlim((0.5, 0.9))
        ax.set_ylim((0.0, 50.0))
        if i > 0:
            plt.yticks(np.arange(0 ,50, 10))
        #    ax.yaxis.set_major_locator(MaxNLocator(prune='upper'))
        #ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useOffset=False)
    fig.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
    fig.savefig('psfPaperFigs/seeingDistrib.png', dpi=120, bbox_inches='tight')
    return fig

def equalShapeDifferentBands(snrType='snrPsf', extType='ext', ylim=None, underSample=None, size=1, fontSize=14,
                             xlabel=None, ylabel=None, doTrain=False):
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
        idxSnr = trainSet.names.index(snrType + '_' + band)
        idxExt = trainSet.names.index(extType + '_' + band)
        trainSetBand = trainSet.genTrainSubset(cuts={idxExt:(None, 0.2)}, cols=[idxSnr, idxExt])
        Xtrain, Y = trainSetBand.getAllSet()
        if doTrain:
            clf.fit(Xtrain, Y)
        else:
            with open('svc{0}.pkl'.format(band.upper()), 'rb') as f:
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
        if band == 'g':
            xRange=(45.0, 1000.0)
            yRange=(-0.05, 0.175)
        elif band == 'r':
            xRange=(30.0, 1000.0)
            yRange=(-0.05, 0.175)
        elif band == 'i':
            xRange=(33.0, 1000.0)
            yRange=(-0.05, 0.175)
        elif band == 'z':
            xRange=(30.0, 1000.0)
            yRange=(-0.04, 0.175)
        elif band == 'y':
            xRange=(25.0, 1000.0)
            yRange=(-0.05, 0.175)
        xGrid, yGrid = train.getDecBoundary(0, 1, xRange=xRange, nPoints=100, yRange=yRange,
                                            asLogX=True, fallbackRange=(0.0, 0.175))
        axBdy.plot(xGrid, yGrid)
        ax.plot(xGrid, yGrid, color='black', linestyle='--')
        ax.set_xlim((5.0, 1000.0))
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.set_xscale('log')
        if xlabel is not None:
            ax.set_xlabel(xlabel, fontsize=fontSize)
        if ylabel is not None:
            ax.set_ylabel(ylabel, fontsize=fontSize)
        ax.text(0.75, 0.17, 'HSC-{0}'.format(band.upper()), transform=ax.transAxes, fontsize=fontSize+2,
                verticalalignment='top')
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontSize)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontSize)

    axBdy.set_xlim((5.0, 1000.0))
    axBdy.set_ylim(ylim)
    axBdy.set_xscale('log')
    fig.savefig('psfPaperFigs/equalShapeDiffBand{0}.png'.format(extType), dpi=120, bbox_inches='tight')

    return fig

if __name__ == '__main__':
    #equalShapeDifferentBands(underSample=0.05, extType='ext', ylim=(-0.075, 0.2), xlabel='S/N', ylabel=r'$mag_{psf}-mag_{cmodel}$')
    #trainSVC()
    #trainSVCWSeeing()
    #plotDecFuncCMap(wSeeing=True, seeingVal=0.575)
    #plotDecFuncCMap(wSeeing=True, seeingVal=0.8)
    seeingDistribs()
    plt.show()
