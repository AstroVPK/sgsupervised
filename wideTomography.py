import os
import csv

import numpy as np
import matplotlib.pyplot as plt

_fields = ['XMM', 'GAMA09', 'WIDE12H', 'GAMA15', 'HectoMap', 'VVDS', 'AEGIS']
_bands = ['g', 'r', 'i', 'z', 'y']

def fileLen(fName):
    with open(fName) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def selectFieldSubset(fName, n):
    nField = fileLen(fName) - 1
    subset = np.sort(np.random.choice(nField, size=n, replace=False))
    return subset

def getXFromLine(line, cList):
    X = np.zeros((1, 5))
    magI = np.array([float(line[cList.index('imag')])])
    X[0][0] = float(line[cList.index('gmag')]) - float(line[cList.index('rmag')])
    X[0][1] = float(line[cList.index('rmag')]) - float(line[cList.index('imag')])
    X[0][2] = float(line[cList.index('imag')]) - float(line[cList.index('zmag')])
    X[0][3] = float(line[cList.index('zmag')]) - float(line[cList.index('ymag')])
    errG = float(line[cList.index('gmag_cmodel_err')])
    errR = float(line[cList.index('rmag_cmodel_err')])
    errI = float(line[cList.index('imag_cmodel_err')])
    errZ = float(line[cList.index('zmag_cmodel_err')])
    errY = float(line[cList.index('ymag_cmodel_err')])
    bandBest = _bands[np.array([errG, errR, errI, errZ, errY]).argmin()]
    ext = float(line[cList.index('{0}ext'.format(bandBest))])
    X[0][4] = ext
    errPsf = float(line[cList.index('{0}mag_psf_err'.format(bandBest))])
    errBest = float(line[cList.index('{0}mag_cmodel_err'.format(bandBest))])
    XErr = np.zeros((1, 5, 5))
    XErr[0][0, 0] = errG**2 + errR**2
    XErr[0][0, 1] = -errR**2
    XErr[0][1, 0] = -errR**2
    XErr[0][1, 1] = errR**2 + errI**2
    XErr[0][1, 2] = -errI**2
    XErr[0][2, 1] = -errI**2
    XErr[0][2, 2] = errI**2 + errZ**2
    XErr[0][2, 3] = -errZ**2
    XErr[0][3, 2] = -errZ**2
    XErr[0][3, 3] = errZ**2 + errY**2
    XErr[0][4, 4] = errBest**2
    return X, XErr, magI

def computeFieldPosteriors(field):
    if not field in _fields:
        raise ValueError("Field must be one of {0}".format(_fields))
    magBins = [(18.0, 22.0), (22.0, 24.0), (24.0, 25.0), (25.0, 26.0)]
    with open('clfsColsExt.pkl', 'rb') as f:
        clfs = pickle.load(f)
    clfXd = dGauss.XDClfs(clfs=clfs, magBins=magBins)

    fileInput = '/scr/depot0/garmilla/HSC/wide{0}.csv'.format(field)
    fileOutput = '/scr/depot0/garmilla/HSC/wide{0}Posteriors.csv'.format(field)

    with open(fileInput, 'r') as fInput:
        reader = csv.reader(fInput, delimiter=',')
        cList = reader.next() # Columns
        cList[0] = cList[0][2:] # Remove number sign and space
        with open(fileOutput, 'w') as fOutput:
            fOutput.write('# P(Star)\n')
            for line in reader:
                try:
                    X, XErr, magI = getXFromLine(line, cList)
                except ValueError:
                    fOutput.write('nan\n')
                    continue
                pStar = clfXd.predict_proba(X, XErr, magI)[0]
                fOutput.write('{0}\n'.format(pStar))

def loadFieldData(field, subsetSize=None):
    fNameData = '/scr/depot0/garmilla/HSC/wide{0}.csv'.format(field)
    fNamePost = '/scr/depot0/garmilla/HSC/wide{0}Posteriors.csv'.format(field)
    if subsetSize is None:
        subsetSize = fileLen(fNamePost) - 1
    subset = selectFieldSubset(fNamePost, subsetSize)
    with open(fNameData, 'r') as fData:
        with open(fNamePost, 'r') as fPost:
            readerData = csv.reader(fData, delimiter=',')
            readerPost = csv.reader(fPost, delimiter=',')
            cList = readerData.next() # Columns
            cList[0] = cList[0][2:] # Remove number sign and space
            readerPost.next() # Synchronyze readers
            XList = []; XErrList = []; magIList = []; YList = []
            for line in readerData:
                posterior = float(readerPost.next()[0])
                assert readerPost.line_num == readerData.line_num
                idx = readerData.line_num - 2
                if idx in subset:
                    try:
                        X, XErr, magI = getXFromLine(line, cList)
                        XList.append(X[0])
                        XErrList.append(XErr[0])
                        magIList.append(magI[0])
                        YList.append(posterior)
                    except ValueError:
                        continue
                else:
                    continue
    X = np.array(XList)
    XErr = np.array(XErrList)
    magI = np.array(magIList)
    Y = np.array(YList)
    return X, XErr, magI, Y

def makeCCDiagrams(field, threshold = 0.9, subsetSize=100000, fontSize=18):
    magBins = [(18.0, 22.0), (22.0, 24.0), (24.0, 25.0)]
    X, XErr, magI, Y = loadFieldData(field, subsetSize=subsetSize)
    magString = r'$\mathrm{Mag}_{cmodel}$ HSC-I'
    colNames = ['g-r', 'r-i', 'i-z', 'z-y']
    colLims = [(0.0, 1.5), (-0.2, 2.0), (-0.2, 1.0), (-0.2, 0.4)]
    fig = plt.figure(figsize=(24, 18), dpi=120)
    for i in range(3):
        good = np.logical_and(Y > threshold, np.logical_and(magI > magBins[i][0], magI < magBins[i][1]))
        for j in range(i*3+1, i*3+4):
            ax = fig.add_subplot(3, 3, j)
            ax.set_title('{0} < {1} < {2}'.format(magBins[i][0], magString, magBins[i][1]), fontsize=fontSize)
            ax.set_xlabel(colNames[j-i*3-1], fontsize=fontSize)
            ax.set_ylabel(colNames[j-i*3], fontsize=fontSize)
            ax.set_xlim(colLims[j-i*3-1])
            ax.set_ylim(colLims[j-i*3])
            im = ax.scatter(X[:, j-i*3-1][good], X[:, j-i*3][good], marker='.', s=10, c=Y[good], vmin=0.9, vmax=1.0,
                            edgecolors='none')
        bounds = ax.get_position().bounds
        cax = fig.add_axes([0.93, bounds[1], 0.015, bounds[3]])
        cb = plt.colorbar(im, cax=cax)
        cb.set_label(r'P(Star|Colors+Extendedness)', fontsize=fontSize)
        cb.ax.tick_params(labelsize=fontSize)
    for ax in fig.get_axes():
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontSize)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontSize)
    dirHome = os.path.expanduser('~')
    fig.savefig(os.path.join(dirHome, 'Desktop/wide{0}PstarG{1}.png'.format(field, threshold)), dpi=120, bbox_inches='tight')

if __name__ == '__main__':
    field = 'VVDS'
    computeFieldPosteriors(field)
