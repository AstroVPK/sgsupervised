import os
import csv
import pickle

import numpy as np
from numpy.linalg import inv
from scipy.stats import beta, poisson
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from astropy import units
from astropy.coordinates import SkyCoord

import dGauss

_fields = ['XMM', 'GAMA09', 'WIDE12H', 'GAMA15', 'HectoMap', 'VVDS', 'AEGIS']
#_fields = ['AEGIS']
_bands = ['g', 'r', 'i', 'z', 'y']
_colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']
#_colors = ['black']

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
    try:
        fNameData = '/scr/depot0/garmilla/HSC/wide{0}.csv'.format(field)
        fNamePost = '/scr/depot0/garmilla/HSC/wide{0}Posteriors.csv'.format(field)
        fileLen(fNamePost) - 1
    except IOError:
        fNameData = '/home/jose/Data/wide{0}.csv'.format(field)
        fNamePost = '/home/jose/Data/wide{0}Posteriors.csv'.format(field)
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
            idList = []; raList = []; decList = []
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
                        idList.append(int(line[cList.index('id')]))
                        raList.append(float(line[cList.index('ra2000')]))
                        decList.append(float(line[cList.index('decl2000')]))
                    except ValueError:
                        continue
                else:
                    continue
    ids = np.array(idList)
    ra = np.array(raList)
    dec = np.array(decList)
    X = np.array(XList)
    XErr = np.array(XErrList)
    magI = np.array(magIList)
    Y = np.array(YList)
    return ids, ra, dec, X, XErr, magI, Y

def preLoadField(field, subsetSize=None):
    ids, ra, dec, X, XErr, magI, Y = loadFieldData(field, subsetSize=subsetSize)
    with open('/scr/depot0/garmilla/HSC/data{0}.pkl'.format(field), 'w') as f:
        pickle.dump((ids, ra, dec, X, XErr, magI, Y), f)

def genDBPosts(field, subsetSize=None):
    ids, ra, dec, X, XErr, magI, Y = loadFieldData(field, subsetSize=subsetSize)
    if os.path.isdir('/scr/depot0/garmilla/HSC'):
        fName = '/scr/depot0/garmilla/HSC/{0}Posteriors.txt'.format(field)
    else:
        dirHome = os.path.expanduser('~')
        fName = os.path.join(dirHome, 'Desktop/{0}Posteriors.txt'.format(field))
    with open(fName, 'w') as f:
        f.write('# id, P(Star)\n')
        for i in range(len(ids)):
            f.write('{0}, {1}\n'.format(ids[i], Y[i]))

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

def _getAbsoluteMagR(riSdss):
    return 4.0 + 11.86*riSdss - 10.74*riSdss**2 + 5.99*riSdss**3 - 1.20*riSdss**4

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

def getJeffreysInterval(alpha, n, x):
    if not np.logical_and(0.0 < alpha, 1.0 > alpha):
        raise ValueError("alpha must be between 0 and 1.")
    if not isinstance(x, int) or not isinstance(n, int):
        raise ValueError("x and n must be integers.")
    if x > n:
        raise ValueError("x has to be less than or equal to n.")
    if x == 0:
        boundL = 0.0
    else:
        boundL = beta.ppf(alpha/2, x + 0.5, n - x + 0.5)
    if x == n:
        boundU = 1.0
    else:
        boundU = beta.ppf(1.0 - alpha/2, x + 0.5, n - x + 0.5)
    return boundL, boundU

def makeTomographyField(field, subsetSize=100000, threshold=0.9, fontSize=18):
    ids, ra, dec, X, XErr, magI, Y = loadFieldData(field, subsetSize=subsetSize)
    good = np.logical_and(Y >= threshold, X[:,1] < 0.4)
    good = np.logical_and(good, X[:,2] < 0.2)
    good = np.logical_and(good, magI <= 24.0)
    X = X[good]; XErr = XErr[good]; magI = magI[good]; Y = Y[good]
    magR = X[:,1] + magI
    magG = X[:,0] + magR
    magZ = -X[:,2] + magI
    magRAbsHsc, dKpc = getParallax(magG, magR, magI, magZ)
    fig = plt.figure(figsize=(8, 6), dpi=120)
    ax = fig.add_subplot(2, 2, 1)
    ax.scatter(dKpc, magRAbsHsc, marker='.', s=1, color='black')
    ax = fig.add_subplot(2, 2, 2)
    ax.scatter(dKpc, X[:,1], marker='.', s=1, color='black')
    ax = fig.add_subplot(2, 2, 3)
    ax.scatter(dKpc, magI, marker='.', s=1, color='black')
    return fig

def precomputeTotalCount(field):
    try:
        fNamePost = '/scr/depot0/garmilla/HSC/wide{0}Posteriors.csv'.format(field)
        totalCount = fileLen(fNamePost) - 1
    except IOError:
        fNamePost = '/home/jose/Data/wide{0}Posteriors.csv'.format(field)
        totalCount = fileLen(fNamePost) - 1
    with open('totalCount{0}.txt'.format(field), 'w') as f:
        f.write('{0}\n'.format(totalCount))

def precomputeRadialCounts(field, riMin=0.0, riMax=0.4, nBins=8, nBinsD=10, subsetSize=200000, threshold=0.9):
    width = (riMax - riMin)/nBins
    ids, ra, dec, X, XErr, magI, Y = loadFieldData(field, subsetSize=subsetSize)
    ri = X[:,1]
    good = np.logical_and(True, magI <= 24.0)
    good = np.logical_and(good, X[:,1] < 0.4)
    good = np.logical_and(good, X[:,2] < 0.2)
    good = np.logical_and(good, Y >= threshold)
    ra = ra[good]; dec = dec[good]; ri = ri[good]
    X = X[good]; XErr = XErr[good]; magI = magI[good]; Y = Y[good]
    c = SkyCoord(ra=ra*units.degree, dec=dec*units.degree, frame='icrs')
    b = c.galactic.b.rad
    l = c.galactic.l.rad
    bMean = np.mean(b); lMean = np.mean(l)
    with open('meanCoord{0}.txt'.format(field), 'w') as f:
        f.write('{0}, {1}\n'.format(bMean, lMean))
    magR = X[:,1] + magI
    magG = X[:,0] + magR
    magZ = -X[:,2] + magI
    magRAbsHsc, dKpc = getParallax(magG, magR, magI, magZ)
    dKpcGal = np.sqrt(8.0**2 + dKpc**2 - 2*8.0*dKpc*np.cos(b)*np.cos(l))
    binMin = riMin
    dGrid = np.linspace(10.0, 100.0, num=nBinsD+1)
    counts = np.zeros((nBins, nBinsD))
    binCenters = np.zeros((nBinsD,))
    for i in range(nBins):
        binMax = binMin + width
        good = np.logical_and(ri > binMin, ri < binMax)
        for j in range(nBinsD):
            binCenters[j] = 0.5*(dGrid[j] + dGrid[j+1])
            inDBin = np.logical_and(dKpcGal[good] > dGrid[j], dKpcGal[good] < dGrid[j+1])
            counts[i][j] = np.sum(inDBin)*1.0
        binMin += width
    data = np.zeros((nBins+1, nBinsD))
    data[0,:] = binCenters
    for i in range(nBins):
        data[i+1, :] = counts[i, :]
    np.savetxt('radialCounts{0}.txt'.format(field), data)

def getCountErrorBar(counts, nPure, xPure, nComp, xComp, alpha=0.05, size=100000):
    samplesP = beta.rvs(xPure + 0.5, nPure - xPure + 0.5, size=size)
    samplesC = beta.rvs(xComp + 0.5, nComp - xComp + 0.5, size=size)
    samplesS = poisson.rvs(int(counts), size=size)
    samples = samplesS*samplesP/samplesC
    return 2*np.std(samples)

def makeTomographyCBins(riMin=0.0, riMax=0.4, nBins=8, nBinsD=10, subsetSize=100000, threshold=0.9, fontSize=18,
                        normalA=100.0, qH=0.5, nH=2.5, normalH=2.0e4):
    width = (riMax - riMin)/nBins
    fig = plt.figure(figsize=(24, 18), dpi=120)
    axes = []
    for i in range(nBins):
        axes.append(fig.add_subplot(3, 3, i+1))
    with open('purity.pkl', 'r') as f:
        purity = pickle.load(f)
    with open('completeness.pkl', 'r') as f:
        completeness = pickle.load(f)
    totalCounts = {}
    totalCount = 0
    for i, field in enumerate(_fields):
        totalCounts[field] = int(np.loadtxt('totalCount{0}.txt'.format(field)))
        totalCount += totalCounts[field]
    maxCounts = np.zeros((nBins,))
    for i, field in enumerate(_fields):
        data = np.loadtxt('radialCounts{0}.txt'.format(field))
        binCenters = data[0,:]
        binMin = riMin
        for j in range(nBins):
            binMax = binMin + width
            axes[j].set_title('{0} < r-i < {1}'.format(binMin, binMax), fontsize=fontSize)
            axes[j].set_xlabel('r (kpc)', fontsize=fontSize)
            axes[j].set_ylabel(r'counts ($\mathrm{kpc}^{-1}\mathrm{deg}^{-2}$)', fontsize=fontSize)
            counts = data[j+1,:]
            areaFactor = 100.0*totalCounts[field]/totalCount
            correction = np.zeros((nBinsD,))
            correction = np.zeros((nBinsD,))
            error = np.zeros((nBinsD,))
            riCenter = np.array([0.5*(binMin + binMax)])
            grCenter = 0.15785242 + 1.93645872*riCenter
            izCenter = -0.0207809 + 0.5644657*riCenter
            iCenter = np.array([24.0])
            rCenter = iCenter + riCenter
            gCenter = rCenter + grCenter
            zCenter = iCenter - izCenter
            magRAbsHsc, dKpc = getParallax(gCenter, rCenter, iCenter, zCenter)
            with open('meanCoord{0}.txt'.format(field), 'r') as f:
                reader = csv.reader(f)
                line = reader.next()
                b = float(line[0]); l = float(line[1])
                dEarth = binCenters*np.cos(b)*np.cos(l)+np.sqrt((8.0*np.cos(b)*np.cos(l))**2+binCenters**2-8.0**2)
                sinBStar = dEarth*np.sin(b)/binCenters
                gtr = np.logical_not(sinBStar <= 1.0)
                dEarth[gtr] = binCenters[gtr]*np.cos(b)*np.cos(l)-np.sqrt((8.0*np.cos(b)*np.cos(l))**2+binCenters[gtr]**2-8.0**2)
                sinBStar[gtr] = dEarth[gtr]*np.sin(b)/binCenters[gtr]
                cosBStar = np.sqrt(1.0 - sinBStar**2)
                RStar = binCenters*cosBStar
                ZStar = binCenters*sinBStar
                haloModel = normalH*np.power(np.sqrt(binCenters**2+(ZStar/qH)**2), -nH)
            dKpcGal = np.sqrt(8.0**2 + dKpc**2 - 2*8.0*dKpc*np.cos(b)*np.cos(l))
            for k in range(len(correction)):
                if completeness[j][k][0] == 0.0 or completeness[j][k][1] == 0.0 or\
                   purity[j][k][0] == 0.0 or purity[j][k][1] == 0.0 or\
                   counts[k] == 0.0:
                    correction[k] = 0.0
                    error[k] = 0.0
                else:
                    correction[k] = purity[j][k][1]/purity[j][k][0]/(completeness[j][k][1]/completeness[j][k][0])
                    error[k] = getCountErrorBar(counts[k], purity[j][k][0], purity[j][k][1], completeness[j][k][0], completeness[j][k][1])
            #import ipdb; ipdb.set_trace()
            #axes[j].plot(binCenters, counts/binCenters*correction/areaFactor, color=_colors[i])
            gtr = counts*correction < error
            error[gtr] = 0.999*counts[gtr]*correction[gtr]
            axes[j].plot(binCenters, haloModel*counts[0]/binCenters[0]*correction[0]/areaFactor/haloModel[0], color=_colors[i], linestyle='-')
            axes[j].errorbar(binCenters, counts/binCenters*correction/areaFactor, yerr=error/binCenters/areaFactor, fmt='o', color=_colors[i],
                             label=r'Limit = {:2.0f} kpc'.format(dKpcGal[0]))
            axes[j].plot([dKpcGal[0], dKpcGal[0]], [0.05, 40.0], linestyle='--', color=_colors[i])
            if (counts/binCenters*correction/areaFactor).max() > maxCounts[j]:
                maxCounts[j] = (counts/binCenters*correction/areaFactor).max()
                #axes[j].set_ylim((maxCounts[j]*1.0e-2, maxCounts[j]*2.0))
                axes[j].set_ylim((0.05, 40.0))
                axes[j].set_xlim((14.0, 120.0))
                axes[j].set_yscale('log')
                axes[j].set_xscale('log')
                axes[j].set_yticks([0.1, 0.2, 0.4, 1.0, 2.0, 4.0, 10.0, 20.0])
                axes[j].set_yticklabels(['0.1', '0.2', '0.4', '1.0', '2.0', '4.0', '10.0', '20.0'])
                axes[j].set_xticks([20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
                axes[j].set_xticklabels(['20.0', '', '40.0', '', '', '', '', '', '100.0'])
            binMin += width
    for ax in fig.get_axes():
        #ax.legend(loc='lower left')
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontSize)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontSize)
    dirHome = os.path.expanduser('~')
    fig.savefig(os.path.join(dirHome, 'Desktop/wideTomography.png'), dpi=120, bbox_inches='tight')
    plt.show()
    return fig

def makeCCDiagrams(field, threshold=0.9, subsetSize=100000, fontSize=18):
    magBins = [(18.0, 22.0), (22.0, 24.0), (24.0, 25.0)]
    ids, ra, dec, X, XErr, magI, Y = loadFieldData(field, subsetSize=subsetSize)
    magString = r'$\mathrm{Mag}_{cmodel}$ HSC-I'
    colNames = ['g-r', 'r-i', 'i-z', 'z-y']
    colLims = [(0.0, 1.5), (-0.2, 2.0), (-0.2, 1.0), (-0.2, 0.4)]
    fig = plt.figure(figsize=(24, 18), dpi=120)
    fig.suptitle(field, fontsize=fontSize)
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

def makeCMDiagram(field, subsetSize=100000, threshold=0.9, fontSize=18, cCut=None, mCut=None):
    ids, ra, dec, X, XErr, magI, Y = loadFieldData(field, subsetSize=subsetSize)
    stellar = np.logical_not(Y < threshold)
    good = False
    if cCut is not None:
        idx = cCut[0]; low = cCut[1]; high = cCut[2]
        goodC = np.logical_and(X[:,idx] > low, X[:,idx] < high)
        if mCut is None:
            good = goodC
    if mCut is not None:
        goodM = np.logical_and(magI > mCut[0], magI < mCut[1])
        if cCut is None:
            good = goodM
        else:
            good = np.logical_and(goodC, goodM)
    stellarGood = np.logical_and(good, stellar)
    stellarBad = np.logical_and(np.logical_not(good), stellar)
    fig = plt.figure(dpi=120)
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(X[:,1][stellarGood], magI[stellarGood], marker='.', s=1, color='red')
    ax.scatter(X[:,1][stellarBad], magI[stellarBad], marker='.', s=1, color='black')
    if cCut is not None and mCut is not None:
        ax.plot([cCut[1], cCut[1]], [mCut[0], mCut[1]], color='black')
        ax.plot([cCut[1], cCut[2]], [mCut[0], mCut[0]], color='black')
        ax.plot([cCut[1], cCut[2]], [mCut[1], mCut[1]], color='black')
        ax.plot([cCut[2], cCut[2]], [mCut[0], mCut[1]], color='black')
    ax.set_xlim((-0.1, 0.4))
    ax.set_ylim((18.0, 25.0))
    ax.set_xlabel(r'$r-i$', fontsize=fontSize)
    ax.set_ylabel(r'$\mathrm{Mag}_{cmodel}$ HSC-I', fontsize=fontSize)
    ax.invert_yaxis()
    dirHome = os.path.expanduser('~')
    fig.savefig(os.path.join(dirHome, 'Desktop/cmDiagram{0}.png'.format(field)), dpi=120, bbox_inches='tight')

def makeRaDecDiagram(field, subsetSize=100000, threshold=0.9, fontSize=18, cCut=None, mCut=None):
    ids, ra, dec, X, XErr, magI, Y = loadFieldData(field, subsetSize=subsetSize)
    stellar = np.logical_not(Y < threshold)
    good = False
    if cCut is not None:
        idx = cCut[0]; low = cCut[1]; high = cCut[2]
        goodC = np.logical_and(X[:,idx] > low, X[:,idx] < high)
        if mCut is None:
            good = goodC
    if mCut is not None:
        goodM = np.logical_and(magI > mCut[0], magI < mCut[1])
        if cCut is None:
            good = goodM
        else:
            good = np.logical_and(goodC, goodM)
    stellarGood = np.logical_and(good, stellar)
    stellarBad = np.logical_and(np.logical_not(good), stellar)
    fig = plt.figure(dpi=120)
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(ra[stellarGood], dec[stellarGood], marker='.', s=1, color='red')
    ax.scatter(ra[stellarBad], dec[stellarBad], marker='.', s=1, color='black')
    ax.set_xlabel('RA', fontsize=fontSize)
    ax.set_ylabel('Dec', fontsize=fontSize)
    ax.set_title('{0}'.format(field))
    dirHome = os.path.expanduser('~')
    fig.savefig(os.path.join(dirHome, 'Desktop/StarsRaDec{0}.png'.format(field)), dpi=120, bbox_inches='tight')

def makePurityCompletenessPlots(riMin=0.0, riMax=0.4, nBins=8, nBinsD=10, computePosteriors=False, fontSize=18,
                                threshold = 0.9, alpha=0.05):
    if computePosteriors:
        with open('trainSet.pkl', 'rb') as f:
            trainSet = pickle.load(f)
        magBins = [(18.0, 22.0), (22.0, 24.0), (24.0, 25.0), (25.0, 26.0)]
        with open('clfsColsExt.pkl', 'rb') as f:
            clfs = pickle.load(f)
        X, XErr, Y = trainSet.genColExtTrainSet(mode='all')
        ra = trainSet.getAllRas()
        dec = trainSet.getAllDecs()
        magI = trainSet.getAllMags(band='i')
        clfXd = dGauss.XDClfs(clfs=clfs, magBins=magBins)
        posteriors = clfXd.predict_proba(X, XErr, magI)
        with open('cosmosTomPosteriors.pkl', 'wb') as f:
            pickle.dump((ra, dec, X, XErr, magI, Y, posteriors), f)
    else:
        with open('cosmosTomPosteriors.pkl', 'rb') as f:
            ra, dec, X, XErr, magI, Y, posteriors = pickle.load(f)
    ri = X[:,1]
    good = np.logical_and(True, magI <= 24.0)
    good = np.logical_and(good, X[:,1] < 0.4)
    good = np.logical_and(good, X[:,2] < 0.2)
    ra = ra[good]; dec = dec[good]; ri = ri[good]; posteriors = posteriors[good]
    X = X[good]; XErr = XErr[good]; magI = magI[good]; Y = Y[good]
    c = SkyCoord(ra=ra*units.radian, dec=dec*units.radian, frame='icrs')
    b = c.galactic.b.rad
    l = c.galactic.l.rad
    magR = X[:,1] + magI
    magG = X[:,0] + magR
    magZ = -X[:,2] + magI
    magRAbsHsc, dKpc = getParallax(magG, magR, magI, magZ)
    dKpcGal = np.sqrt(8.0**2 + dKpc**2 - 2*8.0*dKpc*np.cos(b)*np.cos(l))
    labeledStar = np.logical_and(True, posteriors >= threshold)
    goodStar = np.logical_and(Y, posteriors >= threshold)
    badStar = np.logical_and(np.logical_not(Y), posteriors >= threshold)
    missedStar = np.logical_and(Y, posteriors < threshold)
    fig = plt.figure(figsize=(24, 18), dpi=120)
    width = (riMax - riMin)/nBins
    binMin = riMin
    dGrid = np.linspace(10.0, 100.0, num=nBinsD+1)
    dataP = np.zeros((nBins, nBinsD, 2))
    dataC = np.zeros((nBins, nBinsD, 2))
    for i in range(nBins):
        binMax = binMin + width
        ax = fig.add_subplot(3, 3, i+1)
        ax.set_title('{0} < r-i < {1}'.format(binMin, binMax), fontsize=fontSize)
        ax.set_xlabel('r (kpc)', fontsize=fontSize)
        ax.set_ylabel('Scores', fontsize=fontSize)
        ax.set_ylim((0.0, 1.0))
        inCBin = np.logical_and(ri > binMin, ri < binMax)
        purity = np.zeros((nBinsD,))
        completeness = np.zeros((nBinsD,))
        binCenters = np.zeros((nBinsD,))
        lPure = np.zeros((nBinsD,))
        uPure = np.zeros((nBinsD,))
        lComp = np.zeros((nBinsD,))
        uComp = np.zeros((nBinsD,))
        for j in range(nBinsD):
            binCenters[j] = 0.5*(dGrid[j] + dGrid[j+1])
            inDBin = np.logical_and(dKpcGal[inCBin] > dGrid[j], dKpcGal[inCBin] < dGrid[j+1])
            nPure = np.sum(labeledStar[inCBin][inDBin])
            xPure = np.sum(goodStar[inCBin][inDBin])
            lPure[j], uPure[j] = getJeffreysInterval(alpha, nPure, xPure)
            nComp = np.sum(Y[inCBin][inDBin])
            xComp = np.sum(goodStar[inCBin][inDBin])
            lComp[j], uComp[j] = getJeffreysInterval(alpha, nComp, xComp)
            dataP[i, j, 0] = np.sum(labeledStar[inCBin][inDBin])*1.0
            dataP[i, j, 1] = np.sum(goodStar[inCBin][inDBin])*1.0
            dataC[i, j, 0] = np.sum(Y[inCBin][inDBin])*1.0
            dataC[i, j, 1] = np.sum(goodStar[inCBin][inDBin])*1.0
            if np.sum(labeledStar[inCBin][inDBin]) == 0:
                purity[j] = 0.0
                lPure[j] = 0.0; uPure[j] = 0.0
            else:
                purity[j] = np.sum(goodStar[inCBin][inDBin])*1.0/(np.sum(labeledStar[inCBin][inDBin]))
            if np.sum(Y[inCBin][inDBin]) == 0:
                completeness[j] = 0.0
                lComp[j] = 0.0; uComp[j] = 0.0
            else:
                completeness[j] = np.sum(goodStar[inCBin][inDBin])*1.0/(np.sum(Y[inCBin][inDBin]))
        ax.step(binCenters, purity, color='blue', where='mid')
        ax.step(binCenters, completeness, color='red', where='mid')
        ax.errorbar(binCenters, purity, yerr=[purity-lPure, uPure-purity], color='blue', marker='o', fmt='o')
        ax.errorbar(binCenters, completeness, yerr=[completeness-lComp, uComp-completeness], color='red', marker='o', fmt='o')
        binMin += width
    for ax in fig.get_axes():
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontSize)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontSize)
    fig.tight_layout()
    ax = fig.add_subplot(3, 3, 9, frame_on=False)
    ax.plot([], [], color='blue', marker='o', label='Purity')
    ax.plot([], [], color='red', marker='o', label='Completeness')
    ax.legend(loc='center', prop={'size':40})
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    dirHome = os.path.expanduser('~')
    fig.savefig(os.path.join(dirHome, 'Desktop/wideTomScores.png'), dpi=120, bbox_inches='tight')
    with open('purity.pkl', 'w') as f:
        pickle.dump(dataP, f)
    with open('completeness.pkl', 'w') as f:
        pickle.dump(dataC, f)
    return fig

def _getFl(a=0.064, b=0.970, c=0.233, d=0.776, dGal=10.0, bGal=0.0):
    rhs = 1.0/np.cos(bGal)*(-c*np.sin(bGal) - d/dGal)
    def F(l):
        return a*np.cos(l) + b*np.sin(l) - rhs
    return F

def _getFb(a=0.064, b=0.970, c=0.233, d=0.776, dGal=10.0, l=0.0):
    lhs = a*np.cos(l) + b*np.sin(l)
    def F(bGal):
        return 1.0/np.cos(bGal)*(-c*np.sin(bGal) - d/dGal) - lhs
    return F

def getLBPairsSg(dGal=10.0):
    l = np.linspace(-np.pi, np.pi, num=100)
    b = np.zeros(l.shape)
    for i in range(len(l)):
        F = _getFb(l=l[i])
        brentMin = -np.pi/2; brentMax = np.pi/2
        b[i] = brentq(F, brentMin, brentMax)
    return l, b

def getLBPairsOrphan():
    phi = 128.79*np.pi/180
    theta = 54.39*np.pi/180
    chi = 90.70*np.pi/180
    cosP = np.cos(phi); sinP = np.sin(phi)
    cosT = np.cos(theta); sinT = np.sin(theta)
    cosC = np.cos(chi); sinC = np.sin(chi)
    M = np.array([[cosC*cosP-cosT*sinP*sinC, cosC*sinP+cosT*cosP*sinC, sinC*sinT],
                  [-sinC*cosP-cosT*sinP*cosC, -sinC*sinP+cosT*cosP*cosC, cosC*sinT],
                  [sinT*sinP, -sinT*cosP, cosT]])
    MInv = inv(M)
    Ls = np.linspace(-np.pi, np.pi, num=100)
    l = np.zeros(Ls.shape)
    b = np.zeros(Ls.shape)
    for i in range(len(Ls)):
        vec = np.array([np.cos(Ls[i]), np.sin(Ls[i]), 0.0])
        vecGal = np.dot(MInv, vec)
        assert np.abs(vecGal[2]) <= 1.0
        b[i] = np.arcsin(vecGal[2])
        try:
            assert np.abs(vecGal[0]/np.cos(b[i])) <= 1.0
            assert np.abs(vecGal[1]/np.cos(b[i])) <= 1.0
            assert np.allclose(np.square(vecGal[1]/np.cos(b[i])) + np.square(vecGal[0]/np.cos(b[i])), 1.0)
        except AssertionError:
            import ipdb; ipdb.set_trace()
        if vecGal[1]/np.cos(b[i]) >= 0.0:
            l[i] = np.arccos(vecGal[0]/np.cos(b[i]))
        else:
            l[i] = - np.arccos(vecGal[0]/np.cos(b[i]))
    return l[l.argsort()], b[l.argsort()]

def getLBPairsGD1():
    M = np.array([[-0.4776303088, -0.1738432154, 0.8611897727],
                  [0.510844589, -0.8524449229, 0.111245042],
                  [0.7147776536, 0.4930681392, 0.4959603976]])
    MInv = inv(M)
    phi1 = np.linspace(-60.0*np.pi/180, 0.0, num=10)
    alpha = np.zeros(phi1.shape)
    delta = np.zeros(phi1.shape)
    for i in range(len(phi1)):
        vec = np.array([np.cos(phi1[i]), np.sin(phi1[i]), 0.0])
        vecGal = np.dot(MInv, vec)
        assert np.abs(vecGal[2]) <= 1.0
        delta[i] = np.arcsin(vecGal[2])
        try:
            assert np.abs(vecGal[0]/np.cos(delta[i])) <= 1.0
            assert np.abs(vecGal[1]/np.cos(delta[i])) <= 1.0
            assert np.allclose(np.square(vecGal[1]/np.cos(delta[i])) + np.square(vecGal[0]/np.cos(delta[i])), 1.0)
        except AssertionError:
            import ipdb; ipdb.set_trace()
        if vecGal[1]/np.cos(delta[i]) >= 0.0:
            alpha[i] = np.arccos(vecGal[0]/np.cos(delta[i]))
        else:
            alpha[i] = 2*np.pi - np.arccos(vecGal[0]/np.cos(delta[i]))
    c = SkyCoord(ra=alpha, dec=delta, frame='icrs', unit='deg')
    b = c.galactic.b.rad
    l = c.galactic.l.rad
    return l[l.argsort()], b[l.argsort()]

def getLBPairsPal5(radec0=(226.5, -3.0), radec1=(234.0, 3.5)):
    ras = np.linspace(radec0[0], 234.0, num=5)
    decs = radec0[1] + (ras - radec0[0])*(radec1[1] - radec0[1])/(radec1[0] - radec0[0])
    c = SkyCoord(ra=ras, dec=decs, frame='icrs', unit='deg')
    b = c.galactic.b.rad
    l = c.galactic.l.rad
    return l[l.argsort()], b[l.argsort()]

def getLBPairsMClouds():
    raSmall = '00h52m44.8s'
    decSmall = '-72d49m43s'
    c = SkyCoord(ra=raSmall, dec=decSmall, frame='icrs')
    bSmall = c.galactic.b.rad
    lSmall = c.galactic.l.rad - 2*np.pi
    raLarge = '05h23m34.5s'
    decLarge = '-69d45m22s'
    c = SkyCoord(ra=raLarge, dec=decLarge, frame='icrs')
    bLarge = c.galactic.b.rad
    lLarge = c.galactic.l.rad - 2*np.pi
    return [lSmall, lLarge], [bSmall, bLarge]

def getLBPairsCEquator():
    ras = np.linspace(0.0, 360.0, num=100)
    decs = np.zeros(ras.shape)
    c = SkyCoord(ra=ras, dec=decs, frame='icrs', unit='deg')
    b = c.galactic.b.rad
    l = c.galactic.l.rad
    gtr = np.logical_not(l < np.pi)
    l[gtr] = l[gtr] - 2*np.pi
    return l[l.argsort()], b[l.argsort()]

def makeWideGallacticProjection(subsetSize=1000, fontSize=16):
    fig = plt.figure(dpi=120)
    ax = fig.add_subplot(111, projection='mollweide')
    ax.grid()
    l, b = getLBPairsSg(dGal=5.0)
    ax.plot(l, b, color='black')
    l, b = getLBPairsSg(dGal=20.0)
    ax.plot(l, b, color='black')
    l, b = getLBPairsOrphan()
    ax.plot(l, b, color='black')
    l, b = getLBPairsPal5()
    ax.scatter(l, b, color='black', marker='x')
    l, b = getLBPairsGD1()
    #ax.scatter(l, b, color='black', marker='x')
    ax.plot(l, b, color='black')
    l, b = getLBPairsMClouds()
    ax.scatter(l, b, color='black', marker='x')
    l, b = getLBPairsCEquator()
    ax.plot(l, b, color='black', linestyle=':')
    #for i, field in enumerate(_fields):
    #    ids, ra, dec, X, XErr, magI, Y = loadFieldData(field, subsetSize=subsetSize)
    #    c = SkyCoord(ra=ra*units.degree, dec=dec*units.degree, frame='icrs')
    #    b = c.galactic.b.rad
    #    l = c.galactic.l.rad
    #    gtr = np.logical_and(True, l > np.pi)
    #    l[gtr] = l[gtr] - 2*np.pi
    #    ax.scatter(l, b, marker='.', s=1, color=_colors[i], edgecolor="none")
    ax.set_xlabel('l', fontsize=fontSize)
    ax.set_ylabel('b', fontsize=fontSize)
    ax.set_title('HSC Wide January 2016', fontsize=fontSize)
    for ax in fig.get_axes():
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontSize)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontSize)
    dirHome = os.path.expanduser('~')
    fig.savefig(os.path.join(dirHome, 'Desktop/wideProjection.png'), dpi=120, bbox_inches='tight')
    return fig
    
if __name__ == '__main__':
    #field = 'VVDS'
    #computeFieldPosteriors(field)
    #makeCCDiagrams(field)
    #makeWideGallacticProjection()
    #makeTomographyCBins()
    #genDBPosts('HectoMap')
    preLoadField('XMM')
    #for field in _fields:
        #makeCCDiagrams(field)
        #precomputeRadialCounts(field, subsetSize=None)
        #precomputeTotalCount(field)
