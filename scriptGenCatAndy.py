import numpy as np
import pickle

import dGauss

def getGood(mags, magMin, magMax):
    good = True
    good = np.logical_and(good, mags > magMin)
    good = np.logical_and(good, mags < magMax)
    return good

with open('trainSet.pkl', 'rb') as f: trainSet = pickle.load(f)

X, XErr, Y = trainSet.getAllSet(standardized=False)
mags = trainSet.getAllMags()[:,2]; exts = trainSet.getAllExts()
ids = trainSet.getAllIds(); ras = trainSet.getAllRas(); decs = trainSet.getAllDecs()

good = np.logical_and(True, exts < 0.4)
X = X[good]; XErr = XErr[good]; Y = Y[good]
mags = mags[good]; exts = exts[good]
ids = ids[good]; ras = ras[good]; decs = decs[good]
pStar = np.zeros(mags.shape)
pStar -= 1

good = getGood(mags, 18.0, 22.0)
clf = dGauss.XDClf(ngStar=15, ngGal=15)
clf.fit(X[good], XErr[good], Y[good])
pStar[good] = clf.predict_proba(X[good], XErr[good])

good = getGood(mags, 22.0, 24.0)
clf = dGauss.XDClf(ngStar=10, ngGal=15)
clf.fit(X[good], XErr[good], Y[good])
pStar[good] = clf.predict_proba(X[good], XErr[good])

good = getGood(mags, 24.0, 25.0)
clf = dGauss.XDClf(ngStar=10, ngGal=15)
clf.fit(X[good], XErr[good], Y[good])
pStar[good] = clf.predict_proba(X[good], XErr[good])

good = getGood(mags, 25.0, 26.0)
clf = dGauss.XDClf(ngStar=10, ngGal=10)
clf.fit(X[good], XErr[good], Y[good])
pStar[good] = clf.predict_proba(X[good], XErr[good])

good = np.logical_and(True, pStar >= 0.0)
ids = ids[good]; ras = ras[good]; decs = decs[good]
exts = exts[good]; pStar = pStar[good]

arr = np.zeros((len(ids),), dtype=('i8, f4, f4, f4, f4, f5'))
arr['f0'] = ids[:,2]; arr['f1'] = ras[:,2]; arr['f2'] = decs[:,2]
arr['f3'] = mags; arr['f4'] = exts; arr['f5'] = pStar
header = 'id, RA (rads), Dec (rads), magnitude HSC-I, extendedness (mag_psf-mag_model), probability of being a star based on colors only'
np.savetxt('ucd.dat', arr, fmt='%d, %.18e, %.18e, %.18e, %.18e, %.18e', header=header)
