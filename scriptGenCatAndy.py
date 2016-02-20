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
mags = trainSet.getAllMags()
gi = mags[:,0] - mags[:,2]
magsI = mags[:,2]; exts = trainSet.getAllExts()[:,2]
ids = trainSet.getAllIds(); ras = trainSet.getAllRas(); decs = trainSet.getAllDecs()

good = np.logical_and(True, exts < 0.4)
X = X[good]; XErr = XErr[good]; Y = Y[good]
gi = gi[good]
magsI = magsI[good]; exts = exts[good]
ids = ids[good]; ras = ras[good]; decs = decs[good]
pStar = np.zeros(magsI.shape)
pStar -= 1

good = getGood(magsI, 18.0, 22.0)
clf = dGauss.XDClf(ngStar=15, ngGal=15)
clf.fit(X[good], XErr[good], Y[good])
pStar[good] = clf.predict_proba(X[good], XErr[good])

good = getGood(magsI, 22.0, 24.0)
clf = dGauss.XDClf(ngStar=10, ngGal=15)
clf.fit(X[good], XErr[good], Y[good])
pStar[good] = clf.predict_proba(X[good], XErr[good])

good = getGood(magsI, 24.0, 25.0)
clf = dGauss.XDClf(ngStar=10, ngGal=15)
clf.fit(X[good], XErr[good], Y[good])
pStar[good] = clf.predict_proba(X[good], XErr[good])

good = getGood(magsI, 25.0, 26.0)
clf = dGauss.XDClf(ngStar=10, ngGal=10)
clf.fit(X[good], XErr[good], Y[good])
pStar[good] = clf.predict_proba(X[good], XErr[good])

#good = np.logical_and(True, pStar >= 0.0)
#ids = ids[good]; ras = ras[good]; decs = decs[good]
#gi = gi[good]
#magsI = magsI[good]; exts = exts[good]; pStar = pStar[good]

arr = np.zeros((len(ids),), dtype=('i8, f4, f4, f4, f4, f5, f6'))
arr['f0'] = ids[:,2]; arr['f1'] = ras[:,2]; arr['f2'] = decs[:,2]
arr['f3'] = magsI; arr['f4'] = gi; arr['f5'] = exts; arr['f6'] = pStar
header = 'id, RA (rads), Dec (rads), magnitude HSC-I, g-i, extendedness (mag_psf-mag_model), probability of being a star based on colors only'
np.savetxt('ucd.dat', arr, fmt='%d, %.18e, %.18e, %.18e, %.18e, %.18e, %.18e', header=header)
