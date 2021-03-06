import pickle

import supervisedEtl as etl

from lsst.pex.exceptions import LsstCppException

concatBands=False
bands = ['g', 'r', 'i', 'z', 'y']

try:
    cat = etl.afwTable.SimpleCatalog.readFits('/scr/depot0/garmilla/HSC/matchDeepCoaddMeas-137520151126CosmosGRIZY.fits')
except LsstCppException:
    cat = etl.afwTable.SimpleCatalog.readFits('/home/jose/Data/matchDeepCoaddMeas-137520151126CosmosGRIZY.fits')

trainSet = etl.extractTrainSet(cat, inputs=['snrPsf', 'snrAp', 'mag', 'ext', 'extHsmDeconv', 'seeing', 'dGaussRadInner', 'dGaussRadRat', 'dGaussAmpRat'], bands=bands, concatBands=concatBands)
if concatBands:
    with open('trainSetConcatGRIZY.pkl', 'wb') as f: pickle.dump(trainSet, f)
else:
    with open('trainSetGRIZY.pkl', 'wb') as f: pickle.dump(trainSet, f)

for band in bands:
    trainSet = etl.extractTrainSet(cat, inputs=['snrPsf', 'snrAp', 'mag', 'ext', 'extHsmDeconv', 'seeing', 'dGaussRadInner', 'dGaussRadRat', 'dGaussAmpRat'], bands=[band])
    with open('trainSet{0}.pkl'.format(band.upper()), 'wb') as f: pickle.dump(trainSet, f)
