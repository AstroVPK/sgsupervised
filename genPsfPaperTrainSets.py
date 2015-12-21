import pickle

import supervisedEtl as etl

bands = ['g', 'r', 'i', 'z', 'y']

cat = etl.afwTable.SimpleCatalog.readFits('/home/jose/Data/matchDeepCoaddMeas-137520151126CosmosGRIZY.fits')

trainSet = etl.extractTrainSet(cat, inputs=['snrPsf', 'ext', 'extHsmDeconv', 'seeing', 'dGaussRadInner', 'dGaussRadRat', 'dGaussAmpRat'], bands=bands, concatBands=False)
with open('trainSetGRIZY.pkl', 'wb') as f: pickle.dump(trainSet, f)

for band in bands:
    trainSet = etl.extractTrainSet(cat, inputs=['snrPsf', 'ext', 'extHsmDeconv', 'seeing', 'dGaussRadInner', 'dGaussRadRat', 'dGaussAmpRat'], bands=[band])
    with open('trainSet{0}.pkl'.format(band.upper()), 'wb') as f: pickle.dump(trainSet, f)
