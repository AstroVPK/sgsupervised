import pickle

import supervisedEtl as etl

bands = ['g', 'r', 'i', 'z', 'y']

cat = etl.afwTable.SimpleCatalog.readFits('/scr/depot0/garmilla/HSC/matchDeepCoaddMeas-136120151104CosmosGRIZY.fits')

trainSet = etl.extractTrainSet(cat, inputs=['snrPsf', 'ext', 'extHsmDeconv', 'seeing'], bands=bands, concatBands=False)
with open('trainSetGRIZY.pkl', 'wb') as f: pickle.dump(trainSet, f)

for band in bands:
    trainSet = etl.extractTrainSet(cat, inputs=['snrPsf', 'ext', 'extHsmDeconv', 'seeing'], bands=[band])
    with open('trainSet{0}.pkl'.format(band.upper()), 'wb') as f: pickle.dump(trainSet, f)
