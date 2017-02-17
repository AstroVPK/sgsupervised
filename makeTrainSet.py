import os
import cPickle as pickle

import lsst.afw.table as afwTable

import supervisedEtl as etl

depth = 'deep'

# sgsDir should point at a directory called sgs.
# Assuming that the user is using lsstsw to build the lsst stack, and that the sgs folder is located at the
# same level as the lsstsw folder, the we use the EUPS_PATH environment variable to get the location of sgs.
sgsDir = os.path.join(os.environ['EUPS_PATH'], '..', '..', 'sgs')
matchedCatFile = os.path.join(sgsDir, '%sHscClass.fits'%(depth))
trainSetFile = os.path.join(sgsDir, '%sTrainSet.pkl'%(depth))

matchedCat = afwTable.SimpleCatalog.readFits(matchedCatFile)
trainSet = etl.extractTrainSet(matchedCat, inputs=['mag'], bands=['g', 'r', 'i', 'z', 'y'], withErr=True,
                               mode='colors', concatBands=False, fromDB=True)
pickle.dump(trainSet, open(trainSetFile, 'wb'))
