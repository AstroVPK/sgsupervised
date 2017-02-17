import os
import cPickle as pickle

import lsst.afw.table as afwTable

import supervisedEtl as etl

depth = 'udeepwide'

# Assume the input data lives in the ``input`` directory located relative to
# this file. Obviously, this is an ugly hack.
inputDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "input")

matchedCatFile = os.path.join(inputDir, '%sHscClass.fits'%(depth))
trainSetFile = os.path.join(inputDir, '%sTrainSet.pkl'%(depth))

matchedCat = afwTable.SimpleCatalog.readFits(matchedCatFile)
trainSet = etl.extractTrainSet(matchedCat, inputs=['mag'], bands=['g', 'r', 'i', 'z', 'y'], withErr=True,
                               mode='colors', concatBands=False, fromDB=True)
pickle.dump(trainSet, open(trainSetFile, 'wb'))
