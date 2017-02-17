from __future__ import print_function
import os
import csv
import numpy as np
import cPickle as pickle

import lsst.afw.table as afwTable

import fsButler.utils as fsUtils

import supervisedEtl as etl

depth = 'udeepwide'

# Assume the input data lives in the ``input`` directory located relative to
# this file. Obviously, this is an ugly hack.
inputDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "input")

selectSG = os.path.join(inputDir, 'cosmos_sg_all.fits')  # Alexii's catalog
inputFile = os.path.join(inputDir, '%s.csv'%(depth))  # Reduction of the COSMOS field
matchedCatFile = os.path.join(inputDir, '%sHscClass.fits'%(depth))
trainSetFile = os.path.join(inputDir, '%sTrainSet.pkl'%(depth))

with open(inputFile) as f:
    for i, l in enumerate(f):
        pass
fileLen = i
print('Number of entries: %d'%(fileLen))

with open(inputFile, 'r') as f:
    line = f.readline()
    cList = line.split()[1].split(',')

schema = afwTable.SimpleTable.makeMinimalSchema()
for c in cList:
    if c not in ['id', 'ra2000', 'decl2000']:
        schema.addField(afwTable.Field["F"](c, c))
cat = afwTable.SimpleCatalog(schema)
cat.reserve(fileLen)
for i in range(fileLen):
    cat.addNew()

idCol = np.zeros((fileLen,), dtype=long)
raCol = np.zeros((fileLen,))
declCol = np.zeros((fileLen,))
photArr = np.zeros((fileLen, 25))
with open(inputFile, 'r') as f:
    line = f.readline()
    record = 0
    for line in f:
        words = line.rstrip('\n').split(',')
        idCol[record] = long(words[0])
        try:
            raCol[record] = float(words[1])
        except ValueError:
            raCol[record] = None
        try:
            declCol[record] = float(words[2])
        except ValueError:
            declCol[record] = None
        for j in range(3, 28):
            try:
                photArr[record, j-3] = float(words[j])
            except ValueError:
                photArr[record, j-3] = None
        record += 1
cat.get('id')[:] = idCol
cat.get('coord_ra')[:] = np.radians(raCol)
cat.get('coord_dec')[:] = np.radians(declCol)
for c in cList:
    if c not in ['id', 'ra2000', 'decl2000']:
        cIndex = cList.index(c) - 3
        cat.get(c)[:] = photArr[:, cIndex]

sgTable = afwTable.SimpleCatalog.readFits(selectSG)

matchedCat = fsUtils.matchCats(sgTable, cat, multiMeas=False, includeMismatches=True)
matchedCat.writeFits(matchedCatFile)

trainSet = etl.extractTrainSet(matchedCat, inputs=['mag'], bands=['g', 'r', 'i', 'z', 'y'], withErr=True,
                               mode='colors', concatBands=False, fromDB=True)
pickle.dump(trainSet, open(trainSetFile, 'wb'))
