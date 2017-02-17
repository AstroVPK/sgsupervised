from __future__ import print_function
import os
import csv
import numpy as np

import lsst.afw.table as afwTable

import fsButler.utils as fsUtils

depth = 'udeepwide'

sgsDir = os.path.join(os.environ['EUPS_PATH'], '..', '..', 'sgs')  # sgsDir should point at the directory
# that Alexii Leauthaud's catalog lives in. Assuming that we are using lsstsw to build the stack and that
# Alexii Leauthaud's catalog lives in a directory called sgs that lies at the same-tree level as lsstsw, we
# use EUPS_PATH to get the location of sgs.
selectSG = os.path.join(sgsDir, 'cosmos_sg_all.fits')  # Alexii's catalog
inputFile = os.path.join(sgsDir, '%s.csv'%(depth))  # Reduction of the COSMOS field
outputFile = os.path.join(sgsDir, '%sHscClass.fits'%(depth))

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
    reader = csv.reader(f, delimiter=',')
    reader.next()  # Skip comments
    for line in reader:
        i = reader.line_num - 2
        idCol[i] = long(line[0])
        try:
            raCol[i] = float(line[1])
        except ValueError:
            raCol[i] = None
        try:
            declCol[i] = float(line[2])
        except ValueError:
            declCol[i] = None
        for j in range(3, 28):
            try:
                photArr[i, j-3] = float(line[j])
            except ValueError:
                photArr[i, j-3] = None
cat.get('id')[:] = idCol
cat.get('coord_ra')[:] = np.radians(raCol)
cat.get('coord_dec')[:] = np.radians(declCol)
for c in cList:
    if c not in ['id', 'ra2000', 'decl2000']:
        cIndex = cList.index(c) - 3
        cat.get(c)[:] = photArr[:, cIndex]

sgTable = afwTable.SimpleCatalog.readFits(selectSG)

match = fsUtils.matchCats(sgTable, cat, multiMeas=False, includeMismatches=True)
match.writeFits(outputFile)
