import csv

import numpy as np

import lsst.afw.table as afwTable

import fsButler.utils as fsUtils

inputFile = '/tigress/garmilla/data/wideHscClass.csv'  # Reduction of the COSMOS field
# inputFile = '/tigress/garmilla/data/29629.csv'

with open(inputFile) as f:
    for i, l in enumerate(f):
        pass
fileLen = i

with open(inputFile, 'r') as f:
    line = f.readline()
    cList = line.split()[1].split(',')

schema = afwTable.SimpleTable.makeMinimalSchema()
for c in cList:
    # if c not in ['id', 'ra2000', 'decl2000']:
    if c not in ['id', 'ra', 'decl']:
        schema.addField(afwTable.Field["F"](c, c))
cat = afwTable.SimpleCatalog(schema)
cat.reserve(fileLen)
for i in range(fileLen):
    cat.addNew()

idCol = np.zeros((fileLen,), dtype=long)
raCol = np.zeros((fileLen,))
declCol = np.zeros((fileLen,))
# photArr = np.zeros((fileLen, 20))
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
        # for j in range(3, 23):
        for j in range(3, 28):
            try:
                photArr[i, j-3] = float(line[j])
            except ValueError:
                photArr[i, j-3] = None
cat.get('id')[:] = idCol
cat.get('coord.ra')[:] = np.radians(raCol)
cat.get('coord.dec')[:] = np.radians(declCol)
for c in cList:
    # if c not in ['id', 'ra2000', 'decl2000']:
    if c not in ['id', 'ra', 'dec']:
        cIndex = cList.index(c) - 3
        cat.get(c)[:] = photArr[:, cIndex]

selectSG = "/tigress/garmilla/data/cosmos_sg_all.fits"  # Alexii's catalog
sgTable = afwTable.SimpleCatalog.readFits(selectSG)
sgTable["coord.ra"][:] = np.radians(sgTable["coord.ra"])
sgTable["coord.dec"][:] = np.radians(sgTable["coord.dec"])

match = fsUtils.matchCats(sgTable, cat, multiMeas=False, includeMismatches=True)
# match.writeFits('/tigress/garmilla/data/matchS15B.fits')
match.writeFits('/tigress/garmilla/data/wideHscClass.fits')
