import os
import csv
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
import pdb

import lsst.afw.table as afwTable

import fsButler.utils as fsUtils

import dGauss
import supervisedEtl as etl

depth = 'unclass'

# Assume the input data lives in the ``input`` directory located relative to
# this file. Obviously, this is an ugly hack.
inputDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "input")

selectSG = os.path.join(inputDir, 'cosmos_sg_all.fits')  # Alexii's catalog
classifierFile = os.path.join(inputDir, 'udeepwideClfsColsExt.pkl')

inputFile = os.path.join(inputDir, '%s.csv'%(depth))  # Reduction of the COSMOS field
matchedCatFile = os.path.join(inputDir, '%sHscClass.fits'%(depth))
classifySetFile = os.path.join(inputDir, '%sClassifySet.pkl'%(depth))

with open(inputFile) as f:
    for i, l in enumerate(f):
        pass
fileLen = i
print 'Number of entries: %d'%(fileLen)

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
mockSGTable = afwTable.SimpleCatalog(sgTable.schema)
mockSGTable.reserve(fileLen)
for i in xrange(len(cat)):
    mockSGTable.addNew()
    mockSGTable.get('id')[i] = cat.get('id')[i]
    mockSGTable.get('coord_ra')[i] = cat.get('coord_ra')[i]
    mockSGTable.get('coord_dec')[i] = cat.get('coord_dec')[i]
    mockSGTable.get('mu_class')[i] = 0
    mockSGTable.get('mag_auto')[i] = cat.get('imag')[i]

matchedCat = fsUtils.matchCats(mockSGTable, cat, multiMeas=False, includeMismatches=True)
matchedCat.writeFits(matchedCatFile)

classifySet = etl.extractTrainSet(matchedCat, inputs=['mag'], bands=['g', 'r', 'i', 'z', 'y'], withErr=True,
                                  mode='colors', concatBands=False, fromDB=True)
pickle.dump(classifySet, open(classifySetFile, 'wb'))

X, XErr, Y = classifySet.genColExtTrainSet(mode='all')
mags = classifySet.getAllMags(band='i')
magBins = [(18.0, 22.0), (22.0, 24.0), (24.0, 25.0), (25.0, 26.0)]
with open(classifierFile, 'rb') as f:
    clfs = pickle.load(f)
clfXd = dGauss.XDClfs(clfs=clfs, magBins=magBins)
posteriors = clfXd.predict_proba(X, XErr, mags)  # Posterior likelihood that something is a star...

ids = classifySet.getAllIds()
rexts = np.zeros(ids.shape[0])
rmags = np.zeros(ids.shape[0])
iexts = np.zeros(ids.shape[0])
imags = np.zeros(ids.shape[0])
idsCat = cat.get('id')
rextCat = cat.get('rext')
rmagCat = cat.get('rmag')
iextCat = cat.get('iext')
imagCat = cat.get('imag')
for i in xrange(ids.shape[0]):
    index = np.where(ids[i] == idsCat)[0][0]
    rexts[i] = rextCat[index]
    rmags[i] = rmagCat[index]
    iexts[i] = iextCat[index]
    imags[i] = imagCat[index]

fig = plt.figure(1)
plt.scatter(rmags, rexts, c=posteriors, edgecolors='none')
plt.xlabel('$mag_{\mathrm{r}, \mathrm{cmodel}}$')
plt.ylabel('$mag_{\mathrm{r}, \mathrm{psf}} - mag_{\mathrm{r}, \mathrm{cmodel}}$')
cBar = plt.colorbar()
cBar.set_label('$\mathfrak{P}(\mathrm{star})$')
fig.savefig(os.path.join(sgsDir, '%sClassifyRes_r.png'%(depth)), dpi=120, bbox_inches='tight')

fig = plt.figure(2)
plt.scatter(imags, iexts, c=posteriors, edgecolors='none')
plt.xlabel('$mag_{\mathrm{i}, \mathrm{cmodel}}$')
plt.ylabel('$mag_{\mathrm{i}, \mathrm{psf}} - mag_{\mathrm{i}, \mathrm{cmodel}}$')
cBar = plt.colorbar()
cBar.set_label('$\mathfrak{P}(\mathrm{star})$')
fig.savefig(os.path.join(sgsDir, '%sClassifyRes_i.png'%(depth)), dpi=120, bbox_inches='tight')
