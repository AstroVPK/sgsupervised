####################################################################
Star/Galaxy Separator (SGS) for HSC using Extreme Deconvolution (XD)
####################################################################

An Extreme Deconvolution (XD) based Star/Galaxy Separator (SGS) algorithm was developed by `Jose Garmilla`_ for his doctoral work at Princeton. SGS is designed to classify objects as either stars or galaxies using colors and extendedness. SGS is optimized for the `HSC`_ survey. Interested readers are encouraged to read [Garmilla2016]_ for details on the problem, the approach adopted, and the results obtained. This document assumes that the reader is familiar with [Garmilla2016]_.

SGS requires the `LSST Stack`_, the `LSST analysis`_ repository, and Jo Bovy's `Extreme Deconvolution`_ package in order to run. SGS is implemented in Python and is spread over two git repositories:

#. `sgsupervised`_ (this repository): all the SGS code, documentation and input data.

   Code for the classifier itself is in the ``classifier`` directory. It includes:

    - ``fromDbToTruth.py``: Python script that matches HSC catalogs i.e. ``<depth>.csv``, to Alexie Leauthaud's HST catalog containing star v/s galaxy classifications of objects (i.e. ``cosmos_sg_all.fits``) from COSMOS. After applying some of the cuts described in [Garmilla2016]_, the script writes the resulting classified HSC catalog out with the name ``<depth>HscClass.fits``. Further cuts from [Garmilla2016]_ are then applied to the intermediate catalog to produce the final training set called ``<depth>TrainSet.pkl``.
    - ``supervisedEtl.py``: Python library containing functionality to make training sets including making cuts as described in [Garmilla2016]_, and training the SGS classifier.
    - ``sgSVM.py``: Python library containing auxiliary functionality to train the SVM.
    - ``dGauss.py``: Python library that contains the definition of the SGS classifier - ``sgsupervised.dGauss.XDClfs``.
    - ``utils.py``: Python library of miscellaneous utility functions.
    - ``plotsScripts.py``: Python script that contains the functionality to train the SGS classifier as well as functions that generate various plots of interest.
    - ``classifyObjects.py``: Python script that uses an existing trained SGS classifier and classifies a set of unknown objects by computing the posterior likelihood of the object being a star given the object's magnitudes, extendedness, and the training data.

   Input data is in the ``input`` directory. It includes:

    - ``SQL.txt``: Sample SQL query to run on the HSC database in order to generate a ``.csv`` file of the objects to be used to create the training set. The ``.csv`` files should be named using the form ``<depth>.csv`` where <depth> is one of wide, deep, or udeep.
    - ``cosmos_sg_all_GOLD.fits``: Original copy of Alexie Leauthaud's HST catalog from [Leauthaud2007]_ containing star v/s galaxy classifications for objects in the COSMOS field. Note that this catalog uses degrees rather than radians when giving positions (RA & Dec).
    - ``cosmos_sg_all.fits``: Copy of cosmos_sg_all_GOLD.fits where the position measurements have been converted from degrees to radians so that the catalog can be read in using the `LSST Stack`_.

#. `sgs-fsbutler`_: Utility package for interfacing with the results of SQL queries on the HSC database. Users do not interact directly with the contents of this package. As such, the package can effectively be regarded as a black-box. This package managed by `EUPS`_; declare it and set it up as follows::

   $ eups declare fsButler original -r $EUPS_PATH/../../sgs-fsbutler
   $ eups setup -v -j fsButler

The procedure for using SGS is:

#. Train an instance of the SGS classifier:

    #. Run an SQL query (such as the sample query in SQL.txt) on the HSC database to get a list of objects along with properties such as fluxes and extendedness i.e. produce ``<depth>.csv``. Note that the depth is set in the ``from`` statement of the query line i.e. edit the ``from`` statement in order to select a different depth.
    #. Run ``fromDbToTruth.py`` to match the HSC detections against the HST master list and produce ``<depth>HscClass.fits`` and ``<depth>TrainSet.pkl``. This script takes no arguments - the depth is selected by editing the ``depth`` variable in the script.
    #. Run the function ``xdColExtFitScores(trainClfs=True)`` in ``plotsScripts.py`` to produce the SGS classifier object which gets written to disk in ``<depth>ClfsColsExt.pkl``. This script takes no arguments - the depth is selected by editing the ``depth`` variable in the script.

#. Use an instance of the SGS classifier to classify objects:

    #. Obtain a ``<classify>.csv`` of the (un-classified) objects with the same columns as the ``.csv`` files used to make the training set.
    #. Use ``classifyObjects.py`` to read in the ``<classify>.csv`` file and use an existing SGS classifier (stored in a pickle file named ``<depth>ClfsColsExt.pkl``) to assigned the posterior likelihood of the object being a star. For example::

          with open('unclassTrainSet.pkl', 'rb') as f:
              classifySet = pickle.load(f)
          X, XErr, Y = classifySet.getTrainSet(standardized=False)
          X, XErr, Y = trainSet.genColExtTrainSet(mode='all')
          # These magnitude bins must match the ones supplied when training the classifier.
          magBins = [(18.0, 22.0), (22.0, 24.0), (24.0, 25.0), (25.0, 26.0)]
          with open('<depth>ClfsColsExt.pkl', 'rb') as f:
              clfs = pickle.load(f)
          clfXd = dGauss.XDClfs(clfs=clfs, magBins=magBins)
          posteriors = clfXd.predict_proba(X, XErr, mags)

       Note that you must use a SGS classifier trained at the same depth as the set that you wish to classify at. For example, if your ``<classify>.csv`` file is at wide depth, you must use a classifier trained using wide depth data.


.. _EUPS: https://github.com/RobertLuptonTheGood/eups
.. _Extreme Deconvolution: https://github.com/jobovy/extreme-deconvolution
.. _HSC: http://www.naoj.org/Projects/HSC/
.. _Jose Garmilla: https://www.linkedin.com/in/jos%C3%A9-garmilla-a9b85530
.. _LSST Stack: https://pipelines.lsst.io/
.. _LSST analysis: https://github.com/lsst/analysis
.. _sgs-fsbutler: https://github.com/lsst-dm/sgs-fsbutler
.. _sgs: https://github.com/lsst-dm/sgs
.. _sgsupervised: https://github.com/lsst-dm/sgsupervised
.. [Garmilla2016] Garmilla, J. A. 2016. *Star/Galaxy Separation in Hyper Suprime-Cam
            and Mapping the Milky Way with Star Counts*
            `pqdtopen:10167596 <http://pqdtopen.proquest.com/doc/1831357728.html?FMT=ABS&pubnum=10167596>`_
.. [Leauthaud2007] Leauthaud, A. et al. ApJS, 172:219–238 2007. *Weak Gravitational Lensing
            with COSMOS: Galaxy Selection and Shape Measurements*
            `ADS:2007ApJS..172..219L <http://adsabs.harvard.edu/abs/2007ApJS..172..219L>`_
