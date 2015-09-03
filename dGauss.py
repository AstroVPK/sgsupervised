from sklearn.svm import SVC, LinearSVC 
from sklearn.linear_model import LogisticRegression 
from sklearn.grid_search import GridSearchCV

def linearFit(trainingSet, n_jobs=4, **estKargs):
    X, Y = trainingSet.getTrainSet()
    estimator = LinearSVC(**estKargs)
    param_grid = {'C':[0.1, 1.0, 10.0]}
    clf = GridSearchCV(estimator, param_grid, n_jobs=n_jobs)
    clf.fit(X, Y)
    X, Y = trainingSet.getTestSet()
    score = clf.score(X, Y)
    print "score=", score
    return clf

def logisticFit(trainingSet, n_jobs=4, **estKargs):
    X, Y = trainingSet.getTrainSet()
    estimator = LogisticRegression(**estKargs)
    param_grid = {'C':[ 1.0e5, 2.0e5, 4.0e5, 1.0e6]}
    clf = GridSearchCV(estimator, param_grid, n_jobs=n_jobs)
    clf.fit(X, Y)
    X, Y = trainingSet.getTestSet()
    score = clf.score(X, Y)
    print "score=", score
    return clf

def rbfFit(trainingSet, n_jobs=4):
    X, Y = trainingSet.getTrainSet()
    estimator = SVC()
    param_grid = {'C':[0.1, 1.0, 10.0], 'gamma':[0.1, 1.0, 10.0]}
    clf = GridSearchCV(estimator, param_grid, n_jobs=n_jobs)
    clf.fit(X, Y)
    X, Y = trainingSet.getTestSet()
    score = clf.score(X, Y)
    print "score=", score
    return clf
