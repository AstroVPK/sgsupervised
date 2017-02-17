import dGauss
import pickle
import argparse

parser = argparse.ArgumentParser(description="Run cross-validation tests on XD models.")
parser.add_argument('--magMin', default=18.0, type=float,
                    help='Minimum magnitude to use for the tests.')
parser.add_argument('--magMax', default=22.0, type=float,
                    help='Maximum magnitude to use for the tests.')
parser.add_argument('--extMax', default=5.0, type=float,
                    help='Maximum extendedness to use for the tests.')
kargs = vars(parser.parse_args())
magMin = kargs['magMin']
magMax = kargs['magMax']
extMax = kargs['extMax']

with open('trainSet.pkl', 'rb') as f: trainSet = pickle.load(f)
dGauss.getCVParamsAndy(trainSet, magMin=magMin, magMax=magMax, extMax=extMax,
                       param_grid = {'ngStar':[5, 10, 15], 'ngGal':[5, 10, 15]}, nCV=10)
