import time
import sys
import math
import operator
import numpy
import scipy
import scipy.sparse
import random

import utilities as util

from feature_selection_ks import ks, ks_init
from feature_selection_ks_gamma import load_ks_gamma_tables, build_ks_gamma_tables

def get_algorithm(alg_key):
    if alg_key == 'NL':
        return all_features
    if alg_key == 'RND':
        return random_features
    if alg_key == 'IG':
        return ig_thresholding
    if alg_key == 'KS':
        return ks_algorithm

def prepare(experiment_definition):
    ks_init(experiment_definition)
    print('Feature Selection algorithms initialized.')

## Basic FS algorithms

def random_features(dataset, parameters):
    X = dataset['X']
    Q = parameters['Q']
    random.seed(Q)
    return random.sample(range(X.get_shape()[1]), Q)

def all_features(dataset, parameters):
    X = dataset['X']
    return list(range(X.get_shape()[1]))


## Information Gain Thresholding

def ig_thresholding(dataset, parameters):
    X = dataset['X']
    Y = dataset['Y']
    Q = parameters['Q']
    total_cols = X.get_shape()[1]
    X = X.tocsc()
    ig = []
    for i in range(total_cols):
        Xi = X.getcol(i).toarray().ravel()
        ig.append((i, util.IG(Xi, Y)))

    sorted_ig = sorted(ig, key=operator.itemgetter(1), reverse=True)
    return [i[0] for i in sorted_ig[0:Q]]

## Koller and Sahami's algorithm

def ks_algorithm(dataset, parameters):
    X = dataset['X']
    Y = dataset['Y']
    Q = parameters['Q']
    K = parameters['K']
    target = parameters['target']
    return ks(X, Y, target, Q, K)
    

## Tests

def test_joint_pmfs():
    X = numpy.array([0, 0, 0, 0, 1])
    Y = numpy.array([0, 0, 0, 1, 1])
    Z = numpy.array([1, 0, 0, 1, 0])
    T = numpy.array([1, 1, 1, 1, 1])

    full_joint = util.calculate_joint_pmf([X, Y, Z, T])
    assert full_joint([0,0,0,1]) == (2 / 5), 'pmf 1'
    assert full_joint([0,0,0,0]) == (0), 'pmf 2'
    assert full_joint([1,1,0,1]) == (1 / 5), 'pmf 3'
       
def test_cond_pmfs():
    X = numpy.array([0, 0, 0, 0, 1])
    Y = numpy.array([0, 0, 0, 1, 1])
    Z = numpy.array([1, 0, 0, 1, 0])
    T = numpy.array([1, 1, 1, 1, 1])

    cond_pmf = util.calculate_cond_pmf(Z, [T])
    assert cond_pmf(1, [1]) == (2/5), 'cpmf 1'
    assert cond_pmf(0, [1]) == (3/5), 'cpmf 2'
    assert cond_pmf(0, [0]) == (0), 'cpmf 3'
    assert cond_pmf(1, [0]) == (0), 'cpmf 4'

    cond_pmf = util.calculate_cond_pmf(Z, [X, Y])
    assert cond_pmf(0, [0, 0]) == ((2/5) / (3/5)), 'cpmf 5'
    assert cond_pmf(1, [0, 0]) == ((1/5) / (3/5)), 'cpmf 6'

def compare_dicts(a, b):
    assert set(a.keys()) == set(b.keys())
    for k in a.keys():
        assert a[k] == b[k], 'key: {0}, val_a: {1}, val_b{1}'.format(k, a[k], b[k])
