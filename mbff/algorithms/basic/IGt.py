import operator
import numpy
import math
from pprint import pprint
import mbff.math.infotheory as infotheory

def algorithm_IGt__binary(datasetmatrix, parameters):
    (sample_count, feature_count) = datasetmatrix.X.get_shape()
    Q = parameters['Q']
    objective_vector = datasetmatrix.get_column_Y(parameters['objective_index'])
    IG_per_feature = []

    for feature_index in range(feature_count):
        feature_vector = datasetmatrix.get_column_X(feature_index)
        feature_IG = MI__binary(feature_vector, objective_vector)
        IG_per_feature.append((feature_index, feature_IG))

    sorted_IG_per_feature = sorted(IG_per_feature, key=operator.itemgetter(1), reverse=True)
    selected_features = [pair[0] for pair in sorted_IG_per_feature[0:Q]]
    return selected_features



def MI__binary(X, Y):
    pX = calculate_pmf__binary(X)
    pY = calculate_pmf__binary(Y)
    pXY = calculate_joint_pmf2__binary(X, Y)

    def pmi(x, y):
        marginals = pX[x] * pY[y]
        joint = pXY[(x,y)]
        if joint == 0 or marginals == 0:
            return 0
        else:
            return joint * math.log2(joint / marginals)

    return pmi(0, 0) + pmi(0, 1) + pmi(1, 0) + pmi(1, 1)



def calculate_pmf__binary(X):
    size = len(X)
    p = {}
    p[1] = numpy.sum(X) / size
    p[0] = 1 - p[1]
    return p



def calculate_joint_pmf2__binary(X, Y):
    size = len(X)
    p = {}
    n_X = numpy.logical_not(X)
    n_Y = numpy.logical_not(Y)
    p[(0,0)] = numpy.sum(numpy.logical_and(n_X, n_Y)) / size
    p[(1,1)] = numpy.sum(numpy.logical_and(X, Y)) / size
    p[(0,1)] = numpy.sum(numpy.logical_and(n_X, Y)) / size
    p[(1,0)] = numpy.sum(numpy.logical_and(X, n_Y)) / size
    return p

