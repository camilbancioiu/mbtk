import math
import numpy



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

