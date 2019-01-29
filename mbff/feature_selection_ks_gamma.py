import math
import scipy.sparse
import scipy.stats
import scipy.io
import numpy
import collections
import time
import utilities as util
from experimental_dataset import ExperimentalDataset
from pathlib import Path
from mpprint import mpprint
import multiprocessing
from pprint import pprint
from utilities import H1, H2, H3
import sys

def load_ks_gamma_tables(folder, targets):
    folder += '/ks'
    gamma_tables = {}
    try:
        for i in targets:
            gamma_table = scipy.io.mmread(folder + '/ks-gamma-{0}.mtx'.format(i))
            gamma_tables[i] = gamma_table
    except:
        raise Exception("Gamma tables could not be read from the folder {}. Please make sure you have generated them using the command './exds build-ks-gamma [dataset]'.".format(folder))
    return gamma_tables

def build_ks_gamma_tables(dest_folder, exds, targets, optimization='full_sharing'):
    dest_folder += '/ks'
    path = Path(dest_folder)
    path.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    mpprint('Start time: {0}'.format(util.localftime(start_time)))

    output_file_name = dest_folder + '/ks-gamma-calculation.log'
    with open(output_file_name, 'wt') as f:
        sys.stdout = f
        if optimization == 'off':
            tables = construct_gamma_tables_unoptimized(dest_folder, exds.X, exds.Y, targets)
        else:
            tables = construct_gamma_tables_with_dict_Hcache_main(dest_folder, exds.X, exds.Y, targets, optimization)
        sys.stdout = sys.__stdout__
    end_time = time.time()
    mpprint('Start time: {0}'.format(util.localftime(start_time)))
    mpprint('End time: {0}'.format(util.localftime(end_time)))
    mpprint('Duration: {0}s'.format((end_time - start_time)))


### Gamma with matrix Hcache
def construct_gamma_tables_with_matrix_Hcache(folder, X, Y, Ts=[], optimization='full_sharing'):
    X = X.tocsc()
    Y = Y.tocsc()

    n = X.get_shape()[1]
    if len(Ts) == 0:
        Ts = range(Y.get_shape()[1])
    c = len(Ts)

    timer_complete = util.Timer('Complete gamma tables with Hcache')
    timer_feature = util.Timer('Gamma per feature')
    timer_target = util.Timer('Gamma per target')

    hXj = numpy.zeros(n, dtype=numpy.float32)
    complete_hXj = False

    hXiXj = numpy.zeros((n, n), dtype=numpy.float32)
    complete_hXiXj = False

    hYtXj = numpy.zeros(n, dtype=numpy.float32)
    hYtXiXj = numpy.zeros((n, n), dtype=numpy.float32)

    for t in range(c): 
        Yt = Y.getcol(t).toarray().ravel()
        hYtXiXj = numpy.zeros((n, n), dtype=numpy.float32)
        timer_target.reset()

        for j in range(n):
            timer_feature.reset()
            Xj = X.getcol(j).toarray().ravel()
            if not complete_hXj: 
                hXj[j] = H1(Xj)

            hYtXj[j] = H2(Yt, Xj)

            for i in range(j + 1, n):
                Xi = X.getcol(i).toarray().ravel()
                if not complete_hXiXj:
                    hXiXj[i, j] = H2(Xi, Xj)

                hYtXiXj[i, j] = H3(Yt, Xi, Xj) 
            timer_feature.check('Feature {} / {}, target {} / {}.' .format(j, n - 1, t, c - 1))

        if not complete_hXiXj:
          hXiXj = symmetrize_matrix(hXiXj)

        hYtXiXj = symmetrize_matrix(hYtXiXj)

        gamma_Yt = hYtXj - hXj - hYtXiXj + hXiXj
        numpy.fill_diagonal(gamma_Yt, -1)

        scipy.io.mmwrite(folder + '/ks-gamma-{0}.mtx'.format(t), gamma_Yt)
        complete_hXj = complete_hXiXj = True
        timer_target.check('Target {} / {}.'.format(t, len(Ts) - 1))
    timer_complete.check('Done.')

def symmetrize_matrix(a):
    return a + a.T - numpy.diag(a.diagonal())


### Gamma optimized with Hcache
def construct_gamma_tables_with_dict_Hcache(folder, X, Y, Ts=[], optimization='full_sharing'):
    X = X.tocsc()
    Y = Y.tocsc()

    n = X.get_shape()[1]
    if len(Ts) == 0:
        Ts = range(Y.get_shape()[1])

    cHYXj = {}
    cHXj = {}
    cHYXiXj = {}
    cHXiXj = {}

    timer_complete = util.Timer('Complete gamma tables with Hcache')
    timer_feature = util.Timer('Gamma per feature')
    timer_target = util.Timer('Gamma per target')

    for t in Ts:
        Yt = Y.getcol(t).toarray().ravel()
        gamma_Yt = scipy.sparse.dok_matrix((n, n), dtype=numpy.float32) 
        timer_target.reset()
        for i in range(n):
            #timer_feature.reset()
            Xi = X.getcol(i).toarray().ravel()
            for j in range(n):
                if i == j:
                    gamma_Yt[i, j] = -1
                else:
                    Xj = X.getcol(j).toarray().ravel()
                    # Calculate entropy terms or retrieve from caches.
                    try:
                        H_YXj = cHYXj[(t, j)]
                    except KeyError:
                        H_YXj = H2(Yt, Xj)
                        cHYXj[(t, j)] = H_YXj

                    try:
                        H_Xj = cHXj[j]
                    except KeyError:
                        H_Xj = H1(Xj)
                        cHXj[j] = H_Xj

                    ij = frozenset({i, j})
                    try:
                        H_YXiXj = cHYXiXj[(t, ij)]
                    except KeyError:
                        H_YXiXj = H3(Yt, Xi, Xj)
                        cHYXiXj[(t, ij)] = H_YXiXj

                    try:
                        H_XiXj = cHXiXj[ij]
                    except KeyError:
                        H_XiXj = H2(Xi, Xj)
                        cHXiXj[ij] = H_XiXj

                    # Calculate gamma from entropy terms.
                    gamma_ij = H_YXj - H_Xj - H_YXiXj + H_XiXj
                    # Discard rounding errors
                    gamma_Yt[i, j] = gamma_ij if gamma_ij > 1e-15 else 0
            #timer_feature.check('Feature {} / {}, target {} / {}.'.format(i, n - 1, t, len(Ts) - 1))
        timer_target.check('Target {} / {}.'.format(t, len(Ts) - 1))
        table = gamma_Yt.tocsc()
        scipy.io.mmwrite(folder + '/ks-gamma-{0}.mtx'.format(t), table)
    timer_complete.check('Done.')


### Gamma optimized with Hcache
def construct_gamma_tables_with_dict_Hcache_main(folder, X, Y, Ts=[], optimization='full_sharing'):
    X = X.tocsc()
    Y = Y.tocsc()

    n = X.get_shape()[1]
    if len(Ts) == 0:
        Ts = range(Y.get_shape()[1])
    c = len(Ts)

    timer_complete = util.Timer('Complete gamma tables with Hcache')
    timer_feature = util.Timer('Gamma per feature')
    timer_target = util.Timer('Gamma per target')

    hXj = {}
    complete_hXj = False

    hXiXj = {}
    complete_hXiXj = False

    hYtXj = {}
    hYtXiXj = {}

    for t in Ts:
        gamma_Yt = scipy.sparse.dok_matrix((n, n), dtype=numpy.float32) 

        Yt = Y.getcol(t).toarray().ravel()
        hYtXiXj = {}
        timer_target.reset()

        # Definition: Gamma(Y, Xi, Xj) = H(Y, Xj) - H(Xj) - H(Y, Xi, Xj) + H(Xi, Xj)

        for j in range(n):
            timer_feature.reset()

            Xj = X.getcol(j).toarray().ravel()
            if not complete_hXj: 
                hXj[j] = H1(Xj)

            hYtXj[j] = H2(Yt, Xj)

            for i in range(j + 1, n):
                Xi = X.getcol(i).toarray().ravel()
                ij = frozenset({i, j})
                if not complete_hXiXj:
                    hXiXj[ij] = H2(Xi, Xj)

                hYtXiXj[ij] = H3(Yt, Xi, Xj) 
                gamma_ij = hYtXj[j] - hXj[j] - hYtXiXj[ij] + hXiXj[ij]
                # Discard rounding errors
                gamma_Yt[i, j] = gamma_ij if gamma_ij > 1e-15 else 0

            for i in range(j):
                ij = frozenset({i, j})
                gamma_ij = hYtXj[j] - hXj[j] - hYtXiXj[ij] + hXiXj[ij]
                # Discard rounding errors
                gamma_Yt[i, j] = gamma_ij if gamma_ij > 1e-15 else 0

            gamma_Yt[j, j] = -1

            timer_feature.check('Feature {} / {}, target {} / {}.'.format(j,n-1,t,c-1))
        complete_hXj = complete_hXiXj = True
        timer_target.check('Target {} / {}.'.format(t, c - 1))
        table = gamma_Yt.tocsc()
        scipy.io.mmwrite(folder + '/ks-gamma-{0}.mtx'.format(t), table)
    timer_complete.check('Done.')


### Unoptimized gamma

def construct_gamma_tables_unoptimized(folder, X, Y, Ts=[]):
    X = X.tocsc()
    Y = Y.tocsc()

    Xcols = X.get_shape()[1]
    Ycols = Y.get_shape()[1]
    Features = X.get_shape()[1]

    timer_gamma_per_feature = util.Timer('Gamma per feature')
    timer_gamma_per_target = util.Timer('Gamma per target')
    if len(Ts) == 0:
        Ts = range(Ycols)
    for t in Ts:
        timer_gamma_per_target.reset()
        Yt = Y.getcol(t).toarray().ravel()
        gamma_Yt = scipy.sparse.dok_matrix((Xcols, Xcols), dtype=numpy.float32) 
        for i in range(Xcols):
            timer_gamma_per_feature.reset()
            Xi = X.getcol(i).toarray().ravel()
            for j in range(Xcols):
                if i == j:
                    gamma_Yt[i, j] = -1
                else:
                    Xj = X.getcol(j).toarray().ravel()
                    gamma_ij = gamma(Yt, Xi, Xj, (t, i, j))
                    # Discard rounding errors
                    gamma_Yt[i, j] = gamma_ij if gamma_ij > 1e-15 else 0
            timer_gamma_per_feature.check('Feature {} / {}, target {} / {}.'.format(
                i, Features - 1, t, len(Ts) - 1))
        timer_gamma_per_target.check('Target {} / {}.'.format(t, len(Ts) - 1))
        table = gamma_Yt.tocsc()
        scipy.io.mmwrite(folder + '/ks-gamma-{0}.mtx'.format(t), table)

def gamma(Y, Xi, Xj, index):
    (t, i, j) = index
    p3yxx = util.calculate_joint_pmf3(Y, Xi, Xj)
    p2xx = util.calculate_joint_pmf2(Xi, Xj)
    p2yx = util.calculate_joint_pmf2(Y, Xj)
    p1x = util.calculate_pmf(Xj)

    instances = p3yxx.keys()

    G = 0.0
    for (y, xi, xj) in instances:
        ppn = p2xx.get((xi, xj), 0.0)
        if ppn == 0:
            continue
        ppf = p3yxx.get((y, xi, xj), 0.0)
        pp = ppf / ppn
        pq = p2yx.get((y, xj), 0.0) / p1x.get(xj, 0.0)
        if pp == 0 or pq == 0:
            continue
        else:
            G += ppf * math.log2(pp / pq)

    return G
