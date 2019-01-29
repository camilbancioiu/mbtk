import operator
import numpy
import scipy
import scipy.sparse
import utilities as util
import pickle
from multiprocessing import Pool
import time
import math
import sys
from pathlib import Path

import feature_selection_ks_iteration_cache as ks_iteration_cache
import feature_selection_ks_fdb as ks_fdb
from feature_selection_ks_gamma import load_ks_gamma_tables


FULL_USE = 1
COMPARE_ONLY = 2
DISABLED = 4

gamma_tables = None
use_ksic = FULL_USE
use_ksfdb = FULL_USE
parallelism = 1
experiment = None

## Definition utility


## Initialization

def set_parallelism(p):
    global parallelism
    parallelism = p

def set_gamma_tables(tables):
    global gamma_tables
    gamma_tables = tables

# Caches: 
# F = build and use feature cache
# f = only read feature cache to compare computation, don't actually use; never write
# I = build and use iteration cache
# i = build and only read iteration cache to compare with computation, don't actually use
def ks_init(experiment_definition):
    global experiment 
    global use_ksic
    global use_ksfdb
    experiment = experiment_definition
    if 'KS' in experiment_definition.parameters['algorithm']:
        targets = experiment_definition.parameters['target']
        exds_folder = experiment_definition.exds_definition.folder
        parallelism = experiment_definition.config['ks_parallelism']


        set_parallelism(parallelism)
        saved_gamma_tables = load_ks_gamma_tables(exds_folder, targets) 
        set_gamma_tables(saved_gamma_tables)

        use_ksic = experiment_definition.config.get('ks_iteration_cache', DISABLED)
        use_ksfdb = experiment_definition.config.get('ks_feature_db', DISABLED)

        if use_ksfdb != DISABLED:
            ksfdb = ks_fdb.get_ks_feature_db()
            ksfdb.set_filename(exds_folder + '/ks/ks_fdb.pickle')
        print('KS Feature Selection prepared.')


## Koller and Sahami's algorithm

def ks(X, Y, Tj, Q, K):
    global gamma_tables
    global parallelism
    ksfdb = ks_fdb.get_ks_feature_db()
    ksfdb.load()

    if not ("ks_debug" in experiment.config):
        debug = False
    else:
        debug = experiment.config["ks_debug"]

    if use_ksic != DISABLED:
        path = Path(experiment.folder + '/ks')
        path.mkdir(parents=True, exist_ok=True)
        ksic = ks_iteration_cache.get_ks_iteration_cache()
        ksic.set_stats_filename(experiment.folder + '/ks/ks_ic_stats.pickle')
        ksic.reset()
    if use_ksfdb == COMPARE_ONLY:
        path = Path(experiment.folder + '/ks')
        path.mkdir(parents=True, exist_ok=True)
        ksfdb_filename = experiment.folder + '/ks/ks_compare_fdb.pickle'
        compare_fdb = ks_fdb.KSFeatureDatabase()
        compare_fdb.set_filename(ksfdb_filename)
        compare_fdb.load()
        compare_fdb.save()
    total_cols = X.get_shape()[1]
    if debug: print("Starting KS algorithm, Q = {}, K = {}".format(Q, K))
    if debug: print("KS caching: feature {}, iteration {}".format(use_ksfdb, use_ksic))
    X = X.tocsc()
    G = list(range(total_cols))
    gamma_table = gamma_tables[Tj].todense()
    for q in range(total_cols - Q):
        if use_ksic != DISABLED:
          ksic.set_iteration_key((Tj, q, K))
        if debug: print('KS iteration {} starting...'.format(q))
        start_time = time.time()
        blankets = {}

        if use_ksfdb == FULL_USE:
            try:
                feature_in_fdb = ksfdb.get((Tj, q, K))
                feature_to_remove = feature_in_fdb
                if debug: print('Feature to remove {} found in feature database at key {}' .format(feature_to_remove, (Tj, q, K)))
            except KeyError:
                blankets = get_candidate_Markov_blankets_parallel(X, Y, G, K, gamma_table)
                feature_to_remove = find_strongest_MB(blankets)
                if use_ksic != DISABLED:
                    ksic.remove_feature(feature_to_remove)
                feature_removal_duration = (time.time() - start_time)
                feature_metadata = {'parallelism' : parallelism, 'ksic': use_ksic, 'duration' : feature_removal_duration}
                ksfdb.store((Tj, q, K), feature_to_remove, feature_metadata)
        elif use_ksfdb == DISABLED:
            blankets = get_candidate_Markov_blankets_parallel(X, Y, G, K, gamma_table)
            feature_to_remove = find_strongest_MB(blankets)
            if use_ksic != DISABLED:
                ksic.remove_feature(feature_to_remove)
        elif use_ksfdb == COMPARE_ONLY:
            try:
                feature_in_fdb = ksfdb.get((Tj, q, K))
            except KeyError:
                raise Exception('Feature with key {} not found in the feature database.'.format((Tj, q, K)))
            blankets = get_candidate_Markov_blankets_parallel(X, Y, G, K, gamma_table)
            feature_to_remove = find_strongest_MB(blankets)
            if use_ksic != DISABLED:
                ksic.remove_feature(feature_to_remove)
            feature_removal_duration = (time.time() - start_time)
            feature_metadata = {'parallelism' : parallelism, 'ksic': use_ksic, 'duration' : feature_removal_duration}
            compare_fdb.store((Tj, q, K), feature_to_remove, feature_metadata)
            fdb_accurate = (feature_in_fdb == feature_to_remove)
            if debug: print('Feature to remove: computed {}, in FDB {}, key {}, accurate {}' .format(feature_to_remove, feature_in_fdb, (Tj, q, K), fdb_accurate))
            if not fdb_accurate:
                print('='*80)
                print('WARNING')
                print('Inaccurate feature: Tj {}, q {}, K {}'.format(Tj, q, K))
                print('='*80)
                #raise Exception('Inaccurate feature: Tj {}, q {}, K {}'.format(Tj, q, K))

        G.remove(feature_to_remove)

        if debug: print('KS iteration time: {0}ms'.format((time.time() - start_time)*1000))
        if debug: print('KS iteration {0} out of {1} (Q = {4}, K = {3}): removed feature {2}'.format(q, total_cols - Q - 1, feature_to_remove, K, Q))
        if debug: print('...........................')
        if debug: sys.stdout.flush()
    if use_ksic != DISABLED:
      ksic.save_stats()
    return G

def find_strongest_MB(blankets):
    return min(blankets, key=blankets.get)



## Building candidate Markov blankets

def get_candidate_Markov_blankets_parallel(X, Y, G, K, gamma_table):
    global use_ksic
    if use_ksic != DISABLED:
        ksic = ks_iteration_cache.get_ks_iteration_cache()
    else:
        ksic = None
    pool_args = [(X, Y, G, i, K, gamma_table, ksic) for i in G]
    if parallelism == 0:
        candidates = list(map(get_candidate_MB, pool_args))
    else:
        with Pool(parallelism) as pool:
            candidates = list(pool.map(get_candidate_MB, pool_args))

    if use_ksic != DISABLED:
        ksic.update(candidates)

    blankets = dict([(c[0], c[1]) for c in candidates])

    return blankets

def get_candidate_MB(pool_arg):
    global use_ksic
    (X, Y, G, i, K, gamma_table, ksic) = pool_arg

    if use_ksic == DISABLED:
        Xi = X.getcol(i).toarray().ravel()
        M_indices = get_candidate_MB_indices(i, G, K, gamma_table)
        try:
            M_indices.remove(i)
        except:
            pass
        M = lookup_feature_variables(X, M_indices)
        mb_delta = delta(Y, Xi, M)
        if i in M_indices:
            raise Exception("Feature {} part of its own CMB {}".format(i, M_indices))
        return (i, mb_delta, M_indices, None, None, None)

    if use_ksic == FULL_USE:
        try:
            M_indices = ksic.get_cmb(i)
            mb_delta = ksic.get_delta(i)
            hit = True
        except KeyError:
            Xi = X.getcol(i).toarray().ravel()
            M_indices = get_candidate_MB_indices(i, G, K, gamma_table)
            try:
                M_indices.remove(i)
            except:
                pass
            M = lookup_feature_variables(X, M_indices)
            mb_delta = delta(Y, Xi, M)
            hit = False
        if i in M_indices:
            raise Exception("Feature {} part of its own CMB {}".format(i, M_indices))
        return (i, mb_delta, M_indices, mb_delta, M_indices, hit)

    if use_ksic == COMPARE_ONLY:
        Xi = X.getcol(i).toarray().ravel()
        M_indices = get_candidate_MB_indices(i, G, K, gamma_table)
        try:
            M_indices.remove(i)
        except:
            pass
        M = lookup_feature_variables(X, M_indices)
        mb_delta = delta(Y, Xi, M)
        try:
            cached_M_indices = []
            cached_mb_delta = 0
            cached_M_indices = ksic.get_cmb(i)
            cached_mb_delta = ksic.get_delta(i)
            hit = True
        except KeyError:
            hit = False
        if i in M_indices:
            raise Exception("Feature {} part of its own CMB {}".format(i, M_indices))
        return (i, mb_delta, M_indices, cached_mb_delta, cached_M_indices, hit)

def get_candidate_MB_indices(i, G, K, gamma_table):
    gammas = [(j, gamma_table[i, j]) if i != j else (j, math.inf) for j in G]
    sorted_gammas = [j[0] for j in sorted(gammas, key=operator.itemgetter(1))]
    return sorted_gammas[0:K]

def lookup_feature_variables(X, indices):
    return [X.getcol(i).toarray().ravel() for i in indices]

def delta(Y, X, M):
    (pm, pmx, pmy, pmxy) = util.calculate_joint_pmf_MXY(M, X, Y)

    def pdelta(mi, x, y, mx):
        mxy = pmxy.get(mi + str(x) + str(y), 0.0)
        if mxy == 0.0:
            return 0.0
        my = pmy.get(mi + str(y), 0.0)
        if my == 0.0:
            return 0.0
        m = pm.get(mi, 0.0)
        if m == 0.0:
            return 0
        pd = ((mxy / mx) * math.log2((mxy / mx) / (my / m)))
        return pd

    D = 0.0
    for mi in pm.keys():
        for x in [0, 1]:
            mx = pmx.get(mi + str(x), 0.0)
            if mx == 0:
                continue
            for y in [0, 1]:
                D += mx * pdelta(mi, x, y, mx)
    return abs(D)

