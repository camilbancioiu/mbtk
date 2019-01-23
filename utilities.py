import math
import collections
import numpy
import scipy
import time
import datetime
import scipy.io
from mpprint import mpprint
from pathlib import Path
import sys

autoListFolders = []

def ensure_folder(folder):
    path = Path(folder)
    path.mkdir(parents=True, exist_ok=True)

def consume(iterator):
    collections.deque(iterator, maxlen=0)

def create_ordered_dict(key_order, unordered_dict):
    od = collections.OrderedDict()
    for key in key_order:
        od[key] = unordered_dict[key]
    return od

def load_list_from_file(fileprefix, fileid='', folder='.'):
    if fileprefix in autoListFolders:
        folder = './' + fileprefix
    if fileid == '':
        fname = '{}/{}.txt'.format(folder, fileprefix)
    else:
        fname = '{}/{}_{}.txt'.format(folder, fileprefix, fileid)
    with open(fname, mode='rt') as f:
        loaded_list = list(f)
    loaded_list = map(str.strip, loaded_list)
    return loaded_list

def save_list_to_file(list_to_save, fileprefix, fileid='', folder='.'):
    if fileprefix in autoListFolders:
        folder = './' + fileprefix

    path = Path('./' + folder)
    path.mkdir(parents=True, exist_ok=True)

    if fileid == '':
        fname = '{}/{}.txt'.format(folder, fileprefix)
    else:
        fname = '{}/{}_{}.txt'.format(folder, fileprefix, fileid)
    with open(fname, mode='wt') as f:
        f.write('\n'.join(list_to_save))

def load_matrix(folder, matrix_name):
    matrix = None
    fname = folder + "/{0}.mtx".format(matrix_name)
    try:
        matrix = scipy.io.mmread(fname)
    except FileNotFoundError:
        matrix = None
        mpprint('Matrix {0} could not be loaded from file {1} .'.format(matrix_name, fname))
        return matrix
    if isinstance(matrix, numpy.ndarray):
        matrix = numpy.matrix(matrix)
    else:
        try:
            matrix = matrix.tocsr()
        except AttributeError:
            pass
    return matrix

def save_matrix(folder, matrix_name, matrix):
    fname = folder + "/{0}.mtx".format(matrix_name)
    scipy.io.mmwrite(fname, matrix)

def keep_matrix_columns(X, cols):
    return X.tocsc()[:, cols].tocsr()

def keep_matrix_rows(X, rows):
    return X[rows, :]

def get_classifier_evaluation(validation, classifier):
    n_validation = numpy.logical_not(validation)
    n_classifier = numpy.logical_not(classifier)
    evaluation = collections.OrderedDict()
    evaluation['TP'] = numpy.asscalar(numpy.sum(numpy.logical_and(validation, classifier)))
    evaluation['TN'] = numpy.asscalar(numpy.sum(numpy.logical_and(n_validation, n_classifier)))
    evaluation['FP'] = numpy.asscalar(numpy.sum(numpy.logical_and(n_validation, classifier)))
    evaluation['FN'] = numpy.asscalar(numpy.sum(numpy.logical_and(validation, n_classifier)))
    return evaluation

def convert_to_csr(tuples):
    indptr = [0]
    indices = []
    data = []
    for t in tuples:
        indices += t[0]
        data += t[1]
        indptr.append(len(data))
    return scipy.sparse.csr_matrix((data, indices, indptr), dtype=numpy.int32)

def binarize_csr(csr):
    binary_data = numpy.ones(csr.data.size)
    return scipy.sparse.csr_matrix((binary_data, csr.indices, csr.indptr), dtype=numpy.int32)

def create_index(l):
    return dict(zip(l, range(0, len(l))))

def create_lookup(l):
    return dict(zip(range(0, len(l)), l))

def stopwatch(f):
    def wrapper(*args):
        start_time = time.time()
        f_return = f(*args)
        end_time = time.time()
        mpprint('Function {0} run duration: {1}ms'.format(f.__name__, (end_time-start_time)*1000.0))
        return f_return
    return wrapper

def calculate_pmf(X):
    size = len(X)
    p = {}
    p[0] = numpy.sum(numpy.logical_not(X)) / size
    p[1] = numpy.sum(X) / size
    return p

def calculate_joint_pmf2(X, Y):
    size = len(X)
    p = {}
    n_X = numpy.logical_not(X)
    n_Y = numpy.logical_not(Y)
    p[(0,0)] = numpy.sum(numpy.logical_and(n_X, n_Y)) / size
    p[(1,1)] = numpy.sum(numpy.logical_and(X, Y)) / size
    p[(0,1)] = numpy.sum(numpy.logical_and(n_X, Y)) / size
    p[(1,0)] = numpy.sum(numpy.logical_and(X, n_Y)) / size
    return p

def calculate_joint_pmf3x(X, Y, Z):
    size = len(X)
    p = {}
    nX = numpy.logical_not(X)
    nY = numpy.logical_not(Y)
    nZ = numpy.logical_not(Z)
    XY = numpy.logical_and(X, Y)
    XnY = numpy.logical_and(X, nY)
    nXY = numpy.logical_and(nX, Y)
    nXnY = numpy.logical_and(nX, nY)
    p[(0,0,0)] = numpy.sum(numpy.logical_and(nXnY, nZ)) / size
    p[(0,0,1)] = numpy.sum(numpy.logical_and(nXnY,  Z)) / size
    p[(0,1,0)] = numpy.sum(numpy.logical_and(nXY,  nZ)) / size
    p[(0,1,1)] = numpy.sum(numpy.logical_and(nXY,   Z)) / size
    p[(1,0,0)] = numpy.sum(numpy.logical_and(XnY,  nZ)) / size
    p[(1,0,1)] = numpy.sum(numpy.logical_and(XnY,   Z)) / size
    p[(1,1,0)] = numpy.sum(numpy.logical_and(XY,   nZ)) / size
    p[(1,1,1)] = numpy.sum(numpy.logical_and(XY,    Z)) / size
    return p

def H1(X):
  pmf = calculate_pmf(X)
  probabilities = numpy.array(list(pmf.values()))
  return scipy.stats.entropy(probabilities, base=2)

def H2(X, Y):
  pmf = calculate_joint_pmf2(X, Y)
  probabilities = numpy.array(list(pmf.values()))
  return scipy.stats.entropy(probabilities, base=2)

def H3(X, Y, Z):
  pmf = calculate_joint_pmf3(X, Y, Z)
  probabilities = numpy.array(list(pmf.values()))
  return scipy.stats.entropy(probabilities, base=2)

def test_pmf3x():
    x = numpy.array([0,0,0,0,1,1,1,1])
    y = numpy.array([0,0,1,1,0,0,1,1])
    z = numpy.array([0,1,0,1,0,1,0,1])
    p = calculate_joint_pmf3x(x, y, z)
    assert(p[(0,0,0)] == 1/8)
    assert(p[(0,0,1)] == 1/8)
    assert(p[(0,1,0)] == 1/8)
    assert(p[(0,1,1)] == 1/8)
    assert(p[(1,0,0)] == 1/8)
    assert(p[(1,0,1)] == 1/8)
    assert(p[(1,1,0)] == 1/8)
    assert(p[(1,1,1)] == 1/8)

    x = numpy.array([1,0,0,0,1,1,1,1])
    y = numpy.array([0,0,1,1,0,0,1,1])
    z = numpy.array([0,1,0,1,0,1,0,1])
    p = calculate_joint_pmf3x(x, y, z)
    assert(p[(0,0,0)] == 0/8)
    assert(p[(0,0,1)] == 1/8)
    assert(p[(0,1,0)] == 1/8)
    assert(p[(0,1,1)] == 1/8)
    assert(p[(1,0,0)] == 2/8)
    assert(p[(1,0,1)] == 1/8)
    assert(p[(1,1,0)] == 1/8)
    assert(p[(1,1,1)] == 1/8)

def calculate_joint_pmf3(X, Y, Z):
    size = len(X)
    p = {}
    for i in range(size):
        t = (X[i], Y[i], Z[i])
        try:
            p[t] += 1
        except KeyError:
            p[t] = 1

    for t in p.keys():
        p[t] /= size

    return p


def calculate_joint_pmf_MXY(M, X, Y):
    M_index = create_pmf_key_index(M)
    size = len(M_index)
    pmf_m = collections.Counter(M_index)
    for m_i in pmf_m:
        pmf_m[m_i] /= size

    MX_index = [M_index[i] + str(X[i]) for i in range(size)]
    pmf_mx = collections.Counter(MX_index)
    for mx_i in pmf_mx:
        pmf_mx[mx_i] /= size

    MY_index = [M_index[i] + str(Y[i]) for i in range(size)]
    pmf_my = collections.Counter(MY_index)
    for my_i in pmf_my:
        pmf_my[my_i] /= size

    MXY_index = [M_index[i] + str(X[i]) + str(Y[i]) for i in range(size)]
    pmf_mxy = collections.Counter(MXY_index)
    for mxy_i in pmf_mxy:
        pmf_mxy[mxy_i] /= size

    return (pmf_m, pmf_mx, pmf_my, pmf_mxy)

def create_pmf_key_index(variables):
    size = len(variables[0])
    keys = [''.join([str(v[i]) for v in variables]) for i in range(size)] 
    return keys
                
def calculate_joint_pmf(variables):
    pmf = {}
    size = len(variables[0])
    for i in range(size):
        instance = [v[i] for v in variables]
        instance_key = ''.join(map(str, instance))
        try:
            pmf[instance_key] += 1
        except KeyError:
            pmf[instance_key] = 1

    for instance_key in pmf.keys():
        pmf[instance_key] /= size

    return pmf

def H(X):
    p = calculate_pmf(X)
    return scipy.stats.entropy(list(p.values()), base=2)

def IG(X, Y):
    pX = calculate_pmf(X)
    pY = calculate_pmf(Y)
    pXY = calculate_joint_pmf2(X, Y)
    
    def pmi(x, y):
        marginals = pX[x] * pY[y]
        joint = pXY[(x,y)]
        if joint == 0 or marginals == 0:
            return 0
        try:
            return joint * math.log2(joint / marginals)
        except ValueError:
            mpprint("x: {0}, y: {1}, marginals: {2}, joint: {3}".format(x, y, marginals, joint))
            raise

    return pmi(0, 0) + pmi(0, 1) + pmi(1, 0) + pmi(1, 1)


start_time = 0.0
def start_timer():
    global start_time
    start_time = time.time()

def check_timer(msg=''):
    global start_time
    t = time.time() - start_time
    mpprint('{0} Time: {1}ms'.format(msg, t*1000.0))

def localftime(sec):
    return time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.localtime(sec))


class Timer():
    def __init__(self, name):
        self.start_time = time.time()
        self.name = name

    def reset(self):
        self.start_time = time.time()

    def check(self, message=''):
        elapsed = (time.time() - self.start_time)
        dt_elapsed = datetime.timedelta(seconds=elapsed)
        if len(message) > 0:
            print('Timer "{}", elapsed {} ({:<20}s): {}'.format(
                self.name, dt_elapsed, elapsed, message))
        else:
            print('Timer "{}", elapsed {} ({:<20}s)'.format(
                self.name, dt_elapsed, elapsed))
        sys.stdout.flush()







### Utilities to operate iteration keys

def list_iteration_keys_Tj(keys):
    # Iteration key format (Tj, q, K)
    return list(set([key[0] for key in keys]))

def list_iteration_keys_K(keys):
    # Iteration key format (Tj, q, K)
    return list(set([key[2] for key in keys]))

def list_iteration_keys(keys, Tj, K):
    selected_keys = []
    for key in keys:
        if key[0] == Tj and key[2] == K:
            selected_keys.append(key)
    return sorted(selected_keys)

