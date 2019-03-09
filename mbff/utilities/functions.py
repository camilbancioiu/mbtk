import sys
import numpy
import scipy.io

from pathlib import Path



def ensure_folder(folder):
    path = Path(folder)
    path.mkdir(parents=True, exist_ok=True)



def load_matrix(path, matrix_name):
    matrix = None
    fname = str(path / "{0}.mtx".format(matrix_name))
    matrix = scipy.io.mmread(fname)
    if isinstance(matrix, numpy.ndarray):
        matrix = numpy.matrix(matrix)
    else:
        matrix = matrix.tocsr()
    return matrix



def save_matrix(path, matrix_name, matrix):
    fname = str(path / "{0}.mtx".format(matrix_name))
    scipy.io.mmwrite(fname, matrix)



def create_index(l):
    return dict(zip(l, range(0, len(l))))



