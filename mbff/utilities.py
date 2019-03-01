import sys
import numpy
import scipy.io

from pathlib import Path



def ensure_folder(folder):
    path = Path(folder)
    path.mkdir(parents=True, exist_ok=True)



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



def create_index(l):
    return dict(zip(l, range(0, len(l))))



class MultiFileWriter:

    def __init__(self, files):
        self.files = files


    def write(self, string):
        for f in self.files:
            f.write(string)


    def flush(self):
        for f in self.files:
            f.flush()


    def close(self):
        for f in self.files():
            if not f is sys.__stdout__:
                f.close()
