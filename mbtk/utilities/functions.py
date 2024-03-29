import pudb
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



def create_index(lst):
    return dict(zip(lst, range(0, len(lst))))



def flatten(lst, ltypes=(list, tuple)):
    ltype = type(lst)
    lst = list(lst)
    i = 0
    while i < len(lst):
        while isinstance(lst[i], ltypes):
            if not lst[i]:
                lst.pop(i)
                i -= 1
                break
            else:
                lst[i:i + 1] = lst[i]
        i += 1
    return ltype(lst)



def breakpoint():
    pudb.set_trace()
