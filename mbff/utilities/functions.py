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



def read_bif_file(path):
    from lark import Lark
    from mbff.utilities.bif.Grammar import bif_grammar
    from mbff.utilities.bif.Transformers import get_transformer_chain
    parser = Lark(bif_grammar)
    tree = parser.parse(path.read_text())

    bayesian_network_model = get_transformer_chain().transform(tree)
    return bayesian_network_model



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
