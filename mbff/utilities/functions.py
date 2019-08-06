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



def create_index(l):
    return dict(zip(l, range(0, len(l))))



def read_bif_file(path):
    from lark import Lark
    from mbff.utilities.bif.Grammar import bif_grammar
    from mbff.utilities.bif.Transformers import get_transformer_chain
    parser = Lark(bif_grammar)
    tree = parser.parse(path.read_text())

    bayesian_network_model = get_transformer_chain().transform(tree)
    return bayesian_network_model



def flatten(l, ltypes=(list, tuple)):
    ltype = type(l)
    l = list(l)
    i = 0
    while i < len(l):
        while isinstance(l[i], ltypes):
            if not l[i]:
                l.pop(i)
                i -= 1
                break
            else:
                l[i:i + 1] = l[i]
        i += 1
    return ltype(l)



def breakpoint():
    pudb.set_trace()
