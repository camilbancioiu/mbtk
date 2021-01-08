import pudb
import pickle
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



def read_bif_file(sourcepath, use_cache=True):
    bayesian_network = None

    # BIF files might be large, so we read them from source and then we pickle
    # them to files. If a pickle-file is found, read it instead of the
    # requested BIF file.
    cachefile = sourcepath.with_suffix('.pickle')
    if use_cache:
        if cachefile.exists():
            with cachefile.open('rb') as f:
                bayesian_network = pickle.load(f)
            return bayesian_network

    bayesian_network = parse_bif_file(sourcepath)
    with cachefile.open('wb') as f:
        pickle.dump(bayesian_network, f)
    return bayesian_network



def parse_bif_file(path):
    from lark import Lark
    from mbtk.utilities.bif.Grammar import bif_grammar
    from mbtk.utilities.bif.Transformers import get_transformer_chain
    parser = Lark(bif_grammar)
    tree = parser.parse(path.read_text())

    bayesian_network = get_transformer_chain().transform(tree)
    return bayesian_network



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
