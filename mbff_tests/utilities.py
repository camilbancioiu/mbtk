import shutil
from pathlib import Path

from mbff.math.Variable import Omega
from mbff.dataset.DatasetMatrix import DatasetMatrix
from mbff.dataset.sources.SampledBayesianNetworkDatasetSource import SampledBayesianNetworkDatasetSource
import mbff.utilities.functions as util


test_folder = Path('mbff_tests', 'testfiles')
bif_folder = Path('mbff_tests', 'bif_files')
tmp_folder = Path(test_folder, 'tmp')


class MockDataset:

    def __init__(self):
        self.label = None
        self.omega = None
        self.datasetmatrix = None
        self.bayesiannetwork = None



def make_test_dataset(configuration):
    ds = MockDataset()
    ds.label = configuration['label']
    ds.omega = Omega(configuration['sample_count'])
    ds.datasetmatrix = make_test_datasetmatrix(configuration)
    ds.bayesiannetwork = make_test_bayesian_network(configuration)
    return ds



def make_test_datasetmatrix(configuration):
    folder = configuration['folder']
    label = configuration['label']
    try:
        datasetmatrix = DatasetMatrix(label)
        datasetmatrix.load(folder)
    except FileNotFoundError:
        sbnds = SampledBayesianNetworkDatasetSource(configuration)
        sbnds.reset_random_seed = True
        datasetmatrix = sbnds.create_dataset_matrix(label)
        datasetmatrix.finalize()
        datasetmatrix.save(folder)
    return datasetmatrix



def make_test_bayesian_network(configuration):
    bn = util.read_bif_file(configuration['sourcepath'])
    bn.finalize()
    return bn



def ensure_empty_tmp_subfolder(subfolder):
    try:
        shutil.rmtree(Path(tmp_folder, subfolder))
    except FileNotFoundError:
        pass
    path = Path(tmp_folder, subfolder)
    path.mkdir(parents=True, exist_ok=True)
    return path



def prepare_AD_tree(configuration, datasetmatrix):
    path = configuration['path']
    debug = configuration['debug']
    leaf_list_threshold = configuration['leaf_list_threshold']
    adtree = None
    if path.exists():
        with path.open('rb') as f:
            adtree = pickle.load(f)
        adtree.debug = debug
        if adtree.debug >= 1:
            adtree.debug_prepare__querying()
    else:
        matrix = datasetmatrix.X
        column_values = datasetmatrix.get_values_per_column('X')
        adtree = ADTree(matrix, column_values, leaf_list_threshold, debug)
        if path is not None:
            with path.open('wb') as f:
                pickle.dump(adtree, f)
    return adtree
