import pickle
import shutil
from pathlib import Path

from mbff.math.Variable import Omega
from mbff.dataset.DatasetMatrix import DatasetMatrix
from mbff.dataset.sources.SampledBayesianNetworkDatasetSource import SampledBayesianNetworkDatasetSource
import mbff.utilities.functions as util


test_folder = Path('tests', 'testfiles')
bif_folder = Path('tests', 'bif_files')
tmp_folder = Path(test_folder, 'tmp')


class MockDataset:

    def __init__(self, configuration):
        self.label = configuration['label']
        self.omega = Omega(configuration['sample_count'])
        self.datasetmatrix = make_test_datasetmatrix(configuration)
        self.bayesiannetwork = make_test_bayesian_network(configuration)



def make_test_datasetmatrix(configuration):
    folder = tmp_folder / 'mockdataset'
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



def ensure_tmp_subfolder(subfolder):
    path = tmp_folder / subfolder
    path.mkdir(parents=True, exist_ok=True)
    return path



def ensure_empty_tmp_subfolder(subfolder):
    try:
        shutil.rmtree(Path(tmp_folder, subfolder))
    except FileNotFoundError:
        pass
    path = tmp_folder / subfolder
    path.mkdir(parents=True, exist_ok=True)
    return path



def prepare_AD_tree(configuration, datasetmatrix):
    adtrees_folder = tmp_folder / 'adtrees'
    adtrees_folder.mkdir(parents=True, exist_ok=True)
    path = adtrees_folder / (configuration['label'] + '.pickle')
    debug = configuration.get('debug', 0)
    leaf_list_threshold = configuration['leaf_list_threshold']
    adtree = None
    if path.exists():
        with path.open('rb') as f:
            adtree = pickle.load(f)
        if debug >= 1:
            adtree.debug = debug
            adtree.debug_prepare__querying()
    else:
        matrix = datasetmatrix.X
        column_values = datasetmatrix.get_values_per_column('X')
        adtree_class = configuration['ci_test_ad_tree_class']
        try:
            adtree = adtree_class(matrix, column_values, leaf_list_threshold, debug)
        except TypeError:
            adtree = adtree_class(matrix, column_values, leaf_list_threshold)

        if path is not None:
            with path.open('wb') as f:
                pickle.dump(adtree, f)
    return adtree
