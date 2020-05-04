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
        self.omega = None
        self.datasetmatrix = None
        self.bayesiannetwork = None



def make_test_dataset(configuration):
    ds = MockDataset()
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
