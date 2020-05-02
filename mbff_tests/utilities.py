from mbff.math.Variable import Omega
from mbff.dataset.DatasetMatrix import DatasetMatrix
from mbff.dataset.sources.SampledBayesianNetworkDatasetSource import SampledBayesianNetworkDatasetSource
import mbff.utilities.functions as util


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
    bayesian_network = util.read_bif_file(configuration['sourcepath'])
    bayesian_network.finalize()
    return bayesian_network
