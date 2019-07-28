import numpy
import scipy
import random

from mbff.dataset.sources.DatasetSource import DatasetSource
from mbff.dataset.DatasetMatrix import DatasetMatrix
import mbff.utilities.functions as util


class SampledBayesianNetworkDatasetSource(DatasetSource):
    """
    A dataset source which loads a specified Bayesian Network from a BIF file,
    then samples it a specified number of times.
    """

    def __init__(self, configuration, finalize_bn=True):
        self.configuration = configuration
        self.bayesian_network = util.read_bif_file(self.configuration['sourcepath'])
        if finalize_bn:
            self.bayesian_network.finalize()
        self.reset_random_seed = True


    def create_dataset_matrix(self, label='bayesian_network', other_random_seed=-1):
        if self.reset_random_seed or (other_random_seed != -1 and other_random_seed != self.configuration['random_seed']):
            if other_random_seed == -1:
                random.seed(self.configuration['random_seed'])
            else:
                random.seed(other_random_seed)
            self.reset_random_seed = False

        sample_count = self.configuration['sample_count']
        objective_names = sorted(self.configuration.get('objectives', []))
        feature_names = list(sorted(list(set(self.bayesian_network.variable_node_names()) - set(objective_names))))

        numpy_datatype = self.configuration.get('numpy_datatype', numpy.int8)

        sample_matrix = self.bayesian_network.sample_matrix(sample_count, dtype=numpy_datatype)

        X = numpy.empty((sample_count, 0), dtype=numpy_datatype)
        Y = numpy.empty((sample_count, 0), dtype=numpy_datatype)

        for varname in feature_names:
            varindex = self.bayesian_network.variable_nodes_index(varname)
            feature = sample_matrix[:, varindex][numpy.newaxis].T
            X = numpy.hstack((X, feature))

        for varname in objective_names:
            varindex = self.bayesian_network.variable_nodes_index(varname)
            objective = sample_matrix[:, varindex][numpy.newaxis].T
            Y = numpy.hstack((Y, objective))

        datasetmatrix = DatasetMatrix(label)
        datasetmatrix.X = scipy.sparse.csr_matrix(X)
        datasetmatrix.Y = scipy.sparse.csr_matrix(Y)
        datasetmatrix.row_labels = ['row{}'.format(i) for i in range(0, sample_count)]
        datasetmatrix.column_labels_X = feature_names
        datasetmatrix.column_labels_Y = objective_names
        datasetmatrix.metadata['source'] = self

        return datasetmatrix
