import numpy
import scipy
import random

from mbtk.structures.BayesianNetwork import BayesianNetwork
from mbtk.dataset.sources.DatasetSource import DatasetSource
from mbtk.dataset.DatasetMatrix import DatasetMatrix


class SampledBayesianNetworkDatasetSource(DatasetSource):
    """
    A dataset source which loads a specified Bayesian Network from a BIF file,
    then samples it a specified number of times.
    """

    def __init__(self, configuration, finalize_bn=True):
        self.configuration = configuration
        path = self.configuration['sourcepath']
        self.bayesian_network = BayesianNetwork.from_bif_file(path, use_cache=False)
        if finalize_bn:
            self.bayesian_network.finalize()
        self.reset_random_seed = True


    def create_dataset_matrix(self, label='bayesian_network', other_random_seed=-1):
        method = self.configuration.get('method', 'random')
        if method == 'random':
            instances_matrix = self.create_random_instances(label, other_random_seed)
        elif method == 'exact':
            instances_matrix = self.create_exact_instances(self, label)

        sample_count = self.configuration['sample_count']
        numpy_datatype = self.configuration.get('numpy_datatype', numpy.int8)

        X = numpy.empty((sample_count, 0), dtype=numpy_datatype)
        Y = numpy.empty((sample_count, 0), dtype=numpy_datatype)

        objective_names = sorted(self.configuration.get('objectives', []))
        feature_names = list(sorted(list(set(self.bayesian_network.variable_node_names()) - set(objective_names))))

        for varname in feature_names:
            varindex = self.bayesian_network.variable_nodes_index(varname)
            feature = instances_matrix[:, varindex][numpy.newaxis].T
            X = numpy.hstack((X, feature))

        for varname in objective_names:
            varindex = self.bayesian_network.variable_nodes_index(varname)
            objective = instances_matrix[:, varindex][numpy.newaxis].T
            Y = numpy.hstack((Y, objective))

        datasetmatrix = DatasetMatrix(label)
        datasetmatrix.X = scipy.sparse.csr_matrix(X)
        datasetmatrix.Y = scipy.sparse.csr_matrix(Y)
        datasetmatrix.row_labels = ['row{}'.format(i) for i in range(0, sample_count)]
        datasetmatrix.column_labels_X = feature_names
        datasetmatrix.column_labels_Y = objective_names
        datasetmatrix.metadata['source'] = self

        return datasetmatrix


    def create_exact_instances(self, label='bayesian_network', other_random_seed=-1):
        sample_count = self.configuration['sample_count']
        numpy_datatype = self.configuration.get('numpy_datatype', numpy.int8)

        joint_pmf = self.bayesian_network.create_joint_pmf(values_as_indices=True)
        instances = joint_pmf.create_instances_list(sample_count)
        instances_as_lists = [list(instance) for instance in instances]
        instances_matrix = numpy.asarray(instances_as_lists, dtype=numpy_datatype)

        return instances_matrix


    def create_random_instances(self, label='bayesian_network', other_random_seed=-1):
        if self.reset_random_seed or (other_random_seed != -1 and other_random_seed != self.configuration['random_seed']):
            if other_random_seed == -1:
                random.seed(self.configuration['random_seed'])
            else:
                random.seed(other_random_seed)
            self.reset_random_seed = False

        sample_count = self.configuration['sample_count']
        numpy_datatype = self.configuration.get('numpy_datatype', numpy.int8)
        instances_matrix = self.bayesian_network.sample_matrix(sample_count, dtype=numpy_datatype)

        return instances_matrix
