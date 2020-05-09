import numpy
import random

import mbff_tests.utilities as testutil

from mbff.dataset.sources.SampledBayesianNetworkDatasetSource import SampledBayesianNetworkDatasetSource
import mbff.utilities.functions as util


# @unittest.skipIf(TestBase.tag_excluded('sampling'), 'Sampling tests excluded')
def test_sampling_bayesian_network_as_dataset_source__random():
    configuration = default_configuration()
    configuration['method'] = 'random'
    sample_count = configuration['sample_count']

    random.seed(configuration['random_seed'])
    bayesian_network = util.read_bif_file(configuration['sourcepath'])
    bayesian_network.finalize()
    sample_matrix = bayesian_network.sample_matrix(configuration['sample_count'])

    sbnds = SampledBayesianNetworkDatasetSource(configuration)
    sbnds.reset_random_seed = True
    datasetmatrix = sbnds.create_dataset_matrix('test_sbnds')

    assert ['AGE', 'EDU', 'OCC', 'SEX'] == datasetmatrix.column_labels_X
    assert ['R', 'TRN'] == datasetmatrix.column_labels_Y
    assert ['row{}'.format(i) for i in range(0, sample_count)] == datasetmatrix.row_labels

    assert (sample_count, 4) == datasetmatrix.X.get_shape()
    assert (sample_count, 2) == datasetmatrix.Y.get_shape()

    assert numpy.array_equal(sample_matrix[:, 0], datasetmatrix.get_column_X(0)) is True
    assert numpy.array_equal(sample_matrix[:, 1], datasetmatrix.get_column_X(1)) is True
    assert numpy.array_equal(sample_matrix[:, 2], datasetmatrix.get_column_X(2)) is True
    assert numpy.array_equal(sample_matrix[:, 4], datasetmatrix.get_column_X(3)) is True

    assert numpy.array_equal(sample_matrix[:, 3], datasetmatrix.get_column_Y(0)) is True
    assert numpy.array_equal(sample_matrix[:, 5], datasetmatrix.get_column_Y(1)) is True


# @unittest.skipIf(TestBase.tag_excluded('sampling'), 'Sampling tests excluded')
def test_sampling_bayesian_network_as_dataset_source__exact():
    configuration = default_configuration()
    configuration['method'] = 'exact'
    sample_count = configuration['sample_count']

    bayesian_network = util.read_bif_file(configuration['sourcepath'])
    bayesian_network.finalize()
    joint_pmf = bayesian_network.create_joint_pmf()
    instances = joint_pmf.create_instances_list(sample_count)
    instances_as_lists = [list(instance) for instance in instances]
    instances_matrix = numpy.asarray(instances_as_lists, dtype=numpy.int8)

    sbnds = SampledBayesianNetworkDatasetSource(configuration)
    datasetmatrix = sbnds.create_dataset_matrix('test_sbnds')

    assert ['AGE', 'EDU', 'OCC', 'SEX'] == datasetmatrix.column_labels_X
    assert ['R', 'TRN'] == datasetmatrix.column_labels_Y
    assert ['row{}'.format(i) for i in range(0, sample_count)] == datasetmatrix.row_labels

    assert (sample_count, 4) == datasetmatrix.X.get_shape()
    assert (sample_count, 2) == datasetmatrix.Y.get_shape()

    assert numpy.array_equal(instances_matrix[:, 0], datasetmatrix.get_column_X(0)) is True
    assert numpy.array_equal(instances_matrix[:, 1], datasetmatrix.get_column_X(1)) is True
    assert numpy.array_equal(instances_matrix[:, 2], datasetmatrix.get_column_X(2)) is True
    assert numpy.array_equal(instances_matrix[:, 4], datasetmatrix.get_column_X(3)) is True

    assert numpy.array_equal(instances_matrix[:, 3], datasetmatrix.get_column_Y(0)) is True
    assert numpy.array_equal(instances_matrix[:, 5], datasetmatrix.get_column_Y(1)) is True



def default_configuration():
    configuration = {}
    configuration['sourcepath'] = testutil.bif_folder / 'survey.bif'
    configuration['sample_count'] = 77000
    configuration['random_seed'] = 42
    configuration['objectives'] = ['R', 'TRN']

    return configuration
