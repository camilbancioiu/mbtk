from pathlib import Path

import numpy
import unittest
import random

from mbff_tests.TestBase import TestBase

from mbff.dataset.sources.SampledBayesianNetworkDatasetSource import SampledBayesianNetworkDatasetSource
import mbff.utilities.functions as util


class TestSampledBayesianNetworkDatasetSource(TestBase):

    @unittest.skipIf(TestBase.tag_excluded('sampling'), 'Sampling tests excluded')
    def test_sampling_bayesian_network_as_dataset_source__random(self):
        configuration = self.default_configuration()
        configuration['method'] = 'random'
        sample_count = configuration['sample_count']

        random.seed(configuration['random_seed'])
        bayesian_network = util.read_bif_file(configuration['sourcepath'])
        bayesian_network.finalize()
        sample_matrix = bayesian_network.sample_matrix(configuration['sample_count'])

        sbnds = SampledBayesianNetworkDatasetSource(configuration)
        sbnds.reset_random_seed = True
        datasetmatrix = sbnds.create_dataset_matrix('test_sbnds')

        self.assertListEqual(['AGE', 'EDU', 'OCC', 'SEX'], datasetmatrix.column_labels_X)
        self.assertListEqual(['R', 'TRN'], datasetmatrix.column_labels_Y)
        self.assertListEqual(['row{}'.format(i) for i in range(0, sample_count)], datasetmatrix.row_labels)

        self.assertEqual((sample_count, 4), datasetmatrix.X.get_shape())
        self.assertEqual((sample_count, 2), datasetmatrix.Y.get_shape())

        self.assertTrue(numpy.array_equal(sample_matrix[:, 0], datasetmatrix.get_column_X(0)))
        self.assertTrue(numpy.array_equal(sample_matrix[:, 1], datasetmatrix.get_column_X(1)))
        self.assertTrue(numpy.array_equal(sample_matrix[:, 2], datasetmatrix.get_column_X(2)))
        self.assertTrue(numpy.array_equal(sample_matrix[:, 4], datasetmatrix.get_column_X(3)))

        self.assertTrue(numpy.array_equal(sample_matrix[:, 3], datasetmatrix.get_column_Y(0)))
        self.assertTrue(numpy.array_equal(sample_matrix[:, 5], datasetmatrix.get_column_Y(1)))


    @unittest.skipIf(TestBase.tag_excluded('sampling'), 'Sampling tests excluded')
    def test_sampling_bayesian_network_as_dataset_source__exact(self):
        configuration = self.default_configuration()
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

        self.assertListEqual(['AGE', 'EDU', 'OCC', 'SEX'], datasetmatrix.column_labels_X)
        self.assertListEqual(['R', 'TRN'], datasetmatrix.column_labels_Y)
        self.assertListEqual(['row{}'.format(i) for i in range(0, sample_count)], datasetmatrix.row_labels)

        self.assertEqual((sample_count, 4), datasetmatrix.X.get_shape())
        self.assertEqual((sample_count, 2), datasetmatrix.Y.get_shape())

        self.assertTrue(numpy.array_equal(instances_matrix[:, 0], datasetmatrix.get_column_X(0)))
        self.assertTrue(numpy.array_equal(instances_matrix[:, 1], datasetmatrix.get_column_X(1)))
        self.assertTrue(numpy.array_equal(instances_matrix[:, 2], datasetmatrix.get_column_X(2)))
        self.assertTrue(numpy.array_equal(instances_matrix[:, 4], datasetmatrix.get_column_X(3)))

        self.assertTrue(numpy.array_equal(instances_matrix[:, 3], datasetmatrix.get_column_Y(0)))
        self.assertTrue(numpy.array_equal(instances_matrix[:, 5], datasetmatrix.get_column_Y(1)))


    def default_configuration(self):
        configuration = {}
        configuration['sourcepath'] = Path('testfiles', 'bif_files', 'survey.bif')
        configuration['sample_count'] = 77000
        configuration['random_seed'] = 42
        configuration['objectives'] = ['R', 'TRN']

        return configuration
