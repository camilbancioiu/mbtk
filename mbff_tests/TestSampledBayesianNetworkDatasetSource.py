from pathlib import Path

import numpy
import unittest
import random

from mbff_tests.TestBase import TestBase

from mbff.dataset.sources.SampledBayesianNetworkDatasetSource import SampledBayesianNetworkDatasetSource
import mbff.utilities.functions as util


class TestSampledBayesianNetworkDatasetSource(TestBase):

    @unittest.skipIf(TestBase.tag_excluded('sampling'), 'Sampling tests excluded')
    def test_sampling_bayesian_network_as_dataset_source(self):
        configuration = self.default_configuration()
        sample_count = configuration['sample_count']

        random.seed(configuration['random_seed'])
        bayesian_network = util.read_bif_file(configuration['sourcepath'])
        bayesian_network.finalize()
        sample_matrix = bayesian_network.sample_matrix(configuration['sample_count'])

        sbnds = SampledBayesianNetworkDatasetSource(self.default_configuration())
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


    def default_configuration(self):
        configuration = {}
        configuration['sourcepath'] = Path('testfiles', 'bif_files', 'survey.bif')
        configuration['sample_count'] = 1000
        configuration['random_seed'] = 42
        configuration['objectives'] = ['R', 'TRN']

        return configuration
