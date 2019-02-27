import scipy
import numpy

from mbff_tests.TestBase import TestBase

from mbff.algorithms.basic.igt import algorithm_IGt__binary

from mbff.dataset.DatasetMatrix import DatasetMatrix

class TestAlgorithmIGt(TestBase):

    def test_selecting_features(self):
        datasetmatrix = self.default_dataset()

        # Select the two best features.
        parameters = {
                'Q': 2,
                'objective_index': 0
                }
        expected_features = [0, 4]
        computed_features = algorithm_IGt__binary(datasetmatrix, parameters)
        self.assertListEqual(expected_features, computed_features)

        # Select the top four features.
        parameters = {
                'Q': 4,
                'objective_index': 0
                }
        expected_features = [0, 4, 1, 5]
        computed_features = algorithm_IGt__binary(datasetmatrix, parameters)
        self.assertListEqual(expected_features, computed_features)

        # Select all features, thus seeing them sorted by MI.
        parameters = {
                'Q': 8,
                'objective_index': 0
                }
        expected_features = [0, 4, 1, 5, 2, 6, 3, 7]
        computed_features = algorithm_IGt__binary(datasetmatrix, parameters)
        self.assertListEqual(expected_features, computed_features)


    def default_dataset(self):
        sample_count = 8
        feature_count = 8
        datasetmatrix = DatasetMatrix("test")
        datasetmatrix.row_labels = ['row{}'.format(i) for i in range(0, sample_count)]
        datasetmatrix.column_labels_X = ['feature{}'.format(i) for i in range(0, feature_count)]
        datasetmatrix.column_labels_Y = ['objective']
        datasetmatrix.Y = scipy.sparse.csr_matrix(numpy.matrix([
            [1],
            [0],
            [1],
            [0],
            [1],
            [0],
            [1],
            [0]
            ]))
        datasetmatrix.X = scipy.sparse.csr_matrix(numpy.matrix([
            [1, 1, 1, 1, 0, 1, 0, 1],
            [0, 1, 1, 1, 1, 0, 0, 1],
            [1, 1, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 1, 1, 1, 0],
            [1, 1, 0, 1, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 1, 0, 1],
            [1, 1, 1, 1, 0, 0, 1, 0],
            [0, 0, 0, 1, 1, 1, 1, 0]
            ]))
        return datasetmatrix
