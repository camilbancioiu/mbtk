import numpy
import scipy

from mbtk.dataset.sources.DatasetSource import DatasetSource
from mbtk.dataset.DatasetMatrix import DatasetMatrix

class MockDatasetSource(DatasetSource):

    def __init__(self, configuration):
        pass


    def default_datasetmatrix(label):
        sample_count = 8
        feature_count = 8
        datasetmatrix = DatasetMatrix(label)
        datasetmatrix.row_labels = ['row{}'.format(i) for i in range(0, sample_count)]
        datasetmatrix.column_labels_X = ['feature{}'.format(i) for i in range(0, feature_count)]
        datasetmatrix.column_labels_Y = ['objective']
        datasetmatrix.Y = scipy.sparse.csr_matrix(numpy.array([
            [1], # training sample
            [0], # training sample
            [1], # testing sample
            [0], # testing sample
            [1], # testing sample
            [0], # training sample
            [1], # testing sample
            [0]  # testing sample
            ]))
        datasetmatrix.X = scipy.sparse.csr_matrix(numpy.array([
            [1, 1, 1, 1, 0, 1, 0, 1], # training sample
            [0, 1, 1, 1, 1, 0, 0, 1], # training sample
            [1, 1, 1, 0, 0, 0, 1, 0], # testing sample
            [0, 0, 1, 0, 1, 1, 1, 0], # testing sample
            [1, 1, 0, 1, 0, 0, 1, 1], # testing sample
            [0, 0, 0, 1, 1, 1, 0, 1], # training sample
            [1, 1, 1, 1, 0, 0, 1, 0], # testing sample
            [0, 0, 0, 1, 1, 1, 1, 0]  # testing sample
            ]))
        return datasetmatrix


    def create_dataset_matrix(self, label):
        return MockDatasetSource.default_datasetmatrix(label)


