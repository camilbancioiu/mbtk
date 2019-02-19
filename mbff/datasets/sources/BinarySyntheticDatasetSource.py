import numpy
import scipy
import random

from mbff.datasets.sources.DatasetSource import DatasetSource
from mbff.datasets.DatasetMatrix import DatasetMatrix

class BinarySyntheticDatasetSource(DatasetSource):
    """
    A dataset source which generates a random binary dataset by specifying the
    proportion of ``1``s for each feature and objective.
    """

    def __init__(self, configuration):
        self.configuration = configuration


    def create_dataset_matrix(self, label='binarydataset'):
        random.seed(self.configuration['random_seed'])
        (X, col_labels_X) = self.create_random_binary_matrix(
                self.configuration['row_count'],
                self.configuration['features']
                )

        (Y, col_labels_Y) = self.create_random_binary_matrix(
                self.configuration['row_count'],
                self.configuration['objectives']
                )

        datasetmatrix = DatasetMatrix(label)
        datasetmatrix.X = X.tocsr()
        datasetmatrix.Y = Y.tocsr()
        datasetmatrix.row_labels = ['row{}'.format(i) for i in range(0, self.configuration['row_count'])]
        datasetmatrix.column_labels_X = col_labels_X
        datasetmatrix.column_labels_Y = col_labels_Y

        return datasetmatrix


    def create_random_binary_matrix(self, row_count, probabilities_per_column):
        column_count = len(probabilities_per_column)
        column_labels = []
        columns = []
        for column_label, p_one in probabilities_per_column.items():
            column = self.create_shuffled_binary_column(row_count, p_one)
            columns.append(column)
            column_labels.append(column_label)

        matrix = scipy.sparse.hstack(columns)
        return (matrix, column_labels)


    def create_shuffled_binary_column(self, row_count, p_one):
        n_ones = int(row_count * p_one)
        binary_list = [1] * n_ones + [0] * (row_count - n_ones)
        random.shuffle(binary_list)
        column = scipy.sparse.coo_matrix([binary_list], dtype=numpy.dtype('B')).transpose()
        return column

