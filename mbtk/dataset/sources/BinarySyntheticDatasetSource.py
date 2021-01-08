import numpy
import scipy
import random

from mbtk.dataset.sources.DatasetSource import DatasetSource
from mbtk.dataset.DatasetMatrix import DatasetMatrix


class BinarySyntheticDatasetSource(DatasetSource):
    """
    A dataset source which generates a random binary dataset by specifying the
    proportion of values of ``1`` for each feature and objective.
    """

    def __init__(self, configuration):
        self.configuration = configuration
        self.reset_random_seed = True


    def create_dataset_matrix(self, label='binarydataset', other_random_seed=-1):
        if self.reset_random_seed or (other_random_seed != -1 and other_random_seed != self.configuration['random_seed']):
            if other_random_seed == -1:
                random.seed(self.configuration['random_seed'])
            else:
                random.seed(other_random_seed)
            self.reset_random_seed = False
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
        datasetmatrix.metadata['source'] = self

        return datasetmatrix


    def create_random_binary_matrix(self, row_count, probabilities_per_column):
        column_count = len(probabilities_per_column)
        column_labels = []
        columns = []
        for column_label in sorted(probabilities_per_column.keys()):
            p_one = probabilities_per_column[column_label]
            column = self.create_shuffled_binary_column(row_count, p_one)
            columns.append(column)
            column_labels.append(column_label)

        matrix = scipy.sparse.hstack(columns, dtype=numpy.int16)
        return (matrix, column_labels)


    def create_shuffled_binary_column(self, row_count, p_one):
        n_ones = int(row_count * p_one)
        binary_list = [1] * n_ones + [0] * (row_count - n_ones)
        random.shuffle(binary_list)
        column = scipy.sparse.coo_matrix([binary_list], dtype=numpy.int16).transpose()
        return column

