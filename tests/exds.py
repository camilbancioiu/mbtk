import unittest
import numpy
from scipy.sparse import csr_matrix, identity

import test_utilities as util
from ..datasets.datasetmatrix import DatasetMatrix

# TODO put the default matrices into CSR format
class TestDatasetMatrix(unittest.TestCase):
    def default_matrix_X(self):
        return numpy.matrix(
                [ 1,  2,  3,  4],
                [ 5,  6,  7,  8],
                [ 9, 10, 11, 12],
                [13, 14, 15, 16])

    def default_matrix_Y(self):
        return numpy.matrix(
                [101, 102],
                [201, 202],
                [301, 302],
                [401, 402])

    def default_row_labels(self):
        matrix = self.default_matrix_X()
        row_count = matrix.shape[0]
        return ["row{}".format(r) for r in range(row_count)]

    def default_column_labels_X(self):
        matrix = self.default_matrix_X()
        col_count = matrix.shape[1]
        return ["colx{}".format(c) for c in range(col_count)]

    def default_column_labels_Y(self):
        matrix = self.default_matrix_Y()
        col_count = matrix.shape[1]
        return ["coly{}".format(c) for c in range(col_count)]

    def configure_default_datasetmatrix(self):
        datasetmatrix.X = self.default_matrix_X()
        datasetmatrix.Y = self.default_matrix_Y()
        datasetmatrix.row_labels = self.default_row_labels()
        datasetmatrix.column_labels_X = self.default_column_labels_X()
        datasetmatrix.column_labels_Y = self.default_column_labels_Y()
        return datasetmatrix

    def test_saving_and_loading(self):
        folder = util.ensure_tmp_subfolder('datasetmatrix')

        datasetmatrix = DatasetMatrix('testmatrix')
        self.configure_default_datasetmatrix(datasetmatrix)
        datasetmatrix.save(folder)

        datasetmatrix2 = DatasetMatrix('testmatrix')
        datasetmatrix2.load(folder)
