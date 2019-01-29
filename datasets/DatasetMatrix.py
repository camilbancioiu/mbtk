import numpy
import scipy
import scipy.io
import pickle
from .. import utilities as util

class DatasetMatrix:

    def __init__(self, label):
        self.label = label
        self.X = None
        self.Y = None
        self.row_labels = []
        self.column_labels_X = []
        self.column_labels_Y = []
        self.final = False

    def finalize(self):
        if self.final == True:
            return

        self.X.eliminate_zeros()
        self.Y.eliminate_zeros()

        self.X.check_format()
        self.Y.check_format()

        self.X = self.X.tocsc()
        self.Y = self.Y.tocsc()

        self.final = True


    def save(self, folder):
        if self.final == False:
            raise DatasetMatrixNotFinalizedError(self, "Cannot save.")

        folder += '/' + self.label
        util.ensure_folder(folder)

        util.save_matrix(folder, "X", self.X)
        util.save_matrix(folder, "Y", self.Y)

        with open(folder + '/row_labels.txt', mode='wt') as f:
            f.write('\n'.join(self.row_labels))

        with open(folder + '/column_labels_X.txt', mode='wt') as f:
            f.write('\n'.join(self.column_labels_X))

        with open(folder + '/column_labels_Y.txt', mode='wt') as f:
            f.write('\n'.join(self.column_labels_Y))


    def load(self, folder):
        folder += '/' + self.label

        self.X = util.load_matrix(folder, "X")
        self.Y = util.load_matrix(folder, "Y")

        with open(folder + '/row_labels.txt', mode='rt') as f:
            self.row_labels = list(map(str.strip, list(f)))

        with open(folder + '/column_labels_X.txt', mode='rt') as f:
            self.column_labels_X = list(map(str.strip, list(f)))

        with open(folder + '/column_labels_Y.txt', mode='rt') as f:
            self.column_labels_Y = list(map(str.strip, list(f)))

        self.final = True


    def __eq__(self, other):
        if self.row_labels != other.row_labels:
            return False
        if self.column_labels_X != other.column_labels_X:
            return False
        if self.column_labels_Y != other.column_labels_Y:
            return False
        if not DatasetMatrix.sparse_equal(self.X, other.X):
            return False
        if not DatasetMatrix.sparse_equal(self.Y, other.Y):
            return False
        return True


    def diff(self, other):
        output = ""
        if self.row_labels != other.row_labels:
            output += "row_labels "
        if self.column_labels_X != other.column_labels_X:
            output += "column_labels_X "
        if self.column_labels_Y != other.column_labels_Y:
            output += "column_labels_Y "
        if not DatasetMatrix.sparse_equal(self.X, other.X):
            output += "X "
        if not DatasetMatrix.sparse_equal(self.Y, other.Y):
            output += "Y"
        return output.strip()


    def delete_row(self, r):
        if self.final == True:
            raise DatasetMatrixFinalizedError(self, "Cannot delete any row.")
        self.X = DatasetMatrix.delete_rows_cols(self.X, row_indices=[r]).tocsr()
        self.Y = DatasetMatrix.delete_rows_cols(self.Y, row_indices=[r]).tocsr()
        del self.row_labels[r]


    def delete_column_X(self, c):
        if self.final == True:
            raise DatasetMatrixFinalizedError(self, "Cannot delete any X column.")
        self.X = DatasetMatrix.delete_rows_cols(self.X, col_indices=[c]).tocsr()
        del self.column_labels_X[c]


    def delete_column_Y(self, c):
        if self.final == True:
            raise DatasetMatrixFinalizedError(self, "Cannot delete any Y column.")
        self.Y = DatasetMatrix.delete_rows_cols(self.Y, col_indices=[c]).tocsr()
        del self.column_labels_Y[c]



    # Utility static methods.

    def sparse_equal(m1, m2):
        if m1.get_shape() != m2.get_shape():
            return False
        if (m1 != m2).nnz != 0:
            return False
        return True


    # Taken from https://stackoverflow.com/a/45486349/583574
    def delete_rows_cols(mat, row_indices=[], col_indices=[]):
        """
        Remove the rows (denoted by ``row_indices``) and columns (denoted by
        ``col_indices``) from the CSR sparse matrix ``mat``.
        WARNING: Indices of altered axes are reset in the returned matrix
        """
        if not isinstance(mat, scipy.sparse.csr_matrix):
            raise ValueError("works only for CSR format -- use .tocsr() first")

        rows = []
        cols = []
        if row_indices:
            rows = list(row_indices)
        if col_indices:
            cols = list(col_indices)

        if len(rows) > 0 and len(cols) > 0:
            row_mask = numpy.ones(mat.shape[0], dtype=bool)
            row_mask[rows] = False
            col_mask = numpy.ones(mat.shape[1], dtype=bool)
            col_mask[cols] = False
            return mat[row_mask][:,col_mask]
        elif len(rows) > 0:
            mask = numpy.ones(mat.shape[0], dtype=bool)
            mask[rows] = False
            return mat[mask]
        elif len(cols) > 0:
            mask = numpy.ones(mat.shape[1], dtype=bool)
            mask[cols] = False
            return mat[:,mask]
        else:
            return mat


class DatasetMatrixNotFinalizedError(Exception):
    def __init__(self, datasetmatrix, attempt):
        self.datasetmatrix = datasetmatrix
        self.message = "DatasetMatrix not finalized. " + attempt
        super().__init__(self.message)


class DatasetMatrixFinalizedError(Exception):
    def __init__(self, datasetmatrix, attempt):
        self.datasetmatrix = datasetmatrix
        self.message = "DatasetMatrix already finalized. " + attempt
        super().__init__(self.message)
