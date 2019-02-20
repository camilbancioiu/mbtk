import numpy
import scipy
import scipy.io
import pickle
import mbff.utilities as util
from mbff.datasets.Exceptions import DatasetMatrixNotFinalizedError, DatasetMatrixFinalizedError

class DatasetMatrix:
    """
    This class maintains a pair of matrices ``X`` and ``Y``, along with their
    row labels and column labels. Its purpose is to pair together the two
    matrices of a real-world dataset, namely the feature variables (``X``) and
    objective variables (``Y``). Normally, a ``DatasetMatrix`` instance is
    generated by a dataset source when reading an external dataset.

    The rows of ``X`` and ``Y`` at the same index represent a single sample in
    the dataset.  This means that the matrices ``X`` and ``Y`` have the same
    number of rows.

    :var label: The label used to identify this instance of ``DatasetMatrix``.
        It will be used when saving / loading the instance too.
    :var X: A ``scipy.sparse`` matrix where the rows represent samples and the
        columns represent the feature variables. Can be either a
        ``scipy.sparse.csr_matrix`` instance or a ``scipy.sparse.csc_matrix``
        instance.
    :var Y: A ``scipy.sparse`` matrix where the rows represent samples and the
        columns represent the objective variables. Can be either a
        ``scipy.sparse.csr_matrix`` instance or a ``scipy.sparse.csc_matrix``
        instance.
    :var row_labels: A list of strings where each element is the label of a
        corresponding sample in ``X`` and ``Y``. Has exactly as many elements
        as ``X`` and ``Y`` have rows.
    :var column_labels_X: A list of strings where each element is the label of
        a corresponding feature variable in ``X``. Has exactly as many elements
        as ``X`` has columns.
    :var column_labels_Y: A list of strings where each element is the label of
        a corresponding objective variable in ``Y``. Has exactly as many elements
        as ``Y`` has columns.
    :var final: A boolean flag which determines whether this ``DatasetMatrix``
        instance is final or not. Initially ``False``. Calling ``.finalize()``
        will set this to ``True``. Saving is possible only after finalizing. A
        loaded ``DatasetMatrix`` is always finalized.
    """

    def __init__(self, label):
        self.label = label
        self.X = None
        self.Y = None
        self.row_labels = []
        self.column_labels_X = []
        self.column_labels_Y = []
        self.final = False

    def finalize(self):
        """
        Make final adjustments to the ``X`` and ``Y`` matrices. Calls
        ``.eliminate_zeros()``, ``.check_format()`` and ``.tocsc()`` on both
        matrices, then set the ``self.final`` flag to ``True``.
        """
        if self.final == True:
            return

        self.X.eliminate_zeros()
        self.Y.eliminate_zeros()

        self.X.check_format()
        self.Y.check_format()

        self.X = self.X.tocsc()
        self.Y = self.Y.tocsc()

        self.final = True


    def unfinalize(self):
        """
        Reset the ``self.final`` flag and converts ``X`` and ``Y`` back to CSR
        format. This method should not really be used unless for development or
        testing purposes.
        """
        self.X = self.X.tocsr()
        self.Y = self.Y.tocsr()

        self.final = False


    def get_matrix(self, matrix_label):
        if matrix_label == 'X':
            return self.X
        elif matrix_label == 'Y':
            return self.Y
        else:
            raise ValueError('Unknown matrix label. Only X and Y are allowed.')


    def get_column(self, matrix_label, column):
        if matrix_label == 'X':
            return self.get_column_X(column)
        elif matrix_label == 'Y':
            return self.get_column_Y(column)
        else:
            raise ValueError('Unknown matrix label. Only X and Y are allowed.')


    def get_column_labels(self, matrix_label):
        if matrix_label == 'X':
            return self.column_labels_X
        elif matrix_label == 'Y':
            return self.column_labels_Y
        else:
            raise ValueError('Unknown matrix label. Only X and Y are allowed.')


    def get_column_X(self, column):
        """
        Get a column from the ``X`` matrix as a simple 1-dimensional Numpy
        array. It is recommended to call DatasetMatrix.finalize() first,
        because finalizing will convert ``X`` to a CSC matrix and CSC matrices
        are optimized for column slicing and retrieval.
        """
        return self.X.getcol(column).transpose().toarray().ravel()


    def get_column_Y(self, column):
        """
        Get a column from the ``Y`` matrix as a simple 1-dimensional Numpy
        array. It is recommended to call DatasetMatrix.finalize() first,
        because finalizing will convert ``Y`` to a CSC matrix and CSC matrices
        are optimized for column slicing and retrieval.
        """
        return self.Y.getcol(column).transpose().toarray().ravel()


    def save(self, folder):
        """
        Save ``X``, ``Y``, row labels and column labels as files in the
        subfolder ``[folder]/[self.label]``.
        """
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
        """
        Load ``X``, ``Y``, row labels and column labels from the files in the
        subfolder ``[folder]/[self.label]``.
        """
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
        """
        Enable equality testing with the ``==`` operator.

        This equality test includes ``X``, ``Y``, ``row_labels``,
        ``column_labels_X`` and ``column_labels_Y``. All of these properties
        must be equal between the two given instances for the equality test to
        pass.
        """
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
        """
        Generate a textual description of how this ``DatasetMatrix`` differs from the ``other``.
        """
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
        """
        Delete the row at index ``r``. This results in the deletion of row
        ``r`` from the ``X`` matrix *and* the ``Y`` matrix. Also deletes the
        corresponding label from ``row_labels``.
        """
        if self.final == True:
            raise DatasetMatrixFinalizedError(self, "Cannot delete any row.")
        self.X = DatasetMatrix.delete_rows_cols(self.X, row_indices=[r]).tocsr()
        self.Y = DatasetMatrix.delete_rows_cols(self.Y, row_indices=[r]).tocsr()
        del self.row_labels[r]


    def keep_rows(self, rows_to_keep):
        """
        Keep only the rows specified by the indices in ``rows_to_keep``, deleting all
        the other rows from both ``X`` and ``Y``. This affects the ``row_labels`` as well.
        """
        if self.final == True:
            raise DatasetMatrixFinalizedError(self, "Cannot delete any row.")
        if not isinstance(rows_to_keep, list):
            raise TypeError("Argument 'rows_to_keep' must be a non-empty list")
        if len(rows_to_keep) == 0:
            raise ValueError("Argument 'rows_to_keep' must be a non-empty list")

        all_rows = range(self.X.get_shape()[0])
        rows_to_delete = list(set(all_rows) - set(rows_to_keep))
        self.X = DatasetMatrix.delete_rows_cols(self.X, row_indices=rows_to_delete).tocsr()
        self.Y = DatasetMatrix.delete_rows_cols(self.Y, row_indices=rows_to_delete).tocsr()

        row_labels_to_keep = [self.row_labels[r] for r in rows_to_keep]
        self.row_labels = row_labels_to_keep


    def select_rows(self, rows_to_keep, new_label=""):
        """
        Create a new DatasetMatrix instance, which only contains the requested
        rows of both ``X`` and ``Y``. Affects ``row_labels`` as well. Does not
        raise ``DatasetMatrixFinalizedError``, because it doesn't try to modify
        the original matrices.
        """
        if not isinstance(rows_to_keep, list):
            raise TypeError("Argument 'rows_to_keep' must be a non-empty list")
        if len(rows_to_keep) == 0:
            raise ValueError("Argument 'rows_to_keep' must be a non-empty list")

        rows_to_keep = sorted(rows_to_keep)
        if new_label == "":
            new_label = self.label

        new_dataset_matrix = DatasetMatrix(new_label)

        all_rows = range(self.X.get_shape()[0])
        new_dataset_matrix.X = self.X[rows_to_keep,]
        new_dataset_matrix.Y = self.Y[rows_to_keep,]
        new_dataset_matrix.row_labels = [self.row_labels[i] for i in rows_to_keep]
        new_dataset_matrix.column_labels_X = self.column_labels_X.copy()
        new_dataset_matrix.column_labels_Y = self.column_labels_Y.copy()

        return new_dataset_matrix


    def delete_columns_X(self, columns_to_delete):
        if self.final == True:
            raise DatasetMatrixFinalizedError(self, "Cannot delete any X column.")
        columns_to_delete = sorted(columns_to_delete)
        self.X = DatasetMatrix.delete_rows_cols(self.X, col_indices=columns_to_delete).tocsr()
        self.column_labels_X = [self.column_labels_X[c] for c in range(len(self.column_labels_X)) if c not in columns_to_delete]


    def delete_columns_Y(self, columns_to_delete):
        if self.final == True:
            raise DatasetMatrixFinalizedError(self, "Cannot delete any Y column.")
        columns_to_delete = sorted(columns_to_delete)
        self.Y = DatasetMatrix.delete_rows_cols(self.Y, col_indices=columns_to_delete).tocsr()
        self.column_labels_Y = [self.column_labels_Y[c] for c in range(len(self.column_labels_Y)) if c not in columns_to_delete]


    def delete_column_X(self, c):
        """ Delete the column at index ``c`` from the ``X`` matrix. Also deletes the corresponding label from ``column_labels_X``. """
        if self.final == True:
            raise DatasetMatrixFinalizedError(self, "Cannot delete any X column.")
        self.X = DatasetMatrix.delete_rows_cols(self.X, col_indices=[c]).tocsr()
        del self.column_labels_X[c]


    def delete_column_Y(self, c):
        """ Delete the column at index ``c`` from the ``X`` matrix. Also deletes the corresponding label from ``column_labels_Y``. """
        if self.final == True:
            raise DatasetMatrixFinalizedError(self, "Cannot delete any Y column.")
        self.Y = DatasetMatrix.delete_rows_cols(self.Y, col_indices=[c]).tocsr()
        del self.column_labels_Y[c]



    # Utility static methods.
    ########################

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
        WARNING: Indices of altered axes are reset in the returned matrix.
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


