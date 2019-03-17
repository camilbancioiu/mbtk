import numpy
import os
import scipy
import shutil
import unittest

from pathlib import Path

from mbff_tests.TestBase import TestBase

from mbff.dataset.DatasetMatrix import DatasetMatrix
from mbff.dataset.Exceptions import DatasetMatrixNotFinalizedError, DatasetMatrixFinalizedError


class TestDatasetMatrix(TestBase):

    def test_saving_and_loading(self):
        # Set up a simple DatasetMatrix
        dm = DatasetMatrix('testmatrix')
        self.configure_default_datasetmatrix(dm)

        folder = str(self.ensure_empty_tmp_subfolder('test_datasetmatrix__save_load'))
        self.check_saving_and_loading(dm, folder)


    def test_removing_rows(self):
        # Set up a simple DatasetMatrix
        dm = DatasetMatrix('testmatrix')
        self.configure_default_datasetmatrix(dm)

        # Remove the third row. Affects X and Y at the same time.
        dm.delete_row(2)
        expected_X = scipy.sparse.csr_matrix(numpy.array([
                [ 1,  2,  3,  4],
                [ 5,  6,  7,  8],
                [13, 14, 15, 16]]))

        expected_Y = scipy.sparse.csr_matrix(numpy.array([
                [101, 102],
                [201, 202],
                [401, 402]]))

        self.assertTrue(DatasetMatrix.sparse_equal(expected_X, dm.X))
        self.assertTrue(DatasetMatrix.sparse_equal(expected_Y, dm.Y))
        self.assertEqual(["row0", "row1", "row3"], dm.row_labels)
        self.assertEqual(self.default_column_labels_X(), dm.column_labels_X)
        self.assertEqual(self.default_column_labels_Y(), dm.column_labels_Y)

        # Remove the first row.
        dm.delete_row(0)
        expected_X = scipy.sparse.csr_matrix(numpy.array([
                [ 5,  6,  7,  8],
                [13, 14, 15, 16]]))

        expected_Y = scipy.sparse.csr_matrix(numpy.array([
                [201, 202],
                [401, 402]]))

        self.assertTrue(DatasetMatrix.sparse_equal(expected_X, dm.X))
        self.assertTrue(DatasetMatrix.sparse_equal(expected_Y, dm.Y))
        self.assertEqual(["row1", "row3"], dm.row_labels)
        self.assertEqual(self.default_column_labels_X(), dm.column_labels_X)
        self.assertEqual(self.default_column_labels_Y(), dm.column_labels_Y)

        folder = str(self.ensure_empty_tmp_subfolder('test_datasetmatrix__removing_rows'))
        self.check_saving_and_loading(dm, folder)

        dm.unfinalize()

        # Remove both remaining rows.
        dm.delete_row(0)
        dm.delete_row(0)
        expected_X = scipy.sparse.csr_matrix((0, 4))
        expected_Y = scipy.sparse.csr_matrix((0, 2))

        self.assertTrue(DatasetMatrix.sparse_equal(expected_X, dm.X))
        self.assertTrue(DatasetMatrix.sparse_equal(expected_Y, dm.Y))
        self.assertEqual([], dm.row_labels)
        self.assertEqual(self.default_column_labels_X(), dm.column_labels_X)
        self.assertEqual(self.default_column_labels_Y(), dm.column_labels_Y)

        folder = str(self.ensure_empty_tmp_subfolder('test_datasetmatrix__removing_rows'))
        self.check_saving_and_loading(dm, folder)


    def test_keeping_rows(self):
        # Set up a simple DatasetMatrix
        dm = DatasetMatrix('testmatrix')
        self.configure_default_datasetmatrix(dm)

        # Empty lists are not allowed.
        with self.assertRaises(ValueError):
            dm.keep_rows([])

        # Keep rows 1 and 3.
        dm.keep_rows([1, 3])
        expected_X = scipy.sparse.csr_matrix(numpy.array([
                [ 5,  6,  7,  8],
                [13, 14, 15, 16]]))

        expected_Y = scipy.sparse.csr_matrix(numpy.array([
                [201, 202],
                [401, 402]]))

        self.assertTrue(DatasetMatrix.sparse_equal(expected_X, dm.X))
        self.assertTrue(DatasetMatrix.sparse_equal(expected_Y, dm.Y))
        self.assertEqual(['row1', 'row3'], dm.row_labels)
        self.assertEqual(self.default_column_labels_X(), dm.column_labels_X)
        self.assertEqual(self.default_column_labels_Y(), dm.column_labels_Y)

        # Keep row 0 of the remaining 2 (labeled 'row1').
        dm.keep_rows([0])
        expected_X = scipy.sparse.csr_matrix(numpy.array([
                [ 5,  6,  7,  8]]))

        expected_Y = scipy.sparse.csr_matrix(numpy.array([
                [201, 202]]))

        self.assertTrue(DatasetMatrix.sparse_equal(expected_X, dm.X))
        self.assertTrue(DatasetMatrix.sparse_equal(expected_Y, dm.Y))
        self.assertEqual(['row1'], dm.row_labels)
        self.assertEqual(self.default_column_labels_X(), dm.column_labels_X)
        self.assertEqual(self.default_column_labels_Y(), dm.column_labels_Y)

        folder = str(self.ensure_empty_tmp_subfolder('test_datasetmatrix__keeping_rows'))
        self.check_saving_and_loading(dm, folder)


    def test_selecting_rows(self):
        # Set up a simple DatasetMatrix
        dm = DatasetMatrix('testmatrix')
        self.configure_default_datasetmatrix(dm)

        # Empty lists are not allowed.
        with self.assertRaises(ValueError):
            dm.select_rows([])

        # Create new matrix by selecting rows 1 and 3.
        dm = dm.select_rows([1, 3], "test_matrix_selected_rows")
        expected_X = scipy.sparse.csr_matrix(numpy.array([
                [ 5,  6,  7,  8],
                [13, 14, 15, 16]]))

        expected_Y = scipy.sparse.csr_matrix(numpy.array([
                [201, 202],
                [401, 402]]))

        self.assertTrue(DatasetMatrix.sparse_equal(expected_X, dm.X))
        self.assertTrue(DatasetMatrix.sparse_equal(expected_Y, dm.Y))
        self.assertEqual(['row1', 'row3'], dm.row_labels)
        self.assertEqual(self.default_column_labels_X(), dm.column_labels_X)
        self.assertEqual(self.default_column_labels_Y(), dm.column_labels_Y)

        # Keep row 0 of the remaining 2 (labeled 'row1').
        dm = dm.select_rows([0], "test_matrix_selected_rows_2")
        expected_X = scipy.sparse.csr_matrix(numpy.array([
                [ 5,  6,  7,  8]]))

        expected_Y = scipy.sparse.csr_matrix(numpy.array([
                [201, 202]]))

        self.assertTrue(DatasetMatrix.sparse_equal(expected_X, dm.X))
        self.assertTrue(DatasetMatrix.sparse_equal(expected_Y, dm.Y))
        self.assertEqual(['row1'], dm.row_labels)
        self.assertEqual(self.default_column_labels_X(), dm.column_labels_X)
        self.assertEqual(self.default_column_labels_Y(), dm.column_labels_Y)

        folder = str(self.ensure_empty_tmp_subfolder('test_datasetmatrix__selecting_rows'))
        self.check_saving_and_loading(dm, folder)


    def test_selecting_columns_X(self):
        # Set up a simple DatasetMatrix
        dm = DatasetMatrix('testmatrix')
        self.configure_default_datasetmatrix(dm)

        # Empty lists are not allowed.
        with self.assertRaises(ValueError):
            dm.select_columns_X([])

        # Create new datasetmatrix where X has only columns 1 and 2.
        dm = dm.select_columns_X([1, 2], 'test_matrix_selected_colsX')
        expected_X = scipy.sparse.csr_matrix(numpy.array([
                [ 2,  3],
                [ 6,  7],
                [10, 11],
                [14, 15]]))
        expected_Y = self.default_matrix_Y()
        self.assertTrue(DatasetMatrix.sparse_equal(expected_X, dm.X))
        self.assertTrue(DatasetMatrix.sparse_equal(expected_Y, dm.Y))
        self.assertListEqual(self.default_row_labels(), dm.row_labels)
        self.assertListEqual(['colx1', 'colx2'], dm.column_labels_X)
        self.assertListEqual(self.default_column_labels_Y(), dm.column_labels_Y)

        # Select X column 0 from the resulting datasetmatrix.
        dm = dm.select_columns_X([0], 'test_matrix_selected_colsX_2')
        expected_X = scipy.sparse.csr_matrix(numpy.array([
                [ 2],
                [ 6],
                [10],
                [14]]))
        expected_Y = self.default_matrix_Y()
        self.assertTrue(DatasetMatrix.sparse_equal(expected_X, dm.X))
        self.assertTrue(DatasetMatrix.sparse_equal(expected_Y, dm.Y))
        self.assertListEqual(self.default_row_labels(), dm.row_labels)
        self.assertListEqual(['colx1'], dm.column_labels_X)
        self.assertListEqual(self.default_column_labels_Y(), dm.column_labels_Y)

        folder = str(self.ensure_empty_tmp_subfolder('test_datasetmatrix__selecting_colsX'))
        self.check_saving_and_loading(dm, folder)


    def test_removing_columns_X(self):
        # Set up a simple DatasetMatrix
        dm = DatasetMatrix('testmatrix')
        self.configure_default_datasetmatrix(dm)

        # Remove the third column from the X matrix.
        dm.delete_column_X(2)
        expected_X = scipy.sparse.csr_matrix(numpy.array([
                [ 1,  2,  4],
                [ 5,  6,  8],
                [ 9, 10, 12],
                [13, 14, 16]]))

        expected_Y = self.default_matrix_Y()

        self.assertTrue(DatasetMatrix.sparse_equal(expected_X, dm.X))
        self.assertTrue(DatasetMatrix.sparse_equal(expected_Y, dm.Y))
        self.assertEqual(self.default_row_labels(), dm.row_labels)
        self.assertEqual(['colx0', 'colx1', 'colx3'], dm.column_labels_X)
        self.assertEqual(self.default_column_labels_Y(), dm.column_labels_Y)

        # Remove the last column from the X matrix.
        dm.delete_column_X(2)
        expected_X = scipy.sparse.csr_matrix(numpy.array([
                [ 1,  2],
                [ 5,  6],
                [ 9, 10],
                [13, 14]]))

        expected_Y = self.default_matrix_Y()

        self.assertTrue(DatasetMatrix.sparse_equal(expected_X, dm.X))
        self.assertTrue(DatasetMatrix.sparse_equal(expected_Y, dm.Y))
        self.assertEqual(self.default_row_labels(), dm.row_labels)
        self.assertEqual(['colx0', 'colx1'], dm.column_labels_X)
        self.assertEqual(self.default_column_labels_Y(), dm.column_labels_Y)

        folder = str(self.ensure_empty_tmp_subfolder('test_datasetmatrix__removing_columns_X'))
        self.check_saving_and_loading(dm, folder)


    def test_removing_columns_Y(self):
        # Set up a simple DatasetMatrix
        dm = DatasetMatrix('testmatrix')
        self.configure_default_datasetmatrix(dm)

        # Remove the first column from the Y matrix.
        dm.delete_column_Y(0)
        expected_X = self.default_matrix_X()
        expected_Y = scipy.sparse.csr_matrix(numpy.array([
                [102],
                [202],
                [302],
                [402]]))

        self.assertTrue(DatasetMatrix.sparse_equal(expected_X, dm.X))
        self.assertTrue(DatasetMatrix.sparse_equal(expected_Y, dm.Y))
        self.assertEqual(self.default_row_labels(), dm.row_labels)
        self.assertEqual(self.default_column_labels_X(), dm.column_labels_X)
        self.assertEqual(['coly1'], dm.column_labels_Y)

        # Remove the last remaining column from the Y matrix.
        dm.delete_column_Y(0)
        expected_X = self.default_matrix_X()
        expected_Y = scipy.sparse.csr_matrix((4, 0))

        self.assertTrue(DatasetMatrix.sparse_equal(expected_X, dm.X))
        self.assertTrue(DatasetMatrix.sparse_equal(expected_Y, dm.Y))
        self.assertEqual(self.default_row_labels(), dm.row_labels)
        self.assertEqual(self.default_column_labels_X(), dm.column_labels_X)
        self.assertEqual([], dm.column_labels_Y)

        folder = str(self.ensure_empty_tmp_subfolder('test_datasetmatrix__removing_columns_Y'))
        self.check_saving_and_loading(dm, folder)


    def default_matrix_X(self):
        return scipy.sparse.csr_matrix(numpy.array([
                [ 1,  2,  3,  4],
                [ 5,  6,  7,  8],
                [ 9, 10, 11, 12],
                [13, 14, 15, 16]]))


    def default_matrix_Y(self):
        return scipy.sparse.csr_matrix(numpy.array([
                [101, 102],
                [201, 202],
                [301, 302],
                [401, 402]]))


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


    def configure_default_datasetmatrix(self, dm):
        dm.X = self.default_matrix_X()
        dm.Y = self.default_matrix_Y()
        dm.row_labels = self.default_row_labels()
        dm.column_labels_X = self.default_column_labels_X()
        dm.column_labels_Y = self.default_column_labels_Y()


    def check_saving_and_loading(self, dm, folder):
        # Saving must fail, because dm.finalize() has not yet been called.
        with self.assertRaises(DatasetMatrixNotFinalizedError):
            dm.save(folder)
        self.check_no_datamatrix_files(folder, dm.label)

        # Finalize the DatasetMatrix and save it.
        dm.finalize()
        dm.save(Path(folder))
        self.check_datamatrix_files(folder, dm.label)

        # Load the saved data into a fresh DatasetMatrix with the same label
        # and compare with the old one.
        dm2 = DatasetMatrix(dm.label)
        dm2.load(Path(folder))
        self.assertEqual(dm, dm2)


    def check_datamatrix_files(self, folder, label):
        folder += '/' + label
        filelist = [f for f in os.listdir(folder) if os.path.isfile(folder + '/' + f)]
        expectedfiles = ["X.mtx", "Y.mtx", "row_labels.txt", "column_labels_X.txt", "column_labels_Y.txt"]
        self.assertEqual(set(expectedfiles), set(filelist))


    def check_no_datamatrix_files(self, folder, label):
        folder += '/' + label
        self.assertFalse(os.path.isdir(folder))


