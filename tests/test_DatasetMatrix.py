import numpy
import scipy

from pathlib import Path

import tests.utilities as testutil
import pytest

from mbff.dataset.DatasetMatrix import DatasetMatrix
from mbff.dataset.Exceptions import DatasetMatrixNotFinalizedError


def test_saving_and_loading():
    # Set up a simple DatasetMatrix
    dm = DatasetMatrix('testmatrix')
    configure_default_datasetmatrix(dm)

    folder = testutil.ensure_empty_tmp_subfolder('test_datasetmatrix__save_load')
    check_saving_and_loading(dm, folder)


def test_removing_rows():
    # Set up a simple DatasetMatrix
    dm = DatasetMatrix('testmatrix')
    configure_default_datasetmatrix(dm)

    # Remove the third row. Affects X and Y at the same time.
    dm.delete_row(2)
    expected_X = scipy.sparse.csr_matrix(numpy.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [13, 14, 15, 16]]))

    expected_Y = scipy.sparse.csr_matrix(numpy.array([
        [101, 102],
        [201, 202],
        [401, 402]]))

    assert DatasetMatrix.sparse_equal(expected_X, dm.X) is True
    assert DatasetMatrix.sparse_equal(expected_Y, dm.Y) is True
    assert ["row0", "row1", "row3"] == dm.row_labels
    assert default_column_labels_X() == dm.column_labels_X
    assert default_column_labels_Y() == dm.column_labels_Y

    # Remove the first row.
    dm.delete_row(0)
    expected_X = scipy.sparse.csr_matrix(numpy.array([
        [5, 6, 7, 8],
        [13, 14, 15, 16]]))

    expected_Y = scipy.sparse.csr_matrix(numpy.array([
        [201, 202],
        [401, 402]]))

    assert DatasetMatrix.sparse_equal(expected_X, dm.X) is True
    assert DatasetMatrix.sparse_equal(expected_Y, dm.Y) is True
    assert ["row1", "row3"] == dm.row_labels
    assert default_column_labels_X() == dm.column_labels_X
    assert default_column_labels_Y() == dm.column_labels_Y

    folder = testutil.ensure_empty_tmp_subfolder('test_datasetmatrix__removing_rows')
    check_saving_and_loading(dm, folder)

    dm.unfinalize()

    # Remove both remaining rows.
    dm.delete_row(0)
    dm.delete_row(0)
    expected_X = scipy.sparse.csr_matrix((0, 4))
    expected_Y = scipy.sparse.csr_matrix((0, 2))

    assert DatasetMatrix.sparse_equal(expected_X, dm.X) is True
    assert DatasetMatrix.sparse_equal(expected_Y, dm.Y) is True
    assert [] == dm.row_labels
    assert default_column_labels_X() == dm.column_labels_X
    assert default_column_labels_Y() == dm.column_labels_Y

    folder = testutil.ensure_empty_tmp_subfolder('test_datasetmatrix__removing_rows')
    check_saving_and_loading(dm, folder)


def test_keeping_rows():
    # Set up a simple DatasetMatrix
    dm = DatasetMatrix('testmatrix')
    configure_default_datasetmatrix(dm)

    # Empty lists are not allowed.
    with pytest.raises(ValueError):
        dm.keep_rows([])

    # Keep rows 1 and 3.
    dm.keep_rows([1, 3])
    expected_X = scipy.sparse.csr_matrix(numpy.array([
        [5, 6, 7, 8],
        [13, 14, 15, 16]]))

    expected_Y = scipy.sparse.csr_matrix(numpy.array([
        [201, 202],
        [401, 402]]))

    assert DatasetMatrix.sparse_equal(expected_X, dm.X) is True
    assert DatasetMatrix.sparse_equal(expected_Y, dm.Y) is True
    assert ['row1', 'row3'] == dm.row_labels
    assert default_column_labels_X() == dm.column_labels_X
    assert default_column_labels_Y() == dm.column_labels_Y

    # Keep row 0 of the remaining 2 (labeled 'row1').
    dm.keep_rows([0])
    expected_X = scipy.sparse.csr_matrix(numpy.array([
        [5, 6, 7, 8]]))

    expected_Y = scipy.sparse.csr_matrix(numpy.array([
        [201, 202]]))

    assert DatasetMatrix.sparse_equal(expected_X, dm.X) is True
    assert DatasetMatrix.sparse_equal(expected_Y, dm.Y) is True
    assert ['row1'] == dm.row_labels
    assert default_column_labels_X() == dm.column_labels_X
    assert default_column_labels_Y() == dm.column_labels_Y

    folder = testutil.ensure_empty_tmp_subfolder('test_datasetmatrix__keeping_rows')
    check_saving_and_loading(dm, folder)


def test_selecting_rows():
    # Set up a simple DatasetMatrix
    dm = DatasetMatrix('testmatrix')
    configure_default_datasetmatrix(dm)

    # Empty lists are not allowed.
    with pytest.raises(ValueError):
        dm.select_rows([])

    # Create new matrix by selecting rows 1 and 3.
    dm = dm.select_rows([1, 3], "test_matrix_selected_rows")
    expected_X = scipy.sparse.csr_matrix(numpy.array([
        [5, 6, 7, 8],
        [13, 14, 15, 16]]))

    expected_Y = scipy.sparse.csr_matrix(numpy.array([
        [201, 202],
        [401, 402]]))

    assert DatasetMatrix.sparse_equal(expected_X, dm.X) is True
    assert DatasetMatrix.sparse_equal(expected_Y, dm.Y) is True
    assert ['row1', 'row3'] == dm.row_labels
    assert default_column_labels_X() == dm.column_labels_X
    assert default_column_labels_Y() == dm.column_labels_Y

    # Keep row 0 of the remaining 2 (labeled 'row1').
    dm = dm.select_rows([0], "test_matrix_selected_rows_2")
    expected_X = scipy.sparse.csr_matrix(numpy.array([
        [5, 6, 7, 8]]))

    expected_Y = scipy.sparse.csr_matrix(numpy.array([
        [201, 202]]))

    assert DatasetMatrix.sparse_equal(expected_X, dm.X) is True
    assert DatasetMatrix.sparse_equal(expected_Y, dm.Y) is True
    assert ['row1'] == dm.row_labels
    assert default_column_labels_X() == dm.column_labels_X
    assert default_column_labels_Y() == dm.column_labels_Y

    folder = testutil.ensure_empty_tmp_subfolder('test_datasetmatrix__selecting_rows')
    check_saving_and_loading(dm, folder)


def test_selecting_columns_X():
    # Set up a simple DatasetMatrix
    dm = DatasetMatrix('testmatrix')
    configure_default_datasetmatrix(dm)

    # Empty lists are not allowed.
    with pytest.raises(ValueError):
        dm.select_columns_X([])

    # Create new datasetmatrix where X has only columns 1 and 2.
    dm = dm.select_columns_X([1, 2], 'test_matrix_selected_colsX')
    expected_X = scipy.sparse.csr_matrix(numpy.array([
        [2, 3],
        [6, 7],
        [10, 11],
        [14, 15]]))
    expected_Y = default_matrix_Y()
    assert DatasetMatrix.sparse_equal(expected_X, dm.X) is True
    assert DatasetMatrix.sparse_equal(expected_Y, dm.Y) is True
    assert default_row_labels() == dm.row_labels
    assert ['colx1', 'colx2'] == dm.column_labels_X
    assert default_column_labels_Y() == dm.column_labels_Y

    # Select X column 0 from the resulting datasetmatrix.
    dm = dm.select_columns_X([0], 'test_matrix_selected_colsX_2')
    expected_X = scipy.sparse.csr_matrix(numpy.array([
        [2],
        [6],
        [10],
        [14]]))
    expected_Y = default_matrix_Y()
    assert DatasetMatrix.sparse_equal(expected_X, dm.X) is True
    assert DatasetMatrix.sparse_equal(expected_Y, dm.Y) is True
    assert default_row_labels() == dm.row_labels
    assert ['colx1'] == dm.column_labels_X
    assert default_column_labels_Y() == dm.column_labels_Y

    folder = testutil.ensure_empty_tmp_subfolder('test_datasetmatrix__selecting_colsX')
    check_saving_and_loading(dm, folder)


def test_removing_columns_X():
    # Set up a simple DatasetMatrix
    dm = DatasetMatrix('testmatrix')
    configure_default_datasetmatrix(dm)

    # Remove the third column from the X matrix.
    dm.delete_column_X(2)
    expected_X = scipy.sparse.csr_matrix(numpy.array([
        [1, 2, 4],
        [5, 6, 8],
        [9, 10, 12],
        [13, 14, 16]]))

    expected_Y = default_matrix_Y()

    assert DatasetMatrix.sparse_equal(expected_X, dm.X) is True
    assert DatasetMatrix.sparse_equal(expected_Y, dm.Y) is True
    assert default_row_labels() == dm.row_labels
    assert ['colx0', 'colx1', 'colx3'] == dm.column_labels_X
    assert default_column_labels_Y() == dm.column_labels_Y

    # Remove the last column from the X matrix.
    dm.delete_column_X(2)
    expected_X = scipy.sparse.csr_matrix(numpy.array([
        [1, 2],
        [5, 6],
        [9, 10],
        [13, 14]]))

    expected_Y = default_matrix_Y()

    assert DatasetMatrix.sparse_equal(expected_X, dm.X) is True
    assert DatasetMatrix.sparse_equal(expected_Y, dm.Y) is True
    assert default_row_labels() == dm.row_labels
    assert ['colx0', 'colx1'] == dm.column_labels_X
    assert default_column_labels_Y() == dm.column_labels_Y

    folder = testutil.ensure_empty_tmp_subfolder('test_datasetmatrix__removing_columns_X')
    check_saving_and_loading(dm, folder)


def test_removing_columns_Y():
    # Set up a simple DatasetMatrix
    dm = DatasetMatrix('testmatrix')
    configure_default_datasetmatrix(dm)

    # Remove the first column from the Y matrix.
    dm.delete_column_Y(0)
    expected_X = default_matrix_X()
    expected_Y = scipy.sparse.csr_matrix(numpy.array([
        [102],
        [202],
        [302],
        [402]]))

    assert DatasetMatrix.sparse_equal(expected_X, dm.X) is True
    assert DatasetMatrix.sparse_equal(expected_Y, dm.Y) is True
    assert default_row_labels() == dm.row_labels
    assert default_column_labels_X() == dm.column_labels_X
    assert ['coly1'] == dm.column_labels_Y

    # Remove the last remaining column from the Y matrix.
    dm.delete_column_Y(0)
    expected_X = default_matrix_X()
    expected_Y = scipy.sparse.csr_matrix((4, 0))

    assert DatasetMatrix.sparse_equal(expected_X, dm.X) is True
    assert DatasetMatrix.sparse_equal(expected_Y, dm.Y) is True
    assert default_row_labels() == dm.row_labels
    assert default_column_labels_X() == dm.column_labels_X
    assert [] == dm.column_labels_Y

    folder = testutil.ensure_empty_tmp_subfolder('test_datasetmatrix__removing_columns_Y')
    check_saving_and_loading(dm, folder)


def test_making_variables():
    # Set up a simple DatasetMatrix
    dm = DatasetMatrix('testmatrix')
    configure_default_datasetmatrix(dm)

    variable = dm.get_variable('X', 1)
    assert 1 == variable.ID
    assert 'colx1' == variable.name
    assert variable.instances_list is None
    assert variable.values is None
    assert variable.lazy_instances_loader is not None

    variable.load_instances()
    assert variable.instances_list is not None
    assert 4 == len(variable)
    assert dm.get_column('X', 1).tolist() == variable.instances().tolist()
    variable.update_values()
    assert [2, 6, 10, 14] == variable.values


def test_getting_values_per_column():
    dm = DatasetMatrix('testmatrix')
    dm.X = scipy.sparse.csr_matrix(numpy.array([
        [0, 1, 1, 2, 0, 3],
        [1, 4, 1, 2, 0, 1],
        [1, 5, 1, 0, 0, 3],
        [2, 16, 1, 9, 0, 2],
        [2, -5, 1, 3, 0, 1]
    ]))
    dm.Y = dm.X.transpose()

    column_values_X = dm.get_values_per_column('X')
    column_values_Y = dm.get_values_per_column('Y')

    expected_col_values_X = [[0, 1, 2],
                             [-5, 1, 4, 5, 16],
                             [1],
                             [0, 2, 3, 9],
                             [0],
                             [1, 2, 3]]
    assert column_values_X == expected_col_values_X

    expected_col_values_Y = [[0, 1, 2, 3],
                             [0, 1, 2, 4],
                             [0, 1, 3, 5],
                             [0, 1, 2, 9, 16],
                             [-5, 0, 1, 2, 3]]
    assert column_values_Y == expected_col_values_Y



def default_matrix_X():
    return scipy.sparse.csr_matrix(numpy.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]]))


def default_matrix_Y():
    return scipy.sparse.csr_matrix(numpy.array([
        [101, 102],
        [201, 202],
        [301, 302],
        [401, 402]]))


def default_row_labels():
    matrix = default_matrix_X()
    row_count = matrix.shape[0]
    return ["row{}".format(r) for r in range(row_count)]


def default_column_labels_X():
    matrix = default_matrix_X()
    col_count = matrix.shape[1]
    return ["colx{}".format(c) for c in range(col_count)]


def default_column_labels_Y():
    matrix = default_matrix_Y()
    col_count = matrix.shape[1]
    return ["coly{}".format(c) for c in range(col_count)]


def configure_default_datasetmatrix(dm):
    dm.X = default_matrix_X()
    dm.Y = default_matrix_Y()
    dm.row_labels = default_row_labels()
    dm.column_labels_X = default_column_labels_X()
    dm.column_labels_Y = default_column_labels_Y()


def check_saving_and_loading(dm, folder):
    # Saving must fail, because dm.finalize() has not yet been called.
    with pytest.raises(DatasetMatrixNotFinalizedError):
        dm.save(folder)
    check_no_datamatrix_folder(folder, dm.label)

    # Finalize the DatasetMatrix and save it.
    dm.finalize()
    dm.save(Path(folder))
    check_datamatrix_files(folder, dm.label)

    # Load the saved data into a fresh DatasetMatrix with the same label
    # and compare with the old one.
    dm2 = DatasetMatrix(dm.label)
    dm2.load(Path(folder))
    assert dm == dm2


def check_datamatrix_files(folder, label):
    folder = folder / label
    filelist = [f.name for f in folder.iterdir() if f.is_file()]
    expectedfiles = ["X.mtx", "Y.mtx", "row_labels.txt", "column_labels_X.txt", "column_labels_Y.txt"]
    assert set(expectedfiles) == set(filelist)


def check_no_datamatrix_folder(folder, label):
    folder = folder / label
    assert folder.is_dir() is False
