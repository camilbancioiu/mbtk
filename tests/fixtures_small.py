import numpy
import scipy
import pytest


@pytest.fixture(scope='session')
def data_small_1():
    dataset = scipy.sparse.csr_matrix(numpy.array([
        [1, 2, 3, 2, 2, 3, 3, 3],
        [2, 1, 1, 2, 1, 2, 2, 2]]).transpose())
    column_values = {
        0: [1, 2, 3],
        1: [1, 2]}
    return (dataset, column_values)



@pytest.fixture(scope='session')
def data_small_2():
    dataset = scipy.sparse.csr_matrix(numpy.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
        [1, 1, 2, 2, 3, 3, 4, 4, 1, 1, 2, 2, 3, 3, 4, 4],
        [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]]).transpose())
    column_values = {
        0: [1, 2],
        1: [1, 2, 3, 4],
        2: [1, 2]}
    return (dataset, column_values)



@pytest.fixture
def data_small_3():
    dataset = scipy.sparse.csr_matrix(numpy.array([
        [1, 1, 2, 1, 1, 2, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2],
        [2, 2, 1, 3, 3, 2, 2, 1, 1, 2, 3, 3, 1, 1, 2, 3],
        [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]]).transpose())
    column_values = {
        0: [1, 2],
        1: [1, 2, 3, 4],
        2: [1, 2]}
    return (dataset, column_values)



@pytest.fixture
def data_small_4():
    dataset = scipy.sparse.csr_matrix(numpy.array([
        [1, 2, 2, 1, 2, 2],
        [1, 3, 4, 1, 3, 3],
        [1, 1, 2, 1, 1, 1]]).transpose())
    column_values = {
        0: [1, 2],
        1: [1, 2, 3, 4],
        2: [1, 2]}
    return (dataset, column_values)
