import numpy
from collections import Counter

variableIDs = [2, 4, 6, 8]
# variableIDs = [2, 4, 6, 8, 10, 9, 3, 1]


def test_counting_efficiency__Counter(ds_alarm_4e4):
    matrix = ds_alarm_4e4.datasetmatrix
    variables = matrix.get_variables('X', variableIDs)

    instances = variables.instances()
    counts = Counter(instances)
    print(len(counts))



def test_counting_efficiency__numpy(ds_alarm_4e4):
    matrix = ds_alarm_4e4.datasetmatrix.get_matrix('X')
    print(type(matrix))
    matrix = matrix.tocsc()
    instances = matrix[:, variableIDs].toarray()
    unique, counts = numpy.unique(instances, axis=0, return_counts=True)



def test_tryout_numpy_csc_slicing(data_small_4):
    matrix, _ = data_small_4
    matrix = matrix.tocsc()
    submatrix = matrix[:, [0, 1]]
    submatrix_array = submatrix.toarray()
    print()
    # print(numpy.unique(submatrix, axis=0))
    print(submatrix_array)
    print()
    print(submatrix_array.shape)
    print()
    print(numpy.unique(submatrix_array, axis=0, return_counts=True))
