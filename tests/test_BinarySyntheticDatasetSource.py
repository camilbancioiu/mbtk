import numpy

from mbtk.dataset.sources.BinarySyntheticDatasetSource import BinarySyntheticDatasetSource


def test_generating_the_datasetmatrix():
    configuration = default_configuration()
    source = BinarySyntheticDatasetSource(configuration)
    datasetmatrix = source.create_dataset_matrix('binary_synthetic_test')

    # Finalize datasetmatrix, so that its X and Y are converted to CSC
    # format.
    datasetmatrix.finalize()

    expected_counts_per_feature = default_expected_sample_counts__per_feature()
    expected_counts_per_objective = default_expected_sample_counts__per_objective()

    # Count how many values of 1 are there on each feature column.
    computed_counts_per_feature = {}
    for feature_index, feature_label in enumerate(datasetmatrix.column_labels_X):
        feature_column = datasetmatrix.get_column_X(feature_index)
        computed_counts_per_feature[feature_label] = numpy.sum(feature_column)

    # Count how many values of 1 are there on each objective column.
    computed_counts_per_objective = {}
    for objective_index, objective_label in enumerate(datasetmatrix.column_labels_Y):
        objective_column = datasetmatrix.get_column_Y(objective_index)
        computed_counts_per_objective[objective_label] = numpy.sum(objective_column)

    # Compare expected counts of 1 with what was counted in the matrix
    # generated by BinarySyntheticDatasetMatrix.
    assert expected_counts_per_feature == computed_counts_per_feature
    assert expected_counts_per_objective == computed_counts_per_objective

    # Pick a some column from datasetmatrix, sort it, then test for
    # equality with an expected sorted list of 0 and 1. For example, pick
    # column 3.
    test_column = sorted(datasetmatrix.get_column_X(3).tolist())
    expected_column = [0] * 95 + [1] * 5
    assert expected_column == test_column

    # Ensure that the features and objectives have been put in the matrix
    # in alphabetical order.
    assert datasetmatrix.column_labels_X == sorted(datasetmatrix.column_labels_X)
    assert datasetmatrix.column_labels_Y == sorted(datasetmatrix.column_labels_Y)



def default_expected_sample_counts__per_feature():
    # Sorted:
    # almond carbohydrate firefly galaxy oxygen polyrhythm python rocket
    probabilities = {
        'galaxy': 5,
        'almond': 20,
        'python': 0,
        'rocket': 10,
        'carbohydrate': 20,
        'oxygen': 100,
        'polyrhythm': 10,
        'firefly': 20
    }
    return probabilities



def default_expected_sample_counts__per_objective():
    probabilities = {
        'arboreal': 20,
        'encoded': 10,
        'sidereal': 100,
        'unknown': 0
    }
    return probabilities



def default_configuration():
    # The feature labels used here have nothing to do with the words
    # used by the TestRCV1v2DatasetSource class. They are the same words
    # indeed, but there is no real connection.
    configuration = {
        'random_seed': 42,
        'row_count': 100,
        'features': {
            'galaxy': 1 / 20,
            'almond': 1 / 5,
            'python': 0 / 1,
            'rocket': 1 / 10,
            'carbohydrate': 1 / 5,
            'oxygen': 1 / 1,
            'polyrhythm': 1 / 10,
            'firefly': 1 / 5
        },
        'objectives': {
            'arboreal': 1 / 5,
            'encoded': 1 / 10,
            'sidereal': 1 / 1,
            'unknown': 0 / 1
        }
    }
    return configuration
