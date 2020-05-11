import numpy

import tests.utilities as testutil

from mbff.dataset.ExperimentalDatasetDefinition import ExperimentalDatasetDefinition
from mbff.dataset.BinaryExperimentalDataset import BinaryExperimentalDataset
from mbff.dataset.sources.BinarySyntheticDatasetSource import BinarySyntheticDatasetSource


def test_feature_removal__no_thresholds():
    folder = testutil.ensure_empty_tmp_subfolder('test_binary_exds_repository__test_feature_removal__no_thresholds')
    definition = default_exds_definition(folder)

    definition.options['remove_features_by_p_thresholds'] = True
    definition.options['remove_objectives_by_p_thresholds'] = True
    definition.options['probability_thresholds__features'] = {}
    definition.options['probability_thresholds__objectives'] = {}
    exds = definition.create_exds()

    # There should be no change made to exds.matrix, exds.matrix_train and
    # exds.matrix_test because we specified no thresholds, in spite of the
    # flags 'remove_features_by_p_thresholds' and
    # 'remove_objectives_by_p_thresholds' set to True.
    exds.build()

    assertExDsDimensions(exds, 25, 75, 8, 4)



def test_feature_removal__thresholds_on_full():
    folder = testutil.ensure_empty_tmp_subfolder('test_binary_exds_repository__test_feature_removal__full')
    definition = default_exds_definition(folder)

    # First, build the exds WITHOUT removing features. We will inspect what features
    # will be chosen for removal by calling the internal method
    # BinaryExperimentalDataset.determine_thresholded_features_to_remove().
    definition.options['remove_features_by_p_thresholds'] = False
    definition.options['remove_objectives_by_p_thresholds'] = False
    # But we do configure thresholds for features, to be able to verify
    # what the exds would remove, if allowed to.
    definition.options['probability_thresholds__features'] = {
        'full': (0.1, 0.8)
    }
    definition.options['probability_thresholds__objectives'] = {}
    exds = definition.create_exds()
    exds.build()
    assertExDsDimensions(exds, 25, 75, 8, 4)

    # Only when analysing the full matrix should we see features to be
    # removed, since we specified thresholds only for 'full'.
    expected_features_to_remove = {
        3: 'galaxy',
        4: 'oxygen',
        5: 'polyrhythm',
        6: 'python'
    }
    assertThresholdedFeaturesToRemove(exds, expected_features_to_remove, ['full'])
    assertThresholdedObjectivesToRemove(exds, {}, [])

    # Now we rebuild the exds, but with feature removal enabled.
    folder = testutil.ensure_empty_tmp_subfolder('test_binary_exds_repository__test_feature_removal__full')
    definition = default_exds_definition(folder)
    definition.options['probability_thresholds__features'] = {
        'full': (0.1, 0.8)
    }
    definition.options['probability_thresholds__objectives'] = {}
    definition.options['remove_features_by_p_thresholds'] = True
    definition.options['remove_objectives_by_p_thresholds'] = True
    exds = definition.create_exds()
    exds.build()
    assertExDsDimensions(exds, 25, 75, 4, 4)
    assertFeaturesNotInExDs(exds, expected_features_to_remove.keys())



def test_feature_removal__thresholds_on_train():
    folder = testutil.ensure_empty_tmp_subfolder('test_binary_exds_repository__test_feature_removal__train')
    definition = default_exds_definition(folder)

    # First, build the exds WITHOUT removing features. We will inspect what features
    # will be chosen for removal by calling the internal method
    # BinaryExperimentalDataset.determine_thresholded_features_to_remove().
    definition.options['remove_features_by_p_thresholds'] = False
    definition.options['remove_objectives_by_p_thresholds'] = False
    # But we do configure thresholds for features, to be able to verify
    # what the exds would remove, if allowed to.
    definition.options['probability_thresholds__features'] = {
        'train': (0.1, 0.8)
    }
    definition.options['probability_thresholds__objectives'] = {}
    exds = definition.create_exds()
    exds.build()
    assertExDsDimensions(exds, 25, 75, 8, 4)

    # Only when analysing the full matrix should we see features to be
    # removed, since we specified thresholds only for 'train'.
    expected_features_to_remove = {
        3: 'galaxy',
        4: 'oxygen',
        5: 'polyrhythm',
        6: 'python',
        7: 'rocket'
    }
    assertThresholdedFeaturesToRemove(exds, expected_features_to_remove, ['train'])
    assertThresholdedObjectivesToRemove(exds, {}, [])

    # Now we rebuild the exds, but with feature removal enabled.
    folder = testutil.ensure_empty_tmp_subfolder('test_binary_exds_repository__test_feature_removal__train')
    definition = default_exds_definition(folder)
    definition.options['probability_thresholds__features'] = {
        'train': (0.1, 0.8)
    }
    definition.options['probability_thresholds__objectives'] = {}
    definition.options['remove_features_by_p_thresholds'] = True
    definition.options['remove_objectives_by_p_thresholds'] = True
    exds = definition.create_exds()
    exds.build()
    assertExDsDimensions(exds, 25, 75, 3, 4)
    assertFeaturesNotInExDs(exds, expected_features_to_remove.keys())



def test_feature_removal__thresholds_on_test():
    folder = testutil.ensure_empty_tmp_subfolder('test_binary_exds_repository__test_feature_removal__test')
    definition = default_exds_definition(folder)

    # First, build the exds WITHOUT removing features. We will inspect what features
    # will be chosen for removal by calling the internal method
    # BinaryExperimentalDataset.determine_thresholded_features_to_remove().
    definition.options['remove_features_by_p_thresholds'] = False
    definition.options['remove_objectives_by_p_thresholds'] = False
    # But we do configure thresholds for features, to be able to verify
    # what the exds would remove, if allowed to.
    definition.options['probability_thresholds__features'] = {
        'test': (0.1, 0.9)
    }
    definition.options['probability_thresholds__objectives'] = {}
    exds = definition.create_exds()
    exds.build()
    assertExDsDimensions(exds, 25, 75, 8, 4)

    # Only when analysing the full matrix should we see features to be
    # removed, since we specified thresholds only for 'test'.
    expected_features_to_remove = {
        3: 'galaxy',
        4: 'oxygen',
        6: 'python'
    }
    assertThresholdedFeaturesToRemove(exds, expected_features_to_remove, ['test'])
    assertThresholdedObjectivesToRemove(exds, {}, [])

    # Now we rebuild the exds, but with feature removal enabled.
    folder = testutil.ensure_empty_tmp_subfolder('test_binary_exds_repository__test_feature_removal__test')
    definition = default_exds_definition(folder)
    definition.options['probability_thresholds__features'] = {
        'test': (0.1, 0.9)
    }
    definition.options['probability_thresholds__objectives'] = {}
    definition.options['remove_features_by_p_thresholds'] = True
    definition.options['remove_objectives_by_p_thresholds'] = True
    exds = definition.create_exds()
    exds.build()
    assertExDsDimensions(exds, 25, 75, 5, 4)
    assertFeaturesNotInExDs(exds, expected_features_to_remove.keys())



def test_objective_removal__thresholds_on_full():
    folder = testutil.ensure_empty_tmp_subfolder('test_binary_exds_repository__test_objective_removal__full')
    definition = default_exds_definition(folder)

    # First, build the exds WITHOUT removing objectives. We will inspect what objectives
    # will be chosen for removal by calling the internal method
    # BinaryExperimentalDataset.determine_thresholded_objectives_to_remove().
    definition.options['remove_features_by_p_thresholds'] = False
    definition.options['remove_objectives_by_p_thresholds'] = False
    # But we do configure thresholds for objectives, to be able to verify
    # what the exds would remove, if allowed to.
    definition.options['probability_thresholds__features'] = {}
    definition.options['probability_thresholds__objectives'] = {
        'full': (0.1, 0.9)
    }
    exds = definition.create_exds()
    exds.build()
    assertExDsDimensions(exds, 25, 75, 8, 4)

    # Only when analysing the full matrix should we see objectives to be
    # removed, since we specified thresholds only for 'full'.
    expected_objectives_to_remove = {
        2: 'sidereal',
        3: 'unknown'
    }
    assertThresholdedFeaturesToRemove(exds, {}, [])
    assertThresholdedObjectivesToRemove(exds, expected_objectives_to_remove, ['full'])

    # Now we rebuild the exds, but with feature removal enabled.
    folder = testutil.ensure_empty_tmp_subfolder('test_binary_exds_repository__test_objective_removal__full')
    definition = default_exds_definition(folder)
    definition.options['probability_thresholds__features'] = {}
    definition.options['probability_thresholds__objectives'] = {
        'full': (0.1, 0.9)
    }
    definition.options['remove_features_by_p_thresholds'] = True
    definition.options['remove_objectives_by_p_thresholds'] = True
    exds = definition.create_exds()
    exds.build()
    assertExDsDimensions(exds, 25, 75, 8, 2)
    assertObjectivesNotInExDs(exds, expected_objectives_to_remove.keys())



def test_objective_removal__thresholds_on_train():
    folder = testutil.ensure_empty_tmp_subfolder('test_binary_exds_repository__test_objective_removal__train')
    definition = default_exds_definition(folder)

    # First, build the exds WITHOUT removing objectives. We will inspect what objectives
    # will be chosen for removal by calling the internal method
    # BinaryExperimentalDataset.determine_thresholded_objectives_to_remove().
    definition.options['remove_features_by_p_thresholds'] = False
    definition.options['remove_objectives_by_p_thresholds'] = False
    # But we do configure thresholds for objectives, to be able to verify
    # what the exds would remove, if allowed to.
    definition.options['probability_thresholds__features'] = {}
    definition.options['probability_thresholds__objectives'] = {
        'train': (0.2, 0.8)
    }
    exds = definition.create_exds()
    exds.build()
    assertExDsDimensions(exds, 25, 75, 8, 4)

    # Only when analysing the full matrix should we see objectives to be
    # removed, since we specified thresholds only for 'train'.
    expected_objectives_to_remove = {
        1: 'encoded',
        2: 'sidereal',
        3: 'unknown'
    }
    assertThresholdedFeaturesToRemove(exds, {}, [])
    assertThresholdedObjectivesToRemove(exds, expected_objectives_to_remove, ['train'])

    # Now we rebuild the exds, but with feature removal enabled.
    folder = testutil.ensure_empty_tmp_subfolder('test_binary_exds_repository__test_objective_removal__train')
    definition = default_exds_definition(folder)
    definition.options['probability_thresholds__features'] = {}
    definition.options['probability_thresholds__objectives'] = {
        'train': (0.2, 0.8)
    }
    definition.options['remove_features_by_p_thresholds'] = True
    definition.options['remove_objectives_by_p_thresholds'] = True
    exds = definition.create_exds()
    exds.build()
    assertExDsDimensions(exds, 25, 75, 8, 1)
    assertObjectivesNotInExDs(exds, expected_objectives_to_remove.keys())



def test_objective_removal__thresholds_on_test():
    folder = testutil.ensure_empty_tmp_subfolder('test_binary_exds_repository__test_objective_removal__test')
    definition = default_exds_definition(folder)

    # First, build the exds WITHOUT removing objectives. We will inspect what objectives
    # will be chosen for removal by calling the internal method
    # BinaryExperimentalDataset.determine_thresholded_objectives_to_remove().
    definition.options['remove_features_by_p_thresholds'] = False
    definition.options['remove_objectives_by_p_thresholds'] = False
    # But we do configure thresholds for objectives, to be able to verify
    # what the exds would remove, if allowed to.
    definition.options['probability_thresholds__features'] = {}
    definition.options['probability_thresholds__objectives'] = {
        'test': (0.0, 0.5)
    }
    exds = definition.create_exds()
    exds.build()
    assertExDsDimensions(exds, 25, 75, 8, 4)

    # Only when analysing the full matrix should we see objectives to be
    # removed, since we specified thresholds only for 'train'.
    expected_objectives_to_remove = {
        2: 'sidereal'
    }
    assertThresholdedFeaturesToRemove(exds, {}, [])
    assertThresholdedObjectivesToRemove(exds, expected_objectives_to_remove, ['test'])

    # Now we rebuild the exds, but with feature removal enabled.
    folder = testutil.ensure_empty_tmp_subfolder('test_binary_exds_repository__test_objective_removal__test')
    definition = default_exds_definition(folder)
    definition.options['probability_thresholds__features'] = {}
    definition.options['probability_thresholds__objectives'] = {
        'test': (0.0, 0.5)
    }
    definition.options['remove_features_by_p_thresholds'] = True
    definition.options['remove_objectives_by_p_thresholds'] = True
    exds = definition.create_exds()
    exds.build()
    assertExDsDimensions(exds, 25, 75, 8, 3)
    assertObjectivesNotInExDs(exds, expected_objectives_to_remove.keys())



def assertFeaturesNotInExDs(exds, feature_labels):
    for feature_label in feature_labels:
        assertFeatureNotInExDs(exds, feature_label)



def assertObjectivesNotInExDs(exds, objective_labels):
    for objective_label in objective_labels:
        assertObjectiveNotInExDs(exds, objective_label)



def assertFeatureNotInExDs(exds, feature_label):
    assert feature_label not in exds.matrix.column_labels_X
    assert feature_label not in exds.matrix_train.column_labels_X
    assert feature_label not in exds.matrix_test.column_labels_X



def assertObjectiveNotInExDs(exds, objective_label):
    assert objective_label not in exds.matrix.column_labels_Y
    assert objective_label not in exds.matrix_train.column_labels_Y
    assert objective_label not in exds.matrix_test.column_labels_Y



def assertExDsDimensions(exds, train_row_count, test_row_count, feature_count, objective_count):
    total_row_count = train_row_count + test_row_count
    assert (total_row_count, feature_count) == exds.matrix.X.get_shape()
    assert (total_row_count, objective_count) == exds.matrix.Y.get_shape()
    assert (train_row_count, feature_count) == exds.matrix_train.X.get_shape()
    assert (train_row_count, objective_count) == exds.matrix_train.Y.get_shape()
    assert (test_row_count, feature_count) == exds.matrix_test.X.get_shape()
    assert (test_row_count, objective_count) == exds.matrix_test.Y.get_shape()
    assert feature_count == len(exds.matrix.column_labels_X)
    assert feature_count == len(exds.matrix_train.column_labels_X)
    assert feature_count == len(exds.matrix_test.column_labels_X)
    assert objective_count == len(exds.matrix.column_labels_Y)
    assert objective_count == len(exds.matrix_train.column_labels_Y)
    assert objective_count == len(exds.matrix_test.column_labels_Y)



def assertThresholdedFeaturesToRemove(exds, expected_features_to_remove, matrices):
    all_matrices = ['full', 'train', 'test']
    for matrix_label in all_matrices:
        if matrix_label in matrices:
            expected = expected_features_to_remove
        else:
            expected = {}
        computed = exds.thresholded_features_to_remove(matrix_label)
        assert expected == computed



def assertThresholdedObjectivesToRemove(exds, expected_objectives_to_remove, matrices):
    all_matrices = ['full', 'train', 'test']
    for matrix_label in all_matrices:
        if matrix_label in matrices:
            expected = expected_objectives_to_remove
        else:
            expected = {}
        computed = exds.thresholded_objectives_to_remove(matrix_label)
        assert expected == computed



def compute_counts_per_feature_columns(datasetmatrix):
    # Count how many values of 1 are there on each feature column.
    computed_counts_per_feature = {}
    row_count = datasetmatrix.X.get_shape()[0]
    for feature_index, feature_label in enumerate(datasetmatrix.column_labels_X):
        feature_column = datasetmatrix.get_column_X(feature_index)
        computed_counts_per_feature[feature_label] = numpy.sum(feature_column) / row_count
    return computed_counts_per_feature



def compute_counts_per_objective_columns(datasetmatrix):
    # Count how many values of 1 are there on each objective column.
    computed_counts_per_objective = {}
    row_count = datasetmatrix.X.get_shape()[0]
    for objective_index, objective_label in enumerate(datasetmatrix.column_labels_Y):
        objective_column = datasetmatrix.get_column_Y(objective_index)
        computed_counts_per_objective[objective_label] = numpy.sum(objective_column) / row_count
    return computed_counts_per_objective



def default_exds_definition(exds_folder):
    definition = ExperimentalDatasetDefinition(exds_folder, "test_binary_exds")
    definition.exds_class = BinaryExperimentalDataset
    definition.source = BinarySyntheticDatasetSource
    definition.source_configuration = default_binarysyntheticdatasetsource_configuration()
    definition.options['training_subset_size'] = 0.25
    definition.options['random_seed'] = 42
    definition.options['probability_thresholds__features'] = {}
    definition.options['probability_thresholds__objectives'] = {}
    definition.after_save__auto_lock = True
    definition.tags = []

    return definition



def default_binarysyntheticdatasetsource_configuration():
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
            'polyrhythm': 85 / 100,
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
