import unittest
import numpy

from mbff_tests.TestBase import TestBase

from mbff.dataset.DatasetMatrix import DatasetMatrix
from mbff.dataset.ExperimentalDatasetDefinition import ExperimentalDatasetDefinition
from mbff.dataset.BinaryExperimentalDataset import BinaryExperimentalDataset
from mbff.dataset.sources.BinarySyntheticDatasetSource import BinarySyntheticDatasetSource

class TestBinaryExperimentalDataset(TestBase):

    def test_feature_removal__no_thresholds(self):
        folder = str(self.ensure_empty_tmp_subfolder('test_binary_exds_repository__test_feature_removal__no_thresholds'))
        definition = self.default_exds_definition(folder)

        definition.options['remove_features_by_p_thresholds'] = True
        definition.options['remove_objectives_by_p_thresholds'] = True
        definition.options['probability_thresholds__features'] = { }
        definition.options['probability_thresholds__objectives'] = { }
        exds = definition.create_exds()

        # There should be no change made to exds.matrix, exds.matrix_train and
        # exds.matrix_test because we specified no thresholds, in spite of the
        # flags 'remove_features_by_p_thresholds' and
        # 'remove_objectives_by_p_thresholds' set to True.
        exds.build()

        # self.print_exds_probabilities(exds)
        self.assertExDsDimensions(exds, 25, 75, 8, 4)


    def test_feature_removal__thresholds_on_full(self):
        folder = str(self.ensure_empty_tmp_subfolder('test_binary_exds_repository__test_feature_removal__full'))
        definition = self.default_exds_definition(folder)

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
        definition.options['probability_thresholds__objectives'] = { }
        exds = definition.create_exds()
        exds.build()
        self.assertExDsDimensions(exds, 25, 75, 8, 4)

        # Only when analysing the full matrix should we see features to be
        # removed, since we specified thresholds only for 'full'.
        expected_features_to_remove = {
                3: 'galaxy',
                4: 'oxygen',
                5: 'polyrhythm',
                6: 'python'
                }
        self.assertThresholdedFeaturesToRemove(exds, expected_features_to_remove, ['full'])
        self.assertThresholdedObjectivesToRemove(exds, {}, [])

        # Now we rebuild the exds, but with feature removal enabled.
        folder = str(self.ensure_empty_tmp_subfolder('test_binary_exds_repository__test_feature_removal__full'))
        definition = self.default_exds_definition(folder)
        definition.options['probability_thresholds__features'] = {
                'full': (0.1, 0.8)
                }
        definition.options['probability_thresholds__objectives'] = { }
        definition.options['remove_features_by_p_thresholds'] = True
        definition.options['remove_objectives_by_p_thresholds'] = True
        exds = definition.create_exds()
        exds.build()
        self.assertExDsDimensions(exds, 25, 75, 4, 4)
        self.assertFeaturesNotInExDs(exds, expected_features_to_remove.keys())


    def test_feature_removal__thresholds_on_train(self):
        folder = str(self.ensure_empty_tmp_subfolder('test_binary_exds_repository__test_feature_removal__train'))
        definition = self.default_exds_definition(folder)

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
        definition.options['probability_thresholds__objectives'] = { }
        exds = definition.create_exds()
        exds.build()
        self.assertExDsDimensions(exds, 25, 75, 8, 4)

        # Only when analysing the full matrix should we see features to be
        # removed, since we specified thresholds only for 'train'.
        expected_features_to_remove = {
                3: 'galaxy',
                4: 'oxygen',
                5: 'polyrhythm',
                6: 'python',
                7: 'rocket'
                }
        self.assertThresholdedFeaturesToRemove(exds, expected_features_to_remove, ['train'])
        self.assertThresholdedObjectivesToRemove(exds, {}, [])

        # Now we rebuild the exds, but with feature removal enabled.
        folder = str(self.ensure_empty_tmp_subfolder('test_binary_exds_repository__test_feature_removal__train'))
        definition = self.default_exds_definition(folder)
        definition.options['probability_thresholds__features'] = {
                'train': (0.1, 0.8)
                }
        definition.options['probability_thresholds__objectives'] = { }
        definition.options['remove_features_by_p_thresholds'] = True
        definition.options['remove_objectives_by_p_thresholds'] = True
        exds = definition.create_exds()
        exds.build()
        self.assertExDsDimensions(exds, 25, 75, 3, 4)
        self.assertFeaturesNotInExDs(exds, expected_features_to_remove.keys())


    def test_feature_removal__thresholds_on_test(self):
        folder = str(self.ensure_empty_tmp_subfolder('test_binary_exds_repository__test_feature_removal__test'))
        definition = self.default_exds_definition(folder)

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
        definition.options['probability_thresholds__objectives'] = { }
        exds = definition.create_exds()
        exds.build()
        self.assertExDsDimensions(exds, 25, 75, 8, 4)

        # Only when analysing the full matrix should we see features to be
        # removed, since we specified thresholds only for 'test'.
        expected_features_to_remove = {
                3: 'galaxy',
                4: 'oxygen',
                6: 'python'
                }
        self.assertThresholdedFeaturesToRemove(exds, expected_features_to_remove, ['test'])
        self.assertThresholdedObjectivesToRemove(exds, {}, [])

        # Now we rebuild the exds, but with feature removal enabled.
        folder = str(self.ensure_empty_tmp_subfolder('test_binary_exds_repository__test_feature_removal__test'))
        definition = self.default_exds_definition(folder)
        definition.options['probability_thresholds__features'] = {
                'test': (0.1, 0.9)
                }
        definition.options['probability_thresholds__objectives'] = { }
        definition.options['remove_features_by_p_thresholds'] = True
        definition.options['remove_objectives_by_p_thresholds'] = True
        exds = definition.create_exds()
        exds.build()
        self.assertExDsDimensions(exds, 25, 75, 5, 4)
        self.assertFeaturesNotInExDs(exds, expected_features_to_remove.keys())



    def test_objective_removal__thresholds_on_full(self):
        folder = str(self.ensure_empty_tmp_subfolder('test_binary_exds_repository__test_objective_removal__full'))
        definition = self.default_exds_definition(folder)

        # First, build the exds WITHOUT removing objectives. We will inspect what objectives
        # will be chosen for removal by calling the internal method
        # BinaryExperimentalDataset.determine_thresholded_objectives_to_remove().
        definition.options['remove_features_by_p_thresholds'] = False
        definition.options['remove_objectives_by_p_thresholds'] = False
        # But we do configure thresholds for objectives, to be able to verify
        # what the exds would remove, if allowed to.
        definition.options['probability_thresholds__features'] = { }
        definition.options['probability_thresholds__objectives'] = {
                'full': (0.1, 0.9)
                }
        exds = definition.create_exds()
        exds.build()
        self.assertExDsDimensions(exds, 25, 75, 8, 4)

        # Only when analysing the full matrix should we see objectives to be
        # removed, since we specified thresholds only for 'full'.
        expected_objectives_to_remove = {
                2: 'sidereal',
                3: 'unknown'
                }
        self.assertThresholdedFeaturesToRemove(exds, {}, [])
        self.assertThresholdedObjectivesToRemove(exds, expected_objectives_to_remove, ['full'])

        # Now we rebuild the exds, but with feature removal enabled.
        folder = str(self.ensure_empty_tmp_subfolder('test_binary_exds_repository__test_objective_removal__full'))
        definition = self.default_exds_definition(folder)
        definition.options['probability_thresholds__features'] = { }
        definition.options['probability_thresholds__objectives'] = {
                'full': (0.1, 0.9)
                }
        definition.options['remove_features_by_p_thresholds'] = True
        definition.options['remove_objectives_by_p_thresholds'] = True
        exds = definition.create_exds()
        exds.build()
        self.assertExDsDimensions(exds, 25, 75, 8, 2)
        self.assertObjectivesNotInExDs(exds, expected_objectives_to_remove.keys())


    def test_objective_removal__thresholds_on_train(self):
        folder = str(self.ensure_empty_tmp_subfolder('test_binary_exds_repository__test_objective_removal__train'))
        definition = self.default_exds_definition(folder)

        # First, build the exds WITHOUT removing objectives. We will inspect what objectives
        # will be chosen for removal by calling the internal method
        # BinaryExperimentalDataset.determine_thresholded_objectives_to_remove().
        definition.options['remove_features_by_p_thresholds'] = False
        definition.options['remove_objectives_by_p_thresholds'] = False
        # But we do configure thresholds for objectives, to be able to verify
        # what the exds would remove, if allowed to.
        definition.options['probability_thresholds__features'] = { }
        definition.options['probability_thresholds__objectives'] = {
                'train': (0.2, 0.8)
                }
        exds = definition.create_exds()
        exds.build()
        self.assertExDsDimensions(exds, 25, 75, 8, 4)

        # Only when analysing the full matrix should we see objectives to be
        # removed, since we specified thresholds only for 'train'.
        expected_objectives_to_remove = {
                1: 'encoded',
                2: 'sidereal',
                3: 'unknown'
                }
        self.assertThresholdedFeaturesToRemove(exds, {}, [])
        self.assertThresholdedObjectivesToRemove(exds, expected_objectives_to_remove, ['train'])

        # Now we rebuild the exds, but with feature removal enabled.
        folder = str(self.ensure_empty_tmp_subfolder('test_binary_exds_repository__test_objective_removal__train'))
        definition = self.default_exds_definition(folder)
        definition.options['probability_thresholds__features'] = { }
        definition.options['probability_thresholds__objectives'] = {
                'train': (0.2, 0.8)
                }
        definition.options['remove_features_by_p_thresholds'] = True
        definition.options['remove_objectives_by_p_thresholds'] = True
        exds = definition.create_exds()
        exds.build()
        self.assertExDsDimensions(exds, 25, 75, 8, 1)
        self.assertObjectivesNotInExDs(exds, expected_objectives_to_remove.keys())


    def test_objective_removal__thresholds_on_test(self):
        folder = str(self.ensure_empty_tmp_subfolder('test_binary_exds_repository__test_objective_removal__test'))
        definition = self.default_exds_definition(folder)

        # First, build the exds WITHOUT removing objectives. We will inspect what objectives
        # will be chosen for removal by calling the internal method
        # BinaryExperimentalDataset.determine_thresholded_objectives_to_remove().
        definition.options['remove_features_by_p_thresholds'] = False
        definition.options['remove_objectives_by_p_thresholds'] = False
        # But we do configure thresholds for objectives, to be able to verify
        # what the exds would remove, if allowed to.
        definition.options['probability_thresholds__features'] = { }
        definition.options['probability_thresholds__objectives'] = {
                'test': (0.0, 0.5)
                }
        exds = definition.create_exds()
        exds.build()
        self.assertExDsDimensions(exds, 25, 75, 8, 4)

        # Only when analysing the full matrix should we see objectives to be
        # removed, since we specified thresholds only for 'train'.
        expected_objectives_to_remove = {
                2: 'sidereal'
                }
        self.assertThresholdedFeaturesToRemove(exds, {}, [])
        self.assertThresholdedObjectivesToRemove(exds, expected_objectives_to_remove, ['test'])

        # Now we rebuild the exds, but with feature removal enabled.
        folder = str(self.ensure_empty_tmp_subfolder('test_binary_exds_repository__test_objective_removal__test'))
        definition = self.default_exds_definition(folder)
        definition.options['probability_thresholds__features'] = { }
        definition.options['probability_thresholds__objectives'] = {
                'test': (0.0, 0.5)
                }
        definition.options['remove_features_by_p_thresholds'] = True
        definition.options['remove_objectives_by_p_thresholds'] = True
        exds = definition.create_exds()
        exds.build()
        self.assertExDsDimensions(exds, 25, 75, 8, 3)
        self.assertObjectivesNotInExDs(exds, expected_objectives_to_remove.keys())


    def assertFeaturesNotInExDs(self, exds, feature_labels):
        for feature_label in feature_labels:
            self.assertFeatureNotInExDs(exds, feature_label)


    def assertObjectivesNotInExDs(self, exds, objective_labels):
        for objective_label in objective_labels:
            self.assertObjectiveNotInExDs(exds, objective_label)

    def assertFeatureNotInExDs(self, exds, feature_label):
        self.assertNotIn(feature_label, exds.matrix.column_labels_X)
        self.assertNotIn(feature_label, exds.matrix_train.column_labels_X)
        self.assertNotIn(feature_label, exds.matrix_test.column_labels_X)


    def assertObjectiveNotInExDs(self, exds, objective_label):
        self.assertNotIn(objective_label, exds.matrix.column_labels_Y)
        self.assertNotIn(objective_label, exds.matrix_train.column_labels_Y)
        self.assertNotIn(objective_label, exds.matrix_test.column_labels_Y)

    def assertExDsDimensions(self, exds, train_row_count, test_row_count, feature_count, objective_count):
        total_row_count = train_row_count + test_row_count
        self.assertEqual((total_row_count, feature_count), exds.matrix.X.get_shape())
        self.assertEqual((total_row_count, objective_count), exds.matrix.Y.get_shape())
        self.assertEqual((train_row_count, feature_count), exds.matrix_train.X.get_shape())
        self.assertEqual((train_row_count, objective_count), exds.matrix_train.Y.get_shape())
        self.assertEqual((test_row_count, feature_count), exds.matrix_test.X.get_shape())
        self.assertEqual((test_row_count, objective_count), exds.matrix_test.Y.get_shape())
        self.assertEqual(feature_count, len(exds.matrix.column_labels_X))
        self.assertEqual(feature_count, len(exds.matrix_train.column_labels_X))
        self.assertEqual(feature_count, len(exds.matrix_test.column_labels_X))
        self.assertEqual(objective_count, len(exds.matrix.column_labels_Y))
        self.assertEqual(objective_count, len(exds.matrix_train.column_labels_Y))
        self.assertEqual(objective_count, len(exds.matrix_test.column_labels_Y))


    def assertThresholdedFeaturesToRemove(self, exds, expected_features_to_remove, matrices):
        all_matrices = ['full', 'train', 'test']
        for matrix_label in all_matrices:
            if matrix_label in matrices:
                expected = expected_features_to_remove
            else:
                expected = { }
            computed = exds.thresholded_features_to_remove(matrix_label)
            self.assertDictEqual(expected, computed)


    def assertThresholdedObjectivesToRemove(self, exds, expected_objectives_to_remove, matrices):
        all_matrices = ['full', 'train', 'test']
        for matrix_label in all_matrices:
            if matrix_label in matrices:
                expected = expected_objectives_to_remove
            else:
                expected = { }
            computed = exds.thresholded_objectives_to_remove(matrix_label)
            self.assertDictEqual(expected, computed)


    def compute_counts_per_feature_columns(self, datasetmatrix):
        # Count how many values of 1 are there on each feature column.
        computed_counts_per_feature = {}
        row_count = datasetmatrix.X.get_shape()[0]
        for feature_index, feature_label in enumerate(datasetmatrix.column_labels_X):
            feature_column = datasetmatrix.get_column_X(feature_index)
            computed_counts_per_feature[feature_label] = numpy.sum(feature_column) / row_count
        return computed_counts_per_feature


    def compute_counts_per_objective_columns(self, datasetmatrix):
        # Count how many values of 1 are there on each objective column.
        computed_counts_per_objective = {}
        row_count = datasetmatrix.X.get_shape()[0]
        for objective_index, objective_label in enumerate(datasetmatrix.column_labels_Y):
            objective_column = datasetmatrix.get_column_Y(objective_index)
            computed_counts_per_objective[objective_label] = numpy.sum(objective_column) / row_count
        return computed_counts_per_objective


    def default_exds_definition(self, exds_folder):
        definition = ExperimentalDatasetDefinition(exds_folder, "test_binary_exds")
        definition.exds_class = BinaryExperimentalDataset
        definition.source = BinarySyntheticDatasetSource
        definition.source_configuration = self.default_binarysyntheticdatasetsource_configuration()
        definition.training_subset_size = 0.25
        definition.random_seed = 42
        definition.options['probability_thresholds__features'] = {}
        definition.options['probability_thresholds__objectives'] = {}
        definition.auto_lock_after_build = True
        definition.tags = []

        return definition


    def default_binarysyntheticdatasetsource_configuration(self):
        # The feature labels used here have nothing to do with the words
        # used by the TestRCV1v2DatasetSource class. They are the same words
        # indeed, but there is no real connection.
        configuration = {
                'random_seed': 42,
                'row_count': 100,
                'features': {
                    'galaxy':       1/20,
                    'almond':       1/5,
                    'python':       0/1,
                    'rocket':       1/10,
                    'carbohydrate': 1/5,
                    'oxygen':       1/1,
                    'polyrhythm':   85/100,
                    'firefly':      1/5
                },
                'objectives': {
                    'arboreal':     1/5,
                    'encoded':      1/10,
                    'sidereal':     1/1,
                    'unknown':      0/1
                }
            }
        return configuration


    def print_exds_probabilities(self, exds):
        print()
        print()
        print("Full X")
        print(self.compute_counts_per_feature_columns(exds.matrix))
        print("Full Y")
        print(self.compute_counts_per_objective_columns(exds.matrix))
        print()
        print("Train X")
        print(self.compute_counts_per_feature_columns(exds.matrix_train))
        print("Train Y")
        print(self.compute_counts_per_objective_columns(exds.matrix_train))
        print()
        print("Test X")
        print(self.compute_counts_per_feature_columns(exds.matrix_test))
        print("Test Y")
        print(self.compute_counts_per_objective_columns(exds.matrix_test))
        print()

