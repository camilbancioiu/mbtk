import unittest

import mbff_tests.test_utilities as util

from mbff.datasets.DatasetMatrix import DatasetMatrix
from mbff.datasets.ExperimentalDatasetDefinition import ExperimentalDatasetDefinition
from mbff.datasets.ExperimentalDataset import ExperimentalDataset
from mbff.datasets.sources.RCV1v2DatasetSource import RCV1v2DatasetSource

class TestExperimentalDataset(unittest.TestCase):

    def test_exds_build(self):
        folder = str(util.ensure_empty_tmp_subfolder('test_exds_repository__test_build'))
        definition = self.default_exds_definition(folder)
        exds = ExperimentalDataset(definition)

        exds.build(finalize_and_save=False)

        # Make sure 'training_set_size = 0.25' has been properly taken into
        # account.
        self.assertEqual(16, exds.total_row_count)
        self.assertEqual(4, len(exds.train_rows))
        self.assertEqual(12, len(exds.test_rows))
        self.assertEqual(4, exds.matrix_train.X.get_shape()[0])
        self.assertEqual(4, exds.matrix_train.Y.get_shape()[0])
        self.assertEqual(12, exds.matrix_test.X.get_shape()[0])
        self.assertEqual(12, exds.matrix_test.Y.get_shape()[0])

        # Reconstruct the list of all row indices, to make sure the split is
        # consistent.
        all_rows = set(exds.train_rows) | set(exds.test_rows)
        self.assertSetEqual(set(range(16)), all_rows)
        self.assertEqual(0, len(set(exds.train_rows) & set(exds.test_rows)))

        # Ensure that any row of exds.matrix is found either in
        # exds.matrix_train or exds.matrix_test.
        # First try for X.
        for row in range(15):
            original_row = exds.matrix.X.getrow(row)
            if row in exds.train_rows:
                train_row = exds.matrix_train.X.getrow(exds.train_rows.index(row))
                self.assertTrue(DatasetMatrix.sparse_equal(original_row, train_row))
            elif row in exds.test_rows:
                test_row = exds.matrix_test.X.getrow(exds.test_rows.index(row))
                self.assertTrue(DatasetMatrix.sparse_equal(original_row, test_row))
            else:
                self.fail("Row {} not found in neither train nor test X matrices".format(row))

        # Do the same for Y.
        for row in range(15):
            original_row = exds.matrix.X.getrow(row)
            if row in exds.train_rows:
                train_row = exds.matrix_train.X.getrow(exds.train_rows.index(row))
                self.assertTrue(DatasetMatrix.sparse_equal(original_row, train_row))
            elif row in exds.test_rows:
                test_row = exds.matrix_test.X.getrow(exds.test_rows.index(row))
                self.assertTrue(DatasetMatrix.sparse_equal(original_row, test_row))
            else:
                self.fail("Row {} not found in neither train nor test Y matrices".format(row))


    def test_exds_saving_and_loading(self):
        folder = str(util.ensure_empty_tmp_subfolder('test_exds_repository__test_saving_and_loading'))
        definition = self.default_exds_definition(folder)
        exds = ExperimentalDataset(definition)

        # Due to the definition provided by self.default_exds_definition(), the
        # exds will be saved after building.
        exds.build()

        # Verify if the matrices have been finalized.
        self.assertTrue(exds.matrix.final)
        self.assertTrue(exds.matrix_train.final)
        self.assertTrue(exds.matrix_test.final)

        # Verify if the matrices can be loaded individually from the saved
        # ExperimentalDataset.
        # - The original matrix:
        loadedMatrix_original = DatasetMatrix("dataset_full")
        loadedMatrix_original.load(exds.definition.folder)
        self.assertEqual(exds.matrix, loadedMatrix_original)
        # - The training matrix:
        loadedMatrix_train = DatasetMatrix("dataset_train")
        loadedMatrix_train.load(exds.definition.folder)
        self.assertEqual(exds.matrix_train, loadedMatrix_train)
        # - The test matrix:
        loadedMatrix_test = DatasetMatrix("dataset_test")
        loadedMatrix_test.load(exds.definition.folder)
        self.assertEqual(exds.matrix_test, loadedMatrix_test)


    def default_exds_definition(self, exds_folder):
        definition = ExperimentalDatasetDefinition()
        definition.name = "test_exds_reuters"
        definition.source = RCV1v2DatasetSource
        definition.source_configuration = {
                'sourcefolder': 'testfiles/rcv1v2_test_dataset',
                'filters': {},
                'feature_type': 'binary'
                }
        definition.exds_folder = exds_folder
        definition.training_subset_size = 0.25
        definition.random_seed = 42
        definition.auto_lock_after_build = True
        definition.tags = []

        definition.setup()

        return definition

