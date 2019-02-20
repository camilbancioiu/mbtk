import random
import os
import shutil
from pathlib import Path

import mbff.utilities as util
from mbff.datasets.DatasetMatrix import DatasetMatrix

class ExperimentalDataset():
    """
    This class represents an experimental dataset in its entirety. It initially
    contains a ``DatasetMatrix`` (the ``matrix`` property) generated by a
    dataset source (according to the provided ``definition``), but which is
    subsequently split into *training samples* (``matrix_train``) and *test
    samples* (``matrix_test``), to be used when evaluating inductive models
    such as classifiers.

    :var definition: An instance of ``ExperimentalDatasetDefinition``, which
        contains the details about how to build the initial ``matrix`` from
        source, how to split it into *training samples* and *test samples* and
        the folder where to save / to load the matrices from.
    :var matrix: The initial instance of ``DatasetMatrix``, as generated by a
        dataset source. Contains all the samples, the features and objective
        variables, along with their labels.
    :var matrix_train: A smaller ``DatasetMatrix`` instance, which contains a random
        selection of rows taken from ``matrix``, to be used to train an
        inductive model. The parameters that govern this selection of rows are
        found in ``definition``.
    :var matrix_test: A smaller ``DatasetMatrix`` instance, which contains a random
        selection of rows taken from ``matrix``, to be used to evaluate an
        inductive model. The parameters that govern this selection of rows are
        found in ``definition``.
    :var total_row_count: The number of rows in ``matrix`` and also the sum of
        the numbers of rows in ``matrix_train`` and ``matrix_test``.
    :var train_rows: A list of row indices, selected from ``matrix`` to be part
        of ``matrix_train``. Useful to keep track of the train/test split.
    :var test_rows: A list of row indices, selected from ``matrix`` to be part
        of ``matrix_test``. Useful to keep track of the train/test split.
    """
    def __init__(self, definition):
        self.definition = definition
        self.matrix = None
        self.matrix_train = None
        self.matrix_test = None
        self.total_row_count = 0
        self.train_rows = None
        self.test_rows = None


    def build(self, finalize_and_save=True):
        """
        Build an ExperimentalDataset from an external source, as dictated by
        ``definition``.
        """
        datasetsource = self.definition.source(self.definition.source_configuration)
        self.matrix = datasetsource.create_dataset_matrix("dataset_full")
        self.total_row_count = self.matrix.X.get_shape()[0]

        self.process_before_split()

        # Create self.matrix_train and self.matrix_test from self.matrix
        (self.train_rows, self.test_rows) = self.perform_random_dataset_split()
        self.matrix_train = self.matrix.select_rows(self.train_rows, "dataset_train")
        self.matrix_test = self.matrix.select_rows(self.test_rows, "dataset_test")

        self.process_after_split()

        if finalize_and_save:
            self.process_before_finalize_and_save()
            self.finalize_and_save()


    def get_datasetmatrix(self, label):
        if label == 'full':
            return self.matrix
        elif label == 'train':
            return self.matrix_train
        elif label == 'test':
            return self.matrix_test
        else:
            raise ValueError("Unknown DatasetMartix label. Only 'full', 'train' and 'test' are allowed.")


    def process_before_split(self):
        pass


    def process_after_split(self):
        pass


    def process_before_finalize_and_save(self):
        pass


    def perform_random_dataset_split(self):
        # Determine how many rows will be designated as *training rows*.
        train_rows_count = int(self.total_row_count * self.definition.training_subset_size)

        # Create the ``shuffled_rows`` list, which contains the indices of all
        # the rows from ``self.matrix``, but randomly ordered.
        rows = range(self.total_row_count)
        random.seed(self.definition.random_seed)
        shuffled_rows = random.sample(rows, len(rows))

        # Slice ``shuffled_rows`` into two parts: the first part will contain
        # the indices of the *training rows*, while the second part will
        # contain the indices of the *test rows*.
        train_rows = sorted(shuffled_rows[0:train_rows_count])
        test_rows = sorted(shuffled_rows[train_rows_count:])

        return train_rows, test_rows


    def finalize_and_save(self):
        # Finalize all 3 matrices.
        self.matrix.finalize()
        self.matrix_train.finalize()
        self.matrix_test.finalize()

        # Save and optionally lock this dataset
        self.save()
        if self.definition.auto_lock_after_build:
            self.definition.lock_folder()


    def save(self):
        util.ensure_folder(self.definition.folder)

        if not self.definition.folder_is_locked():
            self.matrix.save(self.definition.folder)
            self.matrix_train.save(self.definition.folder)
            self.matrix_test.save(self.definition.folder)
        else:
            raise ExperimentalDatasetError(self.definition, "Cannot save - ExDs is folder locked.")


    def load(self):
        if self.matrix is None:
            self.matrix = DatasetMatrix("dataset_full")
        self.matrix.load(self.definition.folder)

        if self.matrix_train is None:
            self.matrix_train = DatasetMatrix("dataset_train")
        self.matrix_train.load(self.definition.folder)

        if self.matrix_test is None:
            self.matrix_test = DatasetMatrix("dataset_test")
        self.matrix_test.load(self.definition.folder)



class ExperimentalDatasetError(Exception):
    def __init__(self, exds_definition, message):
        self.exds_definition = exds_definition
        self.message = message
        super().__init__(self.message)


