import random
import os
import shutil
from pathlib import Path

import mbtk.utilities.functions as util
from mbtk.dataset.ExperimentalDataset import ExperimentalDataset
from mbtk.dataset.DatasetMatrix import DatasetMatrix


class ModelBuildingExperimentalDataset(ExperimentalDataset):
    """
    This class is a variation of :py:class:`ExperimentalDataset`, which
    subsequently splits the ``DatasetMatrix`` into *training samples*
    (``self.matrix_train``) and *test samples* (``self.matrix_test``), to be
    used when evaluating inductive models such as classifiers.

    An instance of :py:class:`ModelBuildingExperimentalDataset` holds the
    following attributes:

    :var definition: An instance of :py:class:`ExperimentalDatasetDefinition
        <mbtk.dataset.ExperimentalDatasetDefinition.ExperimentalDatasetDefinition>`,
        which contains the details about how to build the initial ``self.matrix`` from
        a dataset source, how to split it into *training samples* and *test
        samples* and the folder where to save / to load the matrices from.
    :var matrix: The instance of ``DatasetMatrix`` generated by a
        dataset source during :py:meth:`build`. Contains all the samples, the
        features and objective variables, along with their labels.
    :var matrix_train: A smaller ``DatasetMatrix`` instance, which contains a random
        selection of rows taken from ``self.matrix``, to be used to train an
        inductive model. The parameters that govern this selection of rows are
        found in ``self.definition``.
    :var matrix_test: A smaller ``DatasetMatrix`` instance, which contains a random
        selection of rows taken from ``self.matrix``, to be used to evaluate an
        inductive model. The parameters that govern this selection of rows are
        found in ``self.definition``.
    :var total_row_count: The number of rows in ``self.matrix`` and also the sum of
        the numbers of rows in ``self.matrix_train`` and ``self.matrix_test``.
    :var train_rows: A list of row indices, selected from ``self.matrix`` to be part
        of ``self.matrix_train``. Useful to keep track of the train/test split.
    :var test_rows: A list of row indices, selected from ``self.matrix`` to be part
        of ``self.matrix_test``. Useful to keep track of the train/test split.
    """

    def __init__(self, definition):
        super().__init__(definition)
        self.matrix_train = None
        self.matrix_test = None
        self.train_rows = None
        self.test_rows = None


    def build(self, finalize_and_save=True):
        """
        After retrieving the full dataset from the external source (see
        :py:meth:`ExperimentalDataset.build`), we perform a random split of the
        samples into two subsets: the training set and the testing (evaluation)
        set, which will later be used to train and evaluate learning algorithm
        (usually a classifier). They will be stored as two separate
        ``DatasetMatrix`` instances, in ``self.matrix_train`` and
        ``self.matrix_test``. Together, they can rebuild the original full
        dataset from ``self.matrix``. The Feature Selection algorithms will
        normally not be interested in these two submatrices.

        :return: Nothing
        """
        # Don't finalize and save just yet. Build the main matrix, but we'll
        # finalize and save after building self.matrix_train and
        # self.matrix_test.
        super().build(finalize_and_save=False)

        self.perform_random_dataset_split()

        if finalize_and_save:
            self.finalize_and_save()


    def finalize(self):
        super().finalize()
        self.matrix_train.finalize()
        self.matrix_test.finalize()


    def save(self):
        """
        Save the training and testing matrices of this
        ModelBuildingExperimentalDataset to ``self.definition.path``. Simply
        calls :py:meth:`save() <mbtk.dataset.DatasetMatrix.DatasetMatrix.save>`
        on ``self.matrix_train`` and ``self.matrix_test``.

        :return: Nothing
        :raises ExperimentalDatasetError: if the ExperimentalDataset folder is locked
        """
        super().save()

        self.definition.ensure_folder()

        if not self.definition.folder_is_locked():
            self.matrix_train.save(self.definition.path)
            self.matrix_test.save(self.definition.path)
        else:
            raise ExperimentalDatasetError(self.definition, "Cannot save - ExDs folder is locked.")


    def load(self):
        super().load()
        if self.matrix_train is None:
            self.matrix_train = DatasetMatrix("dataset_train")
        self.matrix_train.load(self.definition.path)

        if self.matrix_test is None:
            self.matrix_test = DatasetMatrix("dataset_test")
        self.matrix_test.load(self.definition.path)


    def perform_random_dataset_split(self):
        """
        Decide which rows of the full dataset ``self.matrix`` will be selected
        for the training dataset and which rows for the testing dataset, and
        create the matrices ``self.matrix_train`` and ``self.matrix_test``.

        All rows of ``self.matrix`` will be decided on, which means that the
        union between the training rows and the testing rows will result in the
        original dataset.

        The selection is performed randomly, but the seed of the randomness is
        controlled by ``self.definition.options['random_seed']``, which makes this
        selection predictable. This means that for the same seed value, all
        calls to :py:meth`perform_random_dataset_split` will return the same
        selection of training and testing rows.
        """
        # Determine how many rows will be designated as *training rows*.
        train_rows_count = int(self.total_row_count * self.definition.options['training_subset_size'])

        # Create the ``shuffled_rows`` list, which contains the indices of all
        # the rows from ``self.matrix``, but randomly ordered.
        rows = range(self.total_row_count)
        random.seed(self.definition.options['random_seed'])
        shuffled_rows = random.sample(rows, len(rows))

        # Slice ``shuffled_rows`` into two parts: the first part will contain
        # the indices of the *training rows*, while the second part will
        # contain the indices of the *test rows*.
        self.train_rows = sorted(shuffled_rows[0:train_rows_count])
        self.test_rows = sorted(shuffled_rows[train_rows_count:])

        # Create self.matrix_train and self.matrix_test from self.matrix
        self.matrix_train = self.matrix.select_rows(self.train_rows, "dataset_train")
        self.matrix_test = self.matrix.select_rows(self.test_rows, "dataset_test")


    def get_datasetmatrix(self, label):
        """
        Return the DatasetMatrix matrix specified by ``label``.

        The only allowed values for ``label`` are ``"full"``, ``"train"`` and
        ``"test"``, which return the full dataset (``self.matrix``), the
        training dataset (``self.matrix_train``) and the test dataset
        (``self.matrix_test``) respectively. A :py:class:`ValueError` is raised
        for other values of ``label``.

        :param str label: The label of the ``DatasetMatrix`` to retrieve.
        :return: The ``DatasetMatrix`` specified by ``label``, one of the following:

            * ``self.matrix`` for ``label == "full"``
            * ``self.matrix_train`` for ``label == "train"``
            * ``self.matrix_test`` for ``label == "test"``
        :rtype: mbtk.dataset.DatasetMatrix.DatasetMatrix
        :raises ValueError: if ``label`` is not one of ``"full"``, ``"train"`` or ``"test"``
        """
        if label == 'full':
            return self.matrix
        elif label == 'train':
            return self.matrix_train
        elif label == 'test':
            return self.matrix_test
        else:
            raise ValueError("Unknown DatasetMartix label. Only 'full', 'train' and 'test' are allowed.")


