import random
import os
import shutil
from pathlib import Path

import mbff.utilities.functions as util
from mbff.dataset.DatasetMatrix import DatasetMatrix

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
        ``definition``. This is where we instantiate the class that will read
        an external source of data for us, and give back a DatasetMatrix
        instance containing the samples as rows, features as columns of its
        ``X`` matrix and objective variables as its ``Y`` matrix. The class
        that is to be instantiated as the dataset source is specified by
        ``self.definition.source``, and its __init__() parameters are provided
        in ``self.definition.source_configuration``. After retrieval, the
        DatasetMatrix containing the full dataset will be stored in
        ``self.matrix``. The Feature Selection algorithms will only require
        this DatasetMatrix.

        After retrieving the full dataset from the external source, we perform
        a random split of the samples into two subsets: the training set and
        the testing (evaluation) set, which will later be used to train and
        evaluate learning algorithm (usually a classifier). They will be stored
        as two separate DatasetMatrix instances, in ``self.matrix_train`` and
        ``self.matrix_test``. Together, they can rebuild the original full
        dataset from ``self.matrix``. The Feature Selection algorithms will not
        normally be interested in these two submatrices.

        Classes that inherit ExperimentalDataset may also intervene in the
        build process, by overriding the following methods:

        * process_before_split
        * process_after_split
        * process_before_finalize_and_save
        """
        # Retrieve the full dataset from the external data source, which is
        # done by instantiating the class provided as ``source`` in
        # self.definition. This class should be inheriting the
        # mbff.dataset.sources.DatasetSource class, or at least implement the
        # ``create_dataset_matrix(label)`` method.
        datasetsource_class = self.definition.source
        datasetsource = datasetsource_class(self.definition.source_configuration)
        self.matrix = datasetsource.create_dataset_matrix("dataset_full")

        self.total_row_count = self.matrix.X.get_shape()[0]

        # Allow inheriting classes to intervene before the full dataset is
        # about to be split into training and testing samples, yet immediately
        # after the full dataset has been built from the external source.
        self.process_before_split()

        # Create self.matrix_train and self.matrix_test from self.matrix
        (self.train_rows, self.test_rows) = self.perform_random_dataset_split()
        self.matrix_train = self.matrix.select_rows(self.train_rows, "dataset_train")
        self.matrix_test = self.matrix.select_rows(self.test_rows, "dataset_test")

        self.process_after_split()

        if finalize_and_save:
            self.process_before_finalize_and_save()
            self.finalize()
            self.save()
            if self.definition.auto_lock_after_build:
                self.definition.lock_folder()


    def get_datasetmatrix(self, label):
        """
        Return the DatasetMatrix matrix specified by ``label``.

        The only allowed values for ``label`` are ``"full"``, ``"train"`` and
        ``"test"``, which return the full dataset, the training dataset and the
        test dataset respectively. A ValueError is raised for other values.
        """
        if label == 'full':
            return self.matrix
        elif label == 'train':
            return self.matrix_train
        elif label == 'test':
            return self.matrix_test
        else:
            raise ValueError("Unknown DatasetMartix label. Only 'full', 'train' and 'test' are allowed.")


    def process_before_split(self):
        """
        Called after retrieving data from a dataset source, but before
        splitting the dataset into training and testing subsets.

        Classes inheriting ``ExperimentalDataset`` may implement this method to
        intervene at that point.
        """
        pass


    def process_after_split(self):
        """
        Called after splitting the dataset into training and testing subsets,
        but before finalizing and saving the ExperimentalDataset. This method
        is called regardless of the value of the argument
        ``finalize_and_save``, received by the ``build()`` method.

        Classes inheriting ``ExperimentalDataset`` may implement this method to
        intervene at that point.
        """
        pass


    def process_before_finalize_and_save(self):
        """
        Called just before finalizing and saving the ExperimentalDataset. Only
        called if the argument ``finalize_and_save`` received by the
        ``build()`` method evaluates to ``True``.
        """
        pass


    def perform_random_dataset_split(self):
        """
        Decide which rows of the full dataset ``self.matrix`` will be selected
        for the training dataset and which rows for the testing dataset.

        All rows of ``self.matrix`` will be decided on, which means that the
        union between the training rows and the testing rows will result in the
        original dataset.

        The selection is performed randomly, but the seed of the randomness is
        controlled by ``self.definition.random_seed``, which makes this
        selection predictable. This means that for the same seed value, all
        calls to ``self.perform_random_dataset_split()`` will return the same
        selection of training and testing rows.

        Returns a tuple which contains
        * the list of indices of the training rows
        * the list of indices of the testing rows
        """
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


    def finalize(self):
        """
        Finalize the 3 matrices.
        """
        self.matrix.finalize()
        self.matrix_train.finalize()
        self.matrix_test.finalize()


    def save(self):
        """
        Save this ExperimentalDataset to ``self.definition.folder``.
        """
        self.definition.ensure_folder()

        if not self.definition.folder_is_locked():
            self.matrix.save(self.definition.path)
            self.matrix_train.save(self.definition.path)
            self.matrix_test.save(self.definition.path)
        else:
            raise ExperimentalDatasetError(self.definition, "Cannot save - ExDs folder is locked.")


    def load(self):
        """
        Load the ExperimentalDataset from ``self.definition.folder``.
        """
        if self.matrix is None:
            self.matrix = DatasetMatrix("dataset_full")
        self.matrix.load(self.definition.path)

        if self.matrix_train is None:
            self.matrix_train = DatasetMatrix("dataset_train")
        self.matrix_train.load(self.definition.path)

        if self.matrix_test is None:
            self.matrix_test = DatasetMatrix("dataset_test")
        self.matrix_test.load(self.definition.path)



class ExperimentalDatasetError(Exception):
    def __init__(self, exds_definition, message):
        self.exds_definition = exds_definition
        self.message = message
        super().__init__(self.message)


