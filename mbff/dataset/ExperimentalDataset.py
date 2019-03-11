import random
import os
import shutil
from pathlib import Path

import mbff.utilities.functions as util
from mbff.dataset.DatasetMatrix import DatasetMatrix

class ExperimentalDataset():
    """
    This class represents an experimental dataset in its entirety. Upon calling :py:meth:`build`, the ``self.matrix`` attribute will contain
    a :py:class:`DatasetMatrix <mbff.dataset.DatasetMatrix>` instance, as
    generated by a dataset source according to the provided ``self.definition``. The
    ``self.definition`` itself is an instance of
    :py:class:`ExperimentalDatasetDefinition
    <mbff.dataset.ExperimentalDatasetDefinition.ExperimentalDatasetDefinition>`.
    The ``DatasetMatrix`` instance stored in ``self.matrix`` is subsequently
    split into *training samples* (``self.matrix_train``) and *test samples*
    (``self.matrix_test``), to be used when evaluating inductive models such as
    classifiers.

    An instance of :py:class:`ExperimentalDataset` holds the following attributes:

    :var definition: An instance of :py:class:`ExperimentalDatasetDefinition
        <mbff.dataset.ExperimentalDatasetDefinition.ExperimentalDatasetDefinition>`,
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
        self.definition = definition
        self.matrix = None
        self.matrix_train = None
        self.matrix_test = None
        self.total_row_count = 0
        self.train_rows = None
        self.test_rows = None


    def build(self, finalize_and_save=True):
        """
        Build an ExperimentalDataset from an external dataset source, as
        dictated by ``self.definition``.

        This is where we instantiate the class that will read an external
        source of data for us, and give back a :py:class:`DatasetMatrix
        <mbff.dataset.DatasetMatrix>` instance containing the samples as rows,
        features as columns of its ``X`` matrix and objective variables as its
        ``Y`` matrix. The class that is to be instantiated as the dataset
        source is specified by ``self.definition.source``, and its constructor
        parameters are provided in ``self.definition.source_configuration``.
        After retrieval, the ``DatasetMatrix`` object containing the full
        dataset will be stored in ``self.matrix``. The Feature Selection
        algorithms will only require this DatasetMatrix.

        After retrieving the full dataset from the external source, we perform
        a random split of the samples into two subsets: the training set and
        the testing (evaluation) set, which will later be used to train and
        evaluate learning algorithm (usually a classifier). They will be stored
        as two separate ``DatasetMatrix`` instances, in ``self.matrix_train``
        and ``self.matrix_test``. Together, they can rebuild the original full
        dataset from ``self.matrix``. The Feature Selection algorithms will not
        normally be interested in these two submatrices.

        Classes that inherit ExperimentalDataset may also intervene in the
        build process, by overriding the following methods:

        * :py:meth:`process_before_split`
        * :py:meth:`process_after_split`
        * :py:meth:`process_before_finalize_and_save`

        :param bool finalize_and_save: Whether to finalize and save the 3\
            matrices after completing the build.
        :return: Nothing
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
        ``"test"``, which return the full dataset (``self.matrix``), the
        training dataset (``self.matrix_train``) and the test dataset
        (``self.matrix_test``) respectively. A :py:class:`ValueError` is raised
        for other values of ``label``.

        :param str label: The label of the ``DatasetMatrix`` to retrieve.
        :return: The ``DatasetMatrix`` specified by ``label``, one of the following:

            * ``self.matrix`` for ``label == "full"``
            * ``self.matrix_train`` for ``label == "train"``
            * ``self.matrix_test`` for ``label == "test"``
        :rtype: mbff.dataset.DatasetMatrix.DatasetMatrix
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


    def process_before_split(self):
        """
        Called after retrieving data from a dataset source, but before
        splitting the dataset into training and testing subsets.

        Classes inheriting ``ExperimentalDataset`` may implement this method to
        intervene at that point.

        :return: Nothing
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

        :return: Nothing
        """
        pass


    def process_before_finalize_and_save(self):
        """
        Called just before finalizing and saving the ExperimentalDataset. Only
        called if the argument ``finalize_and_save`` received by the
        ``build()`` method evaluates to ``True``.

        Classes inheriting ``ExperimentalDataset`` may implement this method to
        intervene at that point.

        :return: Nothing
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
        calls to :py:meth`perform_random_dataset_split` will return the same
        selection of training and testing rows.

        :return: a tuple which contains two lists:

            # the list of indices of the training rows
            # the list of indices of the testing rows
        :rtype: tuple(list(int), list(int))
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

        return (train_rows, test_rows)


    def finalize(self):
        """
        Finalize the 3 matrices. Simply calls :py:meth:`finalize()
        <mbff.dataset.DatasetMatrix.DatasetMatrix.finalize>` on
        ``self.matrix``, ``self.matrix_train`` and ``self.matrix_test``.

        :return: Nothing
        """
        self.matrix.finalize()
        self.matrix_train.finalize()
        self.matrix_test.finalize()


    def save(self):
        """
        Save this ExperimentalDataset to ``self.definition.path``. Simply calls
        :py:meth:`save() <mbff.dataset.DatasetMatrix.DatasetMatrix.save>` on
        ``self.matrix``, ``self.matrix_train`` and ``self.matrix_test``.

        :return: Nothing
        :raises ExperimentalDatasetError: if the ExperimentalDataset folder is locked
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
        Load the ExperimentalDataset from ``self.definition.path``. Simply calls
        :py:meth:`load() <mbff.dataset.DatasetMatrix.DatasetMatrix.load>` on
        ``self.matrix``, ``self.matrix_train`` and ``self.matrix_test``. In
        case either of these three is ``None``, they are set to new
        :py:class:`DatasetMatrix <mbff.dataset.DatasetMatrix.DatasetMatrix>` instances.

        :return: Nothing
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


