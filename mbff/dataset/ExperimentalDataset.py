from mbff.dataset.DatasetMatrix import DatasetMatrix
from mbff.dataset.Exceptions import ExperimentalDatasetError


class ExperimentalDataset():
    """
    This class represents an experimental dataset in its entirety. Upon calling
    :py:meth:`build`, the ``self.matrix`` attribute will contain a
    :py:class:`DatasetMatrix <mbff.dataset.DatasetMatrix>` instance, as
    generated by a dataset source according to the provided
    ``self.definition``. The ``self.definition`` itself is an instance of
    :py:class:`ExperimentalDatasetDefinition
    <mbff.dataset.ExperimentalDatasetDefinition.ExperimentalDatasetDefinition>`.

    An instance of :py:class:`ExperimentalDataset` holds the following attributes:

    :var definition: An instance of :py:class:`ExperimentalDatasetDefinition
        <mbff.dataset.ExperimentalDatasetDefinition.ExperimentalDatasetDefinition>`,
        which contains the details about how to build the initial ``self.matrix`` from
        a dataset source, how to split it into *training samples* and *test
        samples* and the folder where to save / to load the matrices from.
    :var matrix: The instance of ``DatasetMatrix`` generated by a
        dataset source during :py:meth:`build`. Contains all the samples, the
        features and objective variables, along with their labels.
    :var total_row_count: The number of rows in ``self.matrix``.
    """
    def __init__(self, definition):
        self.definition = definition
        self.matrix = None
        self.total_row_count = 0


    def build(self, finalize_and_save=None):
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

        Classes that inherit ExperimentalDataset may also intervene in the
        build process, by overriding the following methods:

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
        self.matrix = datasetsource.create_dataset_matrix("dataset")

        self.total_row_count = self.matrix.X.get_shape()[0]

        if finalize_and_save is None:
            finalize_and_save = self.definition.after_build__finalize_and_save

        if finalize_and_save:
            self.finalize_and_save()


    def finalize_and_save(self):
        self.finalize()
        self.save()
        if self.definition.after_save__auto_lock:
            self.definition.lock_folder()


    def finalize(self):
        """
        Finalize the 3 matrices. Simply calls :py:meth:`finalize()
        <mbff.dataset.DatasetMatrix.DatasetMatrix.finalize>` on
        ``self.matrix``, ``self.matrix_train`` and ``self.matrix_test``.

        :return: Nothing
        """
        self.matrix.finalize()


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
            self.matrix = DatasetMatrix("dataset")
        self.matrix.load(self.definition.path)


    def info(self):
        return self.matrix.info()
