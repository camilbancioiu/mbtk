import numpy
import scipy

from mbtk.dataset.DatasetMatrix import DatasetMatrix


class DatasetSource:
    """
    Base class for all dataset sources. It doesn't provide any useful
    functionality, but exists for illustrative purposes and as a starting point
    for creating new DatasetSource classes.
    """

    def __init__(self, configuration):
        """
        Configure this instance and store ``configuration`` on ``self``.
        """
        self.configuration = configuration
        self.sourcefolder = self.configuration['sourcefolder']

    def create_dataset_matrix(self, label='datasetsource'):
        """
        The required method of a :py:class:`DatasetSource` class. This method
        reads an external source of data and produces a
        :py:class:`DatasetMatrix <mbtk.dataset.DatasetMatrix.DatasetMatrix>`
        instance based on ``configuration``.

        :param str label: The label of the ``DatasetMatrix``
        """
        datasetmatrix = DatasetMatrix(label)
        datasetmatrix.X = scipy.sparse.csr_matrix(numpy.identity(8))
        datasetmatrix.Y = scipy.sparse.csr_matrix(numpy.identity(8))
        datasetmatrix.row_labels = ["row{}".format(r) for r in range(8)]
        datasetmatrix.column_labels_X = ["colX{}".format(c) for c in range(8)]
        datasetmatrix.column_labels_Y = ["colY{}".format(c) for c in range(8)]
        datasetmatrix.metadata['source'] = self

        return datasetmatrix

