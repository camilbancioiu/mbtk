import numpy
import scipy

from mbff.dataset.DatasetMatrix import DatasetMatrix


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
        The required method of a ``DatasetSource`` class. This method reads an
        external source of data and produces a ``DatasetMatrix`` instance based
        on ``configuration``. The newly created ``DatasetMatrix`` instance is
        then returned.
        """
        datasetmatrix = DatasetMatrix(label)
        datasetmatrix.X = scipy.sparse.csr_matrix(numpy.identity(8))
        datasetmatrix.Y = scipy.sparse.csr_matrix(numpy.identity(8))
        datasetmatrix.row_labels = ["row{}".format(r) for r in range(8)]
        datasetmatrix.column_labels_X = ["colX{}".format(c) for c in range(8)]
        datasetmatrix.column_labels_Y = ["colY{}".format(c) for c in range(8)]

        return datasetmatrix

