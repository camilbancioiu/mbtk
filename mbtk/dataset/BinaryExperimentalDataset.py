import numpy

from mbff.dataset.ModelBuildingExperimentalDataset import ModelBuildingExperimentalDataset

class BinaryExperimentalDataset(ModelBuildingExperimentalDataset):
    """
    An experimental dataset which contains only binary data. This means that
    the three :py:class:`DatasetMatrix
    <mbff.dataset.DatasetMatrix.DatasetMatrix>` instances (``self.matrix``,
    ``self.matrix_train``, ``self.matrix_test``) are treated as if the contain
    only values ``0`` and ``1``.

    The :py:class:`BinaryExperimentalDataset` class provides an extra
    processing step to the default splitting operation performed by
    :py:meth:`ModelBuildingExperimentalDataset.build()
    <mbff.dataset.ExperimentalDataset.ModelBuildingExperimentalDataset.build>`.
    """

    def perform_random_dataset_split(self):
        self.process_before_split()
        super().perform_random_dataset_split()
        self.process_after_split()


    def process_before_split(self):
        if self.definition.options.get('remove_features_by_p_thresholds', False) == True:
            features_to_remove = self.thresholded_features_to_remove('full').keys()
            self.matrix.delete_columns_X(features_to_remove)

        if self.definition.options.get('remove_objectives_by_p_thresholds', False) == True:
            objectives_to_remove = self.thresholded_objectives_to_remove('full').keys()
            self.matrix.delete_columns_Y(objectives_to_remove)


    def process_after_split(self):
        if self.definition.options.get('remove_features_by_p_thresholds', False) == True:
            features_to_remove__train = set(self.thresholded_features_to_remove('train').keys())
            features_to_remove__test = set(self.thresholded_features_to_remove('test').keys())
            features_to_remove = features_to_remove__train | features_to_remove__test
            self.matrix.delete_columns_X(features_to_remove)
            self.matrix_train.delete_columns_X(features_to_remove)
            self.matrix_test.delete_columns_X(features_to_remove)

        if self.definition.options.get('remove_objectives_by_p_thresholds', False) == True:
            objectives_to_remove__train = set(self.thresholded_objectives_to_remove('train').keys())
            objectives_to_remove__test = set(self.thresholded_objectives_to_remove('test').keys())
            objectives_to_remove = objectives_to_remove__train | objectives_to_remove__test
            self.matrix.delete_columns_Y(objectives_to_remove)
            self.matrix_train.delete_columns_Y(objectives_to_remove)
            self.matrix_test.delete_columns_Y(objectives_to_remove)


    def thresholded_features_to_remove(self, matrix_label):
        datasetmatrix = self.get_datasetmatrix(matrix_label)
        try:
            thresholds = self.definition.options['probability_thresholds__features'][matrix_label]
        except KeyError:
            return { }

        return self.thresholded_columns_to_remove(datasetmatrix, 'X', thresholds)


    def thresholded_objectives_to_remove(self, matrix_label):
        datasetmatrix = self.get_datasetmatrix(matrix_label)
        try:
            thresholds = self.definition.options['probability_thresholds__objectives'][matrix_label]
        except KeyError:
            return { }

        return self.thresholded_columns_to_remove(datasetmatrix, 'Y', thresholds)


    def thresholded_columns_to_remove(self, datasetmatrix, matrix_label, thresholds):
        (pmin, pmax) = thresholds
        (row_count, column_count) = datasetmatrix.get_matrix(matrix_label).get_shape()
        column_labels = datasetmatrix.get_column_labels(matrix_label)
        columns_to_remove = {}
        for c in range(column_count):
            column = datasetmatrix.get_column(matrix_label, c)
            p = numpy.sum(column) / row_count
            if p < pmin or p > pmax:
                columns_to_remove[c] = column_labels[c]

        return columns_to_remove


