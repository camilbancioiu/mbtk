import scipy
import numpy

from mbff_tests.TestBase import TestBase

from mbff.dataset.sources.DatasetSource import DatasetSource
from mbff.dataset.DatasetMatrix import DatasetMatrix
from mbff.dataset.ExperimentalDatasetDefinition import ExperimentalDatasetDefinition
from mbff.dataset.ExperimentalDataset import ExperimentalDataset
from mbff.experiment.AlgorithmRun import AlgorithmRun
from mbff.algorithms.basic.IGt import algorithm_IGt__binary

class TestAlgorithmRun(TestBase):

    def test_algorithm_run(self):
        folder = str(self.ensure_empty_tmp_subfolder('test_exds_repository__test_build'))

        # Prepare an ExperimentalDataset to perform the test on.
        definition = self.default_exds_definition(folder)
        exds = definition.create_exds()
        exds.build()

        # Because definition.random_seed = 42, the 'randomly' selected training
        # rows in the ExperimentalDataset will always be [0, 1, 5].

        # Prepare an AlgorithmRun instance.
        parameters = {
                'label': 'test_algrun',
                'classifier_class': MockBernouliClassifier,
                'algorithm': algorithm_IGt__binary,
                'Q': 4,
                'objective_index': 0
                }
        algrun = AlgorithmRun(exds, parameters)

        # We run the algorithm at the specified parameters, on the specified
        # ExDs.
        algrun.run()

        # Verify if the AlgorithmRun now contains the expected results.
        self.assertListEqual([0, 4, 1, 5], algrun.selected_features)
        self.assertLess(0, algrun.duration)

        expected_classifier_evaluation = {
                'TP': 1,
                'TN': 1,
                'FP': 1,
                'FN': 2
                }
        self.assertDictEqual(expected_classifier_evaluation, algrun.classifier_evaluation)


    def default_exds_definition(self, exds_folder):
        definition = ExperimentalDatasetDefinition()
        definition.name = "test_exds_algorithmrun"
        definition.exds_class = ExperimentalDataset
        definition.source = MockDatasetSource
        definition.source_configuration = {}
        definition.exds_folder = exds_folder
        definition.training_subset_size = 3/8
        definition.random_seed = 42
        definition.auto_lock_after_build = True
        definition.tags = []
        return definition


    def default_datasetmatrix_train(self):
        pass


    def default_samples_train(self):
        pass



class MockDatasetSource(DatasetSource):

    def __init__(self, configuration):
        pass


    def default_datasetmatrix(label):
        sample_count = 8
        feature_count = 8
        datasetmatrix = DatasetMatrix(label)
        datasetmatrix.row_labels = ['row{}'.format(i) for i in range(0, sample_count)]
        datasetmatrix.column_labels_X = ['feature{}'.format(i) for i in range(0, feature_count)]
        datasetmatrix.column_labels_Y = ['objective']
        datasetmatrix.Y = scipy.sparse.csr_matrix(numpy.matrix([
            [1], # training sample
            [0], # training sample
            [1], # testing sample
            [0], # testing sample
            [1], # testing sample
            [0], # training sample
            [1], # testing sample
            [0]  # testing sample
            ]))
        datasetmatrix.X = scipy.sparse.csr_matrix(numpy.matrix([
            [1, 1, 1, 1, 0, 1, 0, 1], # training sample
            [0, 1, 1, 1, 1, 0, 0, 1], # training sample
            [1, 1, 1, 0, 0, 0, 1, 0], # testing sample
            [0, 0, 1, 0, 1, 1, 1, 0], # testing sample
            [1, 1, 0, 1, 0, 0, 1, 1], # testing sample
            [0, 0, 0, 1, 1, 1, 0, 1], # training sample
            [1, 1, 1, 1, 0, 0, 1, 0], # testing sample
            [0, 0, 0, 1, 1, 1, 1, 0]  # testing sample
            ]))
        return datasetmatrix


    def create_dataset_matrix(self, label):
        return MockDatasetSource.default_datasetmatrix(label)



class MockBernouliClassifier:

    def __init__(self):
        pass


    def fit(self, samples, objective):
        pass


    def predict(self, samples):
        return numpy.array([ 1, 1, 0, 0, 0 ])



