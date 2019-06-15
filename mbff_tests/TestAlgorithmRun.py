import scipy
import numpy

from string import Template

from mbff_tests.TestBase import TestBase
from mbff_tests.MockDatasetSource import MockDatasetSource

from mbff.dataset.ExperimentalDatasetDefinition import ExperimentalDatasetDefinition
from mbff.dataset.ModelBuildingExperimentalDataset import ModelBuildingExperimentalDataset
from mbff.experiment.AlgorithmRun import AlgorithmAndClassifierRun
from mbff.algorithms.basic.IGt import algorithm_IGt__binary as IGt

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
        configuration = {
                'label': 'test_algrun',
                'classifier': MockBernouliClassifier,
                'algorithm': IGt,
                }
        parameters = {
                'Q': 4,
                'objective_index': 0
                }
        algrun = AlgorithmAndClassifierRun(exds, configuration, parameters)

        # We run the algorithm at the specified parameters, on the specified
        # ExDs.
        algrun.run()

        # Verify if the AlgorithmRun now contains the expected results.
        self.assertEqual('mbff.algorithms.basic.IGt.algorithm_IGt__binary', algrun.algorithm_name)
        self.assertEqual('mbff_tests.TestAlgorithmRun.MockBernouliClassifier', algrun.classifier_classname)
        self.assertListEqual([0, 4, 1, 5], algrun.selected_features)
        self.assertLess(0, algrun.duration)

        expected_classifier_evaluation = {
                'TP': 1,
                'TN': 1,
                'FP': 1,
                'FN': 2
                }
        self.assertDictEqual(expected_classifier_evaluation, algrun.classifier_evaluation)


    def test_algorithm_run_configuration(self):
        # Prepare an AlgorithmRun instance.
        configuration = {
                'label': Template('${nosubstitution}__test_IGt_Q${Q}_Obj${objective_index}'),
                'classifier': MockBernouliClassifier,
                'algorithm': IGt,
                }
        parameters = {
                'Q': 4,
                'objective_index': 0
                }
        algrun = AlgorithmAndClassifierRun(None, configuration, parameters)

        self.assertEqual('${nosubstitution}__test_IGt_Q4_Obj0', algrun.label)
        self.assertEqual('mbff.algorithms.basic.IGt.algorithm_IGt__binary', algrun.algorithm_name)
        self.assertEqual('mbff_tests.TestAlgorithmRun.MockBernouliClassifier', algrun.classifier_classname)



    def default_exds_definition(self, exds_folder):
        definition = ExperimentalDatasetDefinition(exds_folder, "test_exds_algorithmrun")
        definition.exds_class = ModelBuildingExperimentalDataset
        definition.source = MockDatasetSource
        definition.source_configuration = {}
        definition.options['training_subset_size'] = 3/8
        definition.options['random_seed'] = 42
        definition.after_save__auto_lock = True
        definition.tags = []
        return definition


    def default_datasetmatrix_train(self):
        pass


    def default_samples_train(self):
        pass



class MockBernouliClassifier:

    def __init__(self):
        pass


    def fit(self, samples, objective):
        pass


    def predict(self, samples):
        return numpy.array([ 1, 1, 0, 0, 0 ])



