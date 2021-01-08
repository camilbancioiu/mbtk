import numpy

from tests.MockDatasetSource import MockDatasetSource

from mbtk.dataset.ExperimentalDatasetDefinition import ExperimentalDatasetDefinition
from mbtk.dataset.ModelBuildingExperimentalDataset import ModelBuildingExperimentalDataset
from mbtk.experiment.AlgorithmRun import AlgorithmAndClassifierRun
from mbtk.algorithms.basic.IGt import AlgorithmIGt

import pytest
import tests.utilities as testutil


@pytest.fixture(scope='session')
def testfolder():
    return testutil.ensure_empty_tmp_subfolder('test_exds_repository__test_build')


def test_algorithm_run(testfolder):
    # Prepare an ExperimentalDataset to perform the test on.
    definition = default_exds_definition(testfolder)
    exds = definition.create_exds()
    exds.build()

    # Because definition.random_seed = 42, the 'randomly' selected training
    # rows in the ExperimentalDataset will always be [0, 1, 5].

    # Prepare an AlgorithmRun instance.
    configuration = {
        'label': 'test_algrun',
        'classifier': MockBernouliClassifier,
        'algorithm': AlgorithmIGt,
    }
    parameters = {
        'Q': 4,
        'objective_index': 0,
        'ID': 'test_algrun_Q4_T0',
    }
    algrun = AlgorithmAndClassifierRun(exds, configuration, parameters)

    # We run the algorithm at the specified parameters, on the specified
    # ExDs.
    algrun.run()

    # Verify if the AlgorithmRun now contains the expected results.
    assert algrun.algorithm_name == 'mbtk.algorithms.basic.IGt.AlgorithmIGt'
    assert algrun.classifier_classname == 'tests.test_AlgorithmRun.MockBernouliClassifier'
    assert algrun.selected_features == [0, 4, 1, 5]
    assert algrun.duration > 0

    expected_classifier_evaluation = {
        'TP': 1,
        'TN': 1,
        'FP': 1,
        'FN': 2
    }
    assert algrun.classifier_evaluation == expected_classifier_evaluation


def test_algorithm_run_configuration():
    # Prepare an AlgorithmRun instance.
    configuration = {
        'classifier': MockBernouliClassifier,
        'algorithm': AlgorithmIGt,
    }
    parameters = {
        'Q': 4,
        'objective_index': 0,
        'ID': 'test_algrun_Q4_T0',
    }
    algrun = AlgorithmAndClassifierRun(None, configuration, parameters)

    assert algrun.algorithm_name == 'mbtk.algorithms.basic.IGt.AlgorithmIGt'
    assert algrun.classifier_classname == 'tests.test_AlgorithmRun.MockBernouliClassifier'



def default_exds_definition(exds_folder):
    definition = ExperimentalDatasetDefinition(exds_folder, "test_exds_algorithmrun")
    definition.exds_class = ModelBuildingExperimentalDataset
    definition.source = MockDatasetSource
    definition.source_configuration = {}
    definition.options['training_subset_size'] = 3 / 8
    definition.options['random_seed'] = 42
    definition.after_save__auto_lock = True
    definition.tags = []
    return definition



class MockBernouliClassifier:

    def __init__(self):
        pass


    def fit(self, samples, objective):
        pass


    def predict(self, samples):
        return numpy.array([1, 1, 0, 0, 0])
