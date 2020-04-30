import os
from pathlib import Path

from string import Template
from sklearn.naive_bayes import BernoulliNB

from mbff_tests.TestBase import TestBase
from mbff_tests.MockDatasetSource import MockDatasetSource

from mbff.dataset.ExperimentalDatasetDefinition import ExperimentalDatasetDefinition
from mbff.dataset.ModelBuildingExperimentalDataset import ModelBuildingExperimentalDataset
from mbff.experiment.ExperimentDefinition import ExperimentDefinition
from mbff.experiment.ExperimentRun import ExperimentRun
from mbff.experiment.AlgorithmRun import AlgorithmAndClassifierRun
from mbff.experiment.AlgorithmRunDatapoint import AlgorithmAndClassifierRunDatapoint
from mbff.experiment.Exceptions import ExperimentFolderLockedException
from mbff.algorithms.basic.IGt import AlgorithmIGt


class TestExperimentRun(TestBase):

    def test_experiment_run__simple(self):
        # Prepare folders.
        exds_folder = str(self.ensure_empty_tmp_subfolder('test_exds_repository__experiment_run'))
        experiments_folder = str(self.ensure_empty_tmp_subfolder('test_experiment_repository__experiment_run'))

        # Prepare the ExDs.
        exds_definition = self.default_exds_definition(exds_folder, 'test_exds_experimentrun')
        exds = exds_definition.create_exds()
        exds.build()

        # Define the parameters to be passed to each AlgorithmRun, in order in which they should run.
        algrun_id_format = '{}__test_AlgorithmIGt_BernoulliNB__Q{}_Obj{}'
        algorithm_run_parameters = [
            {'ID': algrun_id_format.format(0, 2, 0), 'Q': 2, 'objective_index': 0},
            {'ID': algrun_id_format.format(1, 4, 0), 'Q': 4, 'objective_index': 0},
            {'ID': algrun_id_format.format(2, 6, 0), 'Q': 6, 'objective_index': 0},
            {'ID': algrun_id_format.format(3, 8, 0), 'Q': 8, 'objective_index': 0},
        ]

        # Prepare the experiment
        experiment_definition = self.default_experiment_definition(experiments_folder, exds_folder, algorithm_run_parameters)
        experiment_run = experiment_definition.create_experiment_run()

        # Run the experiment
        experiment_run.run()

        # Test whether the experiment run has generated the expected log files (one for each AlgorithmRun)
        log_folder = experiments_folder + '/test_experiment_run/algorithm_run_logs'
        self.assertTrue(Path(log_folder).exists())
        expected_log_files = [
            '0__test_AlgorithmIGt_BernoulliNB__Q2_Obj0.log',
            '1__test_AlgorithmIGt_BernoulliNB__Q4_Obj0.log',
            '2__test_AlgorithmIGt_BernoulliNB__Q6_Obj0.log',
            '3__test_AlgorithmIGt_BernoulliNB__Q8_Obj0.log'
        ]
        created_log_files = sorted([f for f in os.listdir(log_folder) if os.path.isfile(log_folder + '/' + f)])
        self.assertListEqual(expected_log_files, created_log_files)

        # Test whether the experiment run has generated the expected pickle files (one for each AlgorithmRunDatapoint)
        datapoints_folder = experiments_folder + '/test_experiment_run/algorithm_run_datapoints'
        self.assertTrue(Path(datapoints_folder).exists())
        expected_datapoints_files = [
            '0__test_AlgorithmIGt_BernoulliNB__Q2_Obj0.pickle',
            '1__test_AlgorithmIGt_BernoulliNB__Q4_Obj0.pickle',
            '2__test_AlgorithmIGt_BernoulliNB__Q6_Obj0.pickle',
            '3__test_AlgorithmIGt_BernoulliNB__Q8_Obj0.pickle'
        ]
        created_datapoints_files = sorted([f for f in os.listdir(datapoints_folder) if os.path.isfile(datapoints_folder + '/' + f)])
        self.assertListEqual(expected_datapoints_files, created_datapoints_files)

        # Disallow running the experiment again, because finishing the
        # experiment should also lock its folder.
        self.assertTrue(experiment_run.definition.folder_is_locked())
        with self.assertRaises(ExperimentFolderLockedException):
            experiment_run.run()

        # After unlocking and deleting logs and datapoint files, the experiment should run again normally.
        experiment_run.definition.unlock_folder()
        self.assertFalse(experiment_run.definition.folder_is_locked())
        experiment_run.definition.delete_subfolder('algorithm_run_logs')
        experiment_run.definition.delete_subfolder('algorithm_run_datapoints')

        experiment_run.run()

        # Re-test whether the expected logs and datapoint files were generated.
        created_log_files = sorted([f for f in os.listdir(log_folder) if os.path.isfile(log_folder + '/' + f)])
        created_datapoints_files = sorted([f for f in os.listdir(datapoints_folder) if os.path.isfile(datapoints_folder + '/' + f)])
        self.assertListEqual(expected_log_files, created_log_files)
        self.assertListEqual(expected_datapoints_files, created_datapoints_files)

        # The experiment should have locked its folder.
        self.assertTrue(experiment_run.definition.folder_is_locked())


    def default_experiment_definition(self, experiments_folder, exds_folder, algorithm_run_parameters):
        definition = ExperimentDefinition(experiments_folder, 'test_experiment_run')
        definition.experiment_run_class = ExperimentRun
        definition.algorithm_run_class = AlgorithmAndClassifierRun
        definition.algorithm_run_datapoint_class = AlgorithmAndClassifierRunDatapoint
        definition.exds_definition = self.default_exds_definition(exds_folder, "test_exds_experimentrun")
        definition.algorithm_run_configuration = {
            'classifier': BernoulliNB,
            'algorithm': AlgorithmIGt,
            'label': Template('${algorithm_run_index}__test_AlgorithmIGt_BernoulliNB__Q${Q}_Obj${objective_index}')
        }
        definition.algorithm_run_parameters = algorithm_run_parameters

        definition.save_algorithm_run_datapoints = True
        definition.algorithm_run_log__stdout = True
        definition.algorithm_run_log__file = True
        definition.quiet = True

        return definition


    def default_exds_definition(self, exds_folder, name):
        definition = ExperimentalDatasetDefinition(exds_folder, name)
        definition.exds_class = ModelBuildingExperimentalDataset
        definition.source = MockDatasetSource
        definition.source_configuration = {}
        definition.options['training_subset_size'] = 3 / 8
        definition.options['random_seed'] = 42
        definition.after_save__auto_lock = True
        definition.tags = []

        return definition



class AlgorithmMockFaultyFS:

    def __init__(self, datasetmatrix, parameters):
        self.parameters = parameters


    def select_features(self):
        if self.parameters['fail']:
            raise FaultyFSAlgorithmException
        if self.parameters['interrupt']:
            raise KeyboardInterrupt
        return [0, 3, 1]



class FaultyFSAlgorithmException(Exception):
    def __init__(self):
        self.message = 'Algorithm failed.'
        super().__init__(self.message)
