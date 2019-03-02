import os
from pathlib import Path

from string import Template
from sklearn.naive_bayes import BernoulliNB

from mbff_tests.TestBase import TestBase
from mbff_tests.MockDatasetSource import MockDatasetSource

from mbff.dataset.ExperimentalDatasetDefinition import ExperimentalDatasetDefinition
from mbff.dataset.ExperimentalDataset import ExperimentalDataset
from mbff.experiment.ExperimentDefinition import ExperimentDefinition
from mbff.experiment.ExperimentRun import ExperimentRun
from mbff.experiment.Exceptions import ExperimentFolderLockedException
from mbff.algorithms.basic.IGt import algorithm_IGt__binary as IGt


class TestExperimentRun(TestBase):

    def test_experiment_run__simple(self):
        exds_folder = str(self.ensure_empty_tmp_subfolder('test_exds_repository__experiment_run'))
        experiments_folder = str(self.ensure_empty_tmp_subfolder('test_experiment_repository__experiment_run'))

        exds_definition = self.default_exds_definition(exds_folder, 'test_exds_experimentrun')
        exds = exds_definition.create_exds()
        exds.build()

        algorithm_run_parameters = [
                {'Q': 2, 'objective_index': 0},
                {'Q': 4, 'objective_index': 0},
                {'Q': 6, 'objective_index': 0},
                {'Q': 8, 'objective_index': 0},
                ]

        experiment_definition = self.default_experiment_definition(experiments_folder, exds_folder, algorithm_run_parameters)

        experiment_run = experiment_definition.create_experiment_run()
        experiment_run.run()

        log_folder = experiments_folder + '/test_experiment_run/algorithm_run_logs'
        self.assertTrue(bool(Path(log_folder).exists()))
        expected_log_files = [
                '0__test_IGt_BernoulliNB__Q2_Obj0.log',
                '1__test_IGt_BernoulliNB__Q4_Obj0.log',
                '2__test_IGt_BernoulliNB__Q6_Obj0.log',
                '3__test_IGt_BernoulliNB__Q8_Obj0.log'
                ]
        created_log_files = sorted([f for f in os.listdir(log_folder) if os.path.isfile(log_folder + '/' + f)])
        self.assertListEqual(expected_log_files, created_log_files)

        datapoints_folder = experiments_folder + '/test_experiment_run/algorithm_run_datapoints'
        self.assertTrue(bool(Path(datapoints_folder).exists()))
        expected_datapoints_files = [
                '0__test_IGt_BernoulliNB__Q2_Obj0.pickle',
                '1__test_IGt_BernoulliNB__Q4_Obj0.pickle',
                '2__test_IGt_BernoulliNB__Q6_Obj0.pickle',
                '3__test_IGt_BernoulliNB__Q8_Obj0.pickle'
                ]
        created_datapoints_files = sorted([f for f in os.listdir(datapoints_folder) if os.path.isfile(datapoints_folder + '/' + f)])
        self.assertListEqual(expected_datapoints_files, created_datapoints_files)

        self.assertFalse(bool(Path(experiments_folder + '/test_experiment_run/checkpoint').exists()))

        # Disallow running the experiment again, because finishing the
        # experiment should also lock its folder.
        self.assertTrue(experiment_run.definition.folder_is_locked())
        with self.assertRaises(ExperimentFolderLockedException):
            experiment_run.run()

        # After unlocking, the experiment should run again normally.
        experiment_run.definition.unlock_folder()
        self.assertFalse(experiment_run.definition.folder_is_locked())
        experiment_run.definition.delete_subfolder('algorithm_run_logs')
        experiment_run.definition.delete_subfolder('algorithm_run_datapoints')

        experiment_run.run()

        created_log_files = sorted([f for f in os.listdir(log_folder) if os.path.isfile(log_folder + '/' + f)])
        created_datapoints_files = sorted([f for f in os.listdir(datapoints_folder) if os.path.isfile(datapoints_folder + '/' + f)])
        self.assertListEqual(expected_log_files, created_log_files)
        self.assertListEqual(expected_datapoints_files, created_datapoints_files)
        self.assertFalse(bool(Path(experiments_folder + '/test_experiment_run/checkpoint').exists()))

        self.assertTrue(experiment_run.definition.folder_is_locked())


    def test_experiment_run__interrupted(self):
        exds_folder = str(self.ensure_empty_tmp_subfolder('test_exds_repository__experiment_run_interrupted'))
        experiments_folder = str(self.ensure_empty_tmp_subfolder('test_experiment_repository__experiment_run_interrupted'))

        exds_definition = self.default_exds_definition(exds_folder, 'test_exds_experimentrun_interrupted')
        exds = exds_definition.create_exds()
        exds.build()

        definition = ExperimentDefinition()
        definition.name = "test_experiment_run__interrupted"
        definition.experiment_run_class = ExperimentRun
        definition.experiments_folder = experiments_folder
        definition.exds_definition = self.default_exds_definition(exds_folder, "test_exds_experimentrun_interrupted")
        definition.algorithm_run_configuration = {
                'classifier': BernoulliNB,
                'algorithm': algorithm_MockFaultyFS,
                'label': Template('${algorithm_run_index}__MockFaultyFS')
                }
        definition.algorithm_run_parameters = [
                {'fail': False, 'interrupt': False, 'objective_index': 0},
                {'fail': False, 'interrupt': False, 'objective_index': 0},
                {'fail': False, 'interrupt': False, 'objective_index': 0},
                {'fail': False, 'interrupt': False, 'objective_index': 0},
                {'fail': False, 'interrupt': False, 'objective_index': 0},
                {'fail': False, 'interrupt': False, 'objective_index': 0},
                {'fail': False, 'interrupt': False, 'objective_index': 0},
                {'fail': False, 'interrupt': False, 'objective_index': 0},
                ]
        definition.save_algorithm_run_datapoints = True
        definition.algorithm_run_stdout = 'logfile-and-stdout'
        definition.quiet = True

        experiment_run = definition.create_experiment_run()

        # Trigger a crash at AlgorithmRun index 2.
        experiment_run.definition.algorithm_run_parameters[2]['fail'] = True

        try:
            experiment_run.run()
        except FaultyFSAlgorithmException:
            pass

        # We expect 3 log files, not 2, because even if the third AlgorithmRun
        # has crashed, it has still opened a log file, written to it and then
        # crashed. The log file of the crashed AlgorithmRun is closed by a
        # ``finally`` block in ExperimentRun.run_algorithm().
        log_folder = experiments_folder + '/test_experiment_run__interrupted/algorithm_run_logs'
        self.assertTrue(bool(Path(log_folder).exists()))
        expected_log_files = [
                '0__MockFaultyFS.log',
                '1__MockFaultyFS.log',
                '2__MockFaultyFS.log',
                ]
        created_log_files = sorted([f for f in os.listdir(log_folder) if os.path.isfile(log_folder + '/' + f)])
        self.assertListEqual(expected_log_files, created_log_files)

        # We expect 2 pickle files only, because the third AlgorithmRun has
        # crashed and there was no datapoint file created for it.
        datapoints_folder = experiments_folder + '/test_experiment_run__interrupted/algorithm_run_datapoints'
        self.assertTrue(bool(Path(datapoints_folder).exists()))
        expected_datapoints_files = [
                '0__MockFaultyFS.pickle',
                '1__MockFaultyFS.pickle',
                ]
        created_datapoints_files = sorted([f for f in os.listdir(datapoints_folder) if os.path.isfile(datapoints_folder + '/' + f)])
        self.assertListEqual(expected_datapoints_files, created_datapoints_files)

        self.assertTrue(bool(Path(experiments_folder + '/test_experiment_run__interrupted/checkpoint').exists()))
        self.assertEqual(2, experiment_run.load_checkpoint())

        # Now we prevent the algorithm from crashing, but it will be
        # interrupted with KeyboardInterrupt at index 6.
        experiment_run.definition.algorithm_run_parameters[2]['fail'] = False
        experiment_run.definition.algorithm_run_parameters[6]['interrupt'] = True

        # The experiment should resume.
        try:
            experiment_run.run()
        except KeyboardInterrupt:
            pass

        # The experiment was interrupted at the AlgorithmRun with index 6.
        log_folder = experiments_folder + '/test_experiment_run__interrupted/algorithm_run_logs'
        self.assertTrue(bool(Path(log_folder).exists()))
        expected_log_files = [
                '0__MockFaultyFS.log',
                '1__MockFaultyFS.log',
                '2__MockFaultyFS.log',
                '3__MockFaultyFS.log',
                '4__MockFaultyFS.log',
                '5__MockFaultyFS.log',
                '6__MockFaultyFS.log',
                ]
        created_log_files = sorted([f for f in os.listdir(log_folder) if os.path.isfile(log_folder + '/' + f)])
        self.assertListEqual(expected_log_files, created_log_files)

        datapoints_folder = experiments_folder + '/test_experiment_run__interrupted/algorithm_run_datapoints'
        self.assertTrue(bool(Path(datapoints_folder).exists()))
        expected_datapoints_files = [
                '0__MockFaultyFS.pickle',
                '1__MockFaultyFS.pickle',
                '2__MockFaultyFS.pickle',
                '3__MockFaultyFS.pickle',
                '4__MockFaultyFS.pickle',
                '5__MockFaultyFS.pickle',
                ]
        created_datapoints_files = sorted([f for f in os.listdir(datapoints_folder) if os.path.isfile(datapoints_folder + '/' + f)])
        self.assertListEqual(expected_datapoints_files, created_datapoints_files)

        self.assertTrue(bool(Path(experiments_folder + '/test_experiment_run__interrupted/checkpoint').exists()))
        self.assertEqual(6, experiment_run.load_checkpoint())

        # Now remove all crashes, allowing the experiment to run to the end.
        experiment_run.definition.algorithm_run_parameters[6]['interrupt'] = False

        experiment_run.run()
        log_folder = experiments_folder + '/test_experiment_run__interrupted/algorithm_run_logs'
        self.assertTrue(bool(Path(log_folder).exists()))
        expected_log_files = [
                '0__MockFaultyFS.log',
                '1__MockFaultyFS.log',
                '2__MockFaultyFS.log',
                '3__MockFaultyFS.log',
                '4__MockFaultyFS.log',
                '5__MockFaultyFS.log',
                '6__MockFaultyFS.log',
                '7__MockFaultyFS.log',
                ]
        created_log_files = sorted([f for f in os.listdir(log_folder) if os.path.isfile(log_folder + '/' + f)])
        self.assertListEqual(expected_log_files, created_log_files)

        datapoints_folder = experiments_folder + '/test_experiment_run__interrupted/algorithm_run_datapoints'
        self.assertTrue(bool(Path(datapoints_folder).exists()))
        expected_datapoints_files = [
                '0__MockFaultyFS.pickle',
                '1__MockFaultyFS.pickle',
                '2__MockFaultyFS.pickle',
                '3__MockFaultyFS.pickle',
                '4__MockFaultyFS.pickle',
                '5__MockFaultyFS.pickle',
                '6__MockFaultyFS.pickle',
                '7__MockFaultyFS.pickle',
                ]
        created_datapoints_files = sorted([f for f in os.listdir(datapoints_folder) if os.path.isfile(datapoints_folder + '/' + f)])
        self.assertListEqual(expected_datapoints_files, created_datapoints_files)

        self.assertFalse(bool(Path(experiments_folder + '/test_experiment_run__interrupted/checkpoint').exists()))


    def test_experiment_run__contents_of_logs_and_datapoints(self):
        pass


    def default_experiment_definition(self, experiments_folder, exds_folder, algorithm_run_parameters):
        definition = ExperimentDefinition()
        definition.name = 'test_experiment_run'
        definition.experiment_run_class = ExperimentRun
        definition.experiments_folder = experiments_folder
        definition.exds_definition = self.default_exds_definition(exds_folder, "test_exds_experimentrun")
        definition.algorithm_run_configuration = {
                'classifier': BernoulliNB,
                'algorithm': IGt,
                'label': Template('${algorithm_run_index}__test_IGt_BernoulliNB__Q${Q}_Obj${objective_index}')
                }
        definition.algorithm_run_parameters = algorithm_run_parameters

        definition.save_algorithm_run_datapoints = True
        definition.algorithm_run_stdout = 'logfile-and-stdout'
        definition.quiet = True

        return definition


    def default_exds_definition(self, exds_folder, name):
        definition = ExperimentalDatasetDefinition()
        definition.name = name
        definition.exds_class = ExperimentalDataset
        definition.source = MockDatasetSource
        definition.source_configuration = {}
        definition.exds_folder = exds_folder
        definition.training_subset_size = 3/8
        definition.random_seed = 42
        definition.auto_lock_after_build = True
        definition.tags = []

        return definition



def algorithm_MockFaultyFS(datasetmatrix, parameters):
    if parameters['fail']:
        raise FaultyFSAlgorithmException
    if parameters['interrupt']:
        raise KeyboardInterrupt
    return [0, 3, 1]



class FaultyFSAlgorithmException(Exception):
    def __init__(self):
        self.message = 'Algorithm failed.'
        super().__init__(self.message)
