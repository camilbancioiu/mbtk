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
from mbff.experiment.AlgorithmRun import AlgorithmAndClassifierRun
from mbff.experiment.AlgorithmRunDatapoint import AlgorithmAndClassifierRunDatapoint
from mbff.experiment.Exceptions import ExperimentFolderLockedException
from mbff.algorithms.basic.IGt import algorithm_IGt__binary as IGt


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
        algorithm_run_parameters = [
                {'Q': 2, 'objective_index': 0},
                {'Q': 4, 'objective_index': 0},
                {'Q': 6, 'objective_index': 0},
                {'Q': 8, 'objective_index': 0},
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
                '0__test_IGt_BernoulliNB__Q2_Obj0.log',
                '1__test_IGt_BernoulliNB__Q4_Obj0.log',
                '2__test_IGt_BernoulliNB__Q6_Obj0.log',
                '3__test_IGt_BernoulliNB__Q8_Obj0.log'
                ]
        created_log_files = sorted([f for f in os.listdir(log_folder) if os.path.isfile(log_folder + '/' + f)])
        self.assertListEqual(expected_log_files, created_log_files)

        # Test whether the experiment run has generated the expected pickle files (one for each AlgorithmRunDatapoint)
        datapoints_folder = experiments_folder + '/test_experiment_run/algorithm_run_datapoints'
        self.assertTrue(Path(datapoints_folder).exists())
        expected_datapoints_files = [
                '0__test_IGt_BernoulliNB__Q2_Obj0.pickle',
                '1__test_IGt_BernoulliNB__Q4_Obj0.pickle',
                '2__test_IGt_BernoulliNB__Q6_Obj0.pickle',
                '3__test_IGt_BernoulliNB__Q8_Obj0.pickle'
                ]
        created_datapoints_files = sorted([f for f in os.listdir(datapoints_folder) if os.path.isfile(datapoints_folder + '/' + f)])
        self.assertListEqual(expected_datapoints_files, created_datapoints_files)

        # Test whether the checkpoint file of the ExperimentRun has been
        # removed - it should, if the ExperimentRun has completed successfully.
        self.assertFalse(Path(experiments_folder + '/test_experiment_run/checkpoint').exists())

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

        # The checkpoint file should have been removed at the end of the run.
        self.assertFalse(Path(experiments_folder + '/test_experiment_run/checkpoint').exists())

        # The experiment should have locked its folder.
        self.assertTrue(experiment_run.definition.folder_is_locked())


    def test_experiment_run__interrupted(self):
        # Prepare the folders.
        exds_folder = str(self.ensure_empty_tmp_subfolder('test_exds_repository__experiment_run_interrupted'))
        experiments_folder = str(self.ensure_empty_tmp_subfolder('test_experiment_repository__experiment_run_interrupted'))

        # Prepare the ExDs.
        exds_definition = self.default_exds_definition(exds_folder, 'test_exds_experimentrun_interrupted')
        exds = exds_definition.create_exds()
        exds.build()

        # Define the ExperimentRun. Use a mock algorithm, which will break when requested by the parameters.
        definition = ExperimentDefinition(experiments_folder, "test_experiment_run__interrupted")
        definition.experiment_run_class = ExperimentRun
        definition.algorithm_run_class = AlgorithmAndClassifierRun
        definition.algorithm_run_datapoint_class = AlgorithmAndClassifierRunDatapoint
        definition.exds_definition = self.default_exds_definition(exds_folder, "test_exds_experimentrun_interrupted")
        definition.algorithm_run_configuration = {
                'classifier': BernoulliNB,
                'algorithm': algorithm_MockFaultyFS, # mock algorithm which breaks when requested.
                'label': Template('${algorithm_run_index}__MockFaultyFS')
                }
        # Firstly, prepare the algorithm parameters (fill with defaults).
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
        definition.algorithm_run_log__stdout = True
        definition.algorithm_run_log__file = True
        definition.quiet = True

        experiment_run = definition.create_experiment_run()

        # Trigger a crash at AlgorithmRun index 2.
        experiment_run.definition.algorithm_run_parameters[2]['fail'] = True

        # Run the experiment, expecting it to fail at the third AlgorithmRun.
        try:
            experiment_run.run()
        except FaultyFSAlgorithmException:
            pass

        # We expect 3 log files, not 2, because even if the third AlgorithmRun
        # has crashed, it has still opened a log file, written to it and then
        # crashed. The log file of the crashed AlgorithmRun is closed by a
        # ``finally`` block in ExperimentRun.run_algorithm().
        log_folder = experiments_folder + '/test_experiment_run__interrupted/algorithm_run_logs'
        self.assertTrue(Path(log_folder).exists())
        expected_log_files = [
                '0__MockFaultyFS.log',
                '1__MockFaultyFS.log',
                '2__MockFaultyFS.log',
                ]
        created_log_files = sorted([f for f in os.listdir(log_folder) if os.path.isfile(log_folder + '/' + f)])
        self.assertListEqual(expected_log_files, created_log_files)

        # We expect 2 pickle files only, because the third AlgorithmRun has
        # crashed and there was no datapoint file created for it (a datapoint
        # file would have only been created at the end of the AlgorithmRun).
        datapoints_folder = experiments_folder + '/test_experiment_run__interrupted/algorithm_run_datapoints'
        self.assertTrue(Path(datapoints_folder).exists())
        expected_datapoints_files = [
                '0__MockFaultyFS.pickle',
                '1__MockFaultyFS.pickle',
                ]
        created_datapoints_files = sorted([f for f in os.listdir(datapoints_folder) if os.path.isfile(datapoints_folder + '/' + f)])
        self.assertListEqual(expected_datapoints_files, created_datapoints_files)

        # There should be a checkpoint file remaining in the folder of the
        # ExperimentRun, containing '2', namely the index of the AlgorithmRun
        # that was started last, before the 'crash' happened.
        self.assertTrue(Path(experiments_folder + '/test_experiment_run__interrupted/checkpoint').exists())
        self.assertEqual(2, experiment_run.load_checkpoint())

        # Now we prevent the algorithm from crashing, but it will be
        # interrupted with KeyboardInterrupt at index 6, as if the user pressed
        # Ctrl+C in the terminal when the experiment was running.
        experiment_run.definition.algorithm_run_parameters[2]['fail'] = False
        experiment_run.definition.algorithm_run_parameters[6]['interrupt'] = True

        # The experiment should resume at the third AlgorithmRun, honoring the
        # checkpoint, then stop at the seventh.
        try:
            experiment_run.run()
        except KeyboardInterrupt:
            pass

        # The experiment was interrupted at the AlgorithmRun with index 6.
        # Verify the logs and the datapoint files created.
        log_folder = experiments_folder + '/test_experiment_run__interrupted/algorithm_run_logs'
        self.assertTrue(Path(log_folder).exists())
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
        self.assertTrue(Path(datapoints_folder).exists())
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

        # There should still be a checkpoint file remaining, containing '6',
        # namely the index of the last AlgorithmRun started (but not
        # completed).
        self.assertTrue(Path(experiments_folder + '/test_experiment_run__interrupted/checkpoint').exists())
        self.assertEqual(6, experiment_run.load_checkpoint())

        # Now remove all instructions to crash/interrupt, allowing the
        # experiment to run to the end.
        experiment_run.definition.algorithm_run_parameters[6]['interrupt'] = False

        # Resume the experiment. There should be no exception thrown here.
        experiment_run.run()

        # Test whether the expected logs and datapoint files were created.
        log_folder = experiments_folder + '/test_experiment_run__interrupted/algorithm_run_logs'
        self.assertTrue(Path(log_folder).exists())
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
        self.assertTrue(Path(datapoints_folder).exists())
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

        # There should be no more checkpoint file remaining in the folder of
        # the ExperimentRun, since it ran until the end.
        self.assertFalse(Path(experiments_folder + '/test_experiment_run__interrupted/checkpoint').exists())


    def test_experiment_run__contents_of_logs_and_datapoints(self):
        pass


    def default_experiment_definition(self, experiments_folder, exds_folder, algorithm_run_parameters):
        definition = ExperimentDefinition(experiments_folder, 'test_experiment_run')
        definition.experiment_run_class = ExperimentRun
        definition.algorithm_run_class = AlgorithmAndClassifierRun
        definition.algorithm_run_datapoint_class = AlgorithmAndClassifierRunDatapoint
        definition.exds_definition = self.default_exds_definition(exds_folder, "test_exds_experimentrun")
        definition.algorithm_run_configuration = {
                'classifier': BernoulliNB,
                'algorithm': IGt,
                'label': Template('${algorithm_run_index}__test_IGt_BernoulliNB__Q${Q}_Obj${objective_index}')
                }
        definition.algorithm_run_parameters = algorithm_run_parameters

        definition.save_algorithm_run_datapoints = True
        definition.algorithm_run_log__stdout = True
        definition.algorithm_run_log__file = True
        definition.quiet = True

        return definition


    def default_exds_definition(self, exds_folder, name):
        definition = ExperimentalDatasetDefinition(exds_folder, name)
        definition.exds_class = ExperimentalDataset
        definition.source = MockDatasetSource
        definition.source_configuration = {}
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
