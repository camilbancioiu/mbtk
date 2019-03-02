import os
import sys
import time
import contextlib
import pickle

from pathlib import Path
from string import Template

import mbff.utilities as util
from mbff.experiment.AlgorithmRun import AlgorithmRun
from mbff.experiment.AlgorithmRunDatapoint import AlgorithmRunDatapoint
from mbff.experiment.Exceptions import ExperimentFolderLockedException

class ExperimentRun:

    def __init__(self, definition):
        self.definition = definition
        self.exds = None

        self.start_time = None
        self.end_time = None
        self.duration = None


    def run(self):
        self.definition.ensure_folder()
        if self.definition.folder_is_locked():
            raise ExperimentFolderLockedException(self.definition, self.definition.folder, 'Experiment folder is locked, cannot start.')

        self.start_time = time.time()
        self.print_experiment_run_header()

        self.exds = self.definition.exds_definition.create_exds()
        self.exds.load()

        self.definition.ensure_subfolder('algorithm_run_logs')
        self.definition.ensure_subfolder('algorithm_run_datapoints')

        # Verify if this ExperimentalRun is a continuation of a previous ExperimentalRun.
        # If yes, then resume the previous one by continuing at the
        # algrun_index where it stopped.
        if self.has_previous_checkpoint():
            start_index = self.load_checkpoint()
        else:
            start_index = 0

        for algrun_index in range(start_index, len(self.definition.algorithm_run_parameters)):
            algorithm_run_parameters = self.definition.algorithm_run_parameters[algrun_index]
            self.save_checkpoint(algrun_index)
            self.run_algorithm(algrun_index, algorithm_run_parameters)

        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.remove_checkpoint()
        self.print_experiment_run_footer()

        if self.definition.auto_lock_after_finishing:
            self.definition.lock_folder()


    def run_algorithm(self, algorithm_run_index, algorithm_run_parameters):
        algorithm_run = AlgorithmRun(self.exds, self.definition.algorithm_run_configuration, algorithm_run_parameters)
        algorithm_run.ID = Template(algorithm_run.label).safe_substitute(algorithm_run_index=algorithm_run_index)

        algorithm_run_stdout_destination = self.get_algorithm_run_stdout_destination(algorithm_run)
        try:
            with contextlib.redirect_stdout(algorithm_run_stdout_destination):
                self.print_algorithm_run_header(algorithm_run)
                algorithm_run.run()
                self.print_algorithm_run_footer(algorithm_run)
        finally:
            if not algorithm_run_stdout_destination is sys.__stdout__:
                algorithm_run_stdout_destination.close()

        if self.definition.save_algorithm_run_datapoints:
            self.save_algorithm_run_datapoint(algorithm_run)


    def save_algorithm_run_datapoint(self, algorithm_run):
        self.definition.ensure_subfolder('algorithm_run_datapoints')
        filename = "{}/algorithm_run_datapoints/{}.pickle".format(self.definition.folder, algorithm_run.ID)
        algorithm_run_datapoint = AlgorithmRunDatapoint(algorithm_run)
        with open(filename, 'wb') as f:
            pickle.dump(algorithm_run_datapoint, f)


    def has_previous_checkpoint(self):
        return bool(Path("{}/checkpoint".format(self.definition.folder)).exists())


    def save_checkpoint(self, algorithm_run_index):
        with open("{}/checkpoint".format(self.definition.folder), 'wt') as checkpoint_file:
            checkpoint_file.write(str(algorithm_run_index))


    def load_checkpoint(self):
        with open("{}/checkpoint".format(self.definition.folder), 'rt') as checkpoint_file:
            content = checkpoint_file.read()
        return int(content)


    def remove_checkpoint(self):
        os.remove("{}/checkpoint".format(self.definition.folder))


    def get_algorithm_run_stdout_destination(self, algorithm_run):
        output_file_name = self.definition.folder + '/algorithm_run_logs/{}.log'.format(algorithm_run.ID)
        destination = self.definition.algorithm_run_stdout
        if destination == 'file':
            self.definition.ensure_subfolder('algorithm_run_logs')
            return open(output_file_name, 'wt')
        elif destination == 'logfile-and-stdout':
            self.definition.ensure_subfolder('algorithm_run_logs')
            output_file = open(output_file_name, 'wt')
            return util.MultiFileWriter([output_file, sys.stdout])
        elif destination == 'stdout':
            return sys.stdout
        else:
            return sys.stdout


    def print_experiment_run_header(self):
        if not self.definition.quiet:
            print('Experiment begins.')


    def print_experiment_run_footer(self):
        if not self.definition.quiet:
            print('Experiment ends.')


    def print_algorithm_run_header(self, algorithm_run):
        pass


    def print_algorithm_run_footer(self, algorithm_run):
        pass



