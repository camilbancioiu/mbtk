import sys
import time
import contextlib
import pickle
import gc

from string import Template

from mbff.utilities.MultiFileWriter import MultiFileWriter
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
        if self.definition.folder_is_locked('experiment'):
            raise ExperimentFolderLockedException(self.definition, str(self.definition.path), 'Experiment folder is locked, cannot start.')

        self.definition.ensure_subfolder('algorithm_run_logs')
        self.definition.ensure_subfolder('algorithm_run_datapoints')

        self.prepare_exds()

        self.begin_run()

        for algrun_index in range(0, len(self.definition.algorithm_run_parameters)):
            algorithm_run_parameters = self.definition.algorithm_run_parameters[algrun_index]
            self.run_algorithm(algrun_index, algorithm_run_parameters)
            gc.collect()

        self.end_run()


    def prepare_exds(self):
        if self.definition.exds_definition is not None:
            self.exds = self.definition.exds_definition.create_exds()
            if self.definition.exds_definition.exds_ready():
                self.exds.load()
            else:
                self.exds.build()
        else:
            self.exds = None


    def begin_run(self):
        self.start_time = time.time()
        self.print_experiment_run_header()


    def end_run(self):
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.print_experiment_run_footer()

        if self.definition.after_finishing__auto_lock:
            self.definition.lock_folder('experiment')


    def run_algorithm(self, algorithm_run_index, algorithm_run_parameters):
        algorithm_run = self.definition.algorithm_run_class(self.exds, self.definition.algorithm_run_configuration, algorithm_run_parameters)
        algorithm_run.ID = Template(algorithm_run.label).safe_substitute(algorithm_run_index=algorithm_run_index)

        algorithm_run_stdout_destination = self.get_algorithm_run_stdout_destination(algorithm_run)
        try:
            with contextlib.redirect_stdout(algorithm_run_stdout_destination):
                self.print_algorithm_run_header(algorithm_run)
                algorithm_run.run()
                self.print_algorithm_run_footer(algorithm_run)
        finally:
            if algorithm_run_stdout_destination is not sys.__stdout__:
                algorithm_run_stdout_destination.close()

        if self.definition.save_algorithm_run_datapoints:
            self.save_algorithm_run_datapoint(algorithm_run)

        if not self.definition.quiet:
            print(str(algorithm_run))


    def save_algorithm_run_datapoint(self, algorithm_run):
        datapoint_file = self.definition.subfolder('algorithm_run_datapoints') / '{}.pickle'.format(algorithm_run.ID)
        algorithm_run_datapoint = self.definition.algorithm_run_datapoint_class(algorithm_run)
        with datapoint_file.open(mode='wb') as f:
            pickle.dump(algorithm_run_datapoint, f)


    def get_algorithm_run_stdout_destination(self, algorithm_run):
        destinations = []
        if self.definition.algorithm_run_log__stdout:
            destinations.append(sys.stdout)
        if self.definition.algorithm_run_log__file:
            output_file = self.definition.subfolder('algorithm_run_logs') / '{}.log'.format(algorithm_run.ID)
            destinations.append(output_file.open(mode='wt'))

        return MultiFileWriter(destinations)


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
