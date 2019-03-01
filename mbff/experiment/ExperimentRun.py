import sys
import contextlib

import mbff.utilities as util

class ExperimentRun:

    def __init__(self, definition):
        self.definition = definition
        self.exds = None

        self.start_time = None
        self.end_time = None
        self.duration = None


    def run(self):
        self.start_time = time.time()
        self.print_experiment_run_header()

        self.definition.ensure_folder()
        if self.definition.folder_is_locked():
            raise ExperimentError(self.definition, 'Experiment folder is locked, cannot start.')

        self.exds = self.definition.exds_definition.create_exds()
        self.exds.load()


        self.definition.ensure_subfolder('logs')
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
            self.run_algorithm(algorithm_run_parameters)
            self.save_checkpoint(algrun_index)

        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.remove_checkpoint()
        self.print_experiment_run_footer()


    def run_algorithm(self, algorithm_run_index, algorithm_run_parameters):
        algorithm_run = AlgorithmRun(self.exds, algorithm_run_parameters)
        algorithm_run.ID = "{}__{}".format(algorithm_run_index, algorithm_run.label)

        algorithm_run_stdout_destination = self.get_algorithm_run_stdout_destination(algorithm_run)
        with contextlib.redirect_stdout(algorithm_run_stdout_destination):
            self.print_algorithm_run_header(algorithm_run)
            algorithm_run.run()
            self.print_algorithm_run_footer(algorithm_run)

        if not algorithm_run_stdout_destination is sys.__stdout__:
            algorithm_run_stdout_destination.close()

        if self.definition.configuration.get('save_algorithm_run_datapoints', True):
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
            checkpoint_file.write(algorithm_run_index)


    def load_checkpoint(self):
        with open("{}/checkpoint".format(self.definition.folder), 'rt') as checkpoint_file:
            content = checkpoint_file.read()
        return int(content)


    def remove_checkpoint(self):
        os.remove("{}/checkpoint".format(self.definition.folder))


    def get_algorithm_run_stdout_destination(self, algorithm_run):
        output_file_name = self.definition.folder + '/algorithm_run_logs/{}.log'.format(algorithm_run.ID)
        destination = self.definition.configuration.get('algorithm_run_stdout', 'stdout')
        if destination == 'file':
            self.definition.ensure_subfolder('algorithm_run_logs')
            return open(output_file_name, 'wt')
        elif destination == 'file-and-stdout':
            self.definition.ensure_subfolder('algorithm_run_logs')
            output_file = open(output_file_name, 'wt')
            return util.MultiFileWriter([output_file, sys.stdout])
        elif destination == 'stdout':
            return sys.stdout
        else:
            return sys.stdout


    def print_experiment_run_header(self):
        quiet = self.definition.configuration.get('quiet', False)
        if not quiet:
            print('Experiment begins.')


    def print_experiment_run_footer(self):
        quiet = self.definition.configuration.get('quiet', False)
        if not quiet:
            print('Experiment ends.')


    def print_algorithm_run_header(self, algorithm_run):
        pass


    def print_algorithm_run_footer(self, algorithm_run):
        pass



