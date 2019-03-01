class ExperimentRun:

    def __init__(self, definition):
        self.definition = definition
        self.exds = None

        self.start_time = None
        self.end_time = None
        self.duration = None


    def run(self):
        self.start_time = time.time()
        self.print_header()

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
        self.print_footer()
        self.remove_checkpoint()


    def run_algorithm(self, algorithm_run_parameters):
        algorithm_run = AlgorithmRun(self.exds, algorithm_run_parameters)
        pass


    def save_checkpoint(self, algorithm_run_index):
        pass


    def load_checkpoint(self):
        pass


    def remove_checkpoint(self):
        pass

