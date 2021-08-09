from mbtk.utilities.LockablePath import LockablePath


class ExperimentDefinition(LockablePath):

    def __init__(self, experiments_folder, name, subexp_name=None):
        self.experiments_folder = experiments_folder

        self.name = name
        if subexp_name is None:
            subexp_name = 'main'
        self.subexperiment_name = subexp_name

        super().__init__(self.experiments_folder, self.name)
        self.default_lock_type = 'experiment'
        self.experiment_run_class = None
        self.algorithm_run_class = None
        self.algorithm_run_datapoint_class = None
        self.exds_definition = None
        self.algorithm_run_configuration = {}
        self.algorithm_run_parameters = []
        self.save_algorithm_run_datapoints = False
        self.algorithm_run_log__stdout = True
        self.algorithm_run_log__file = False
        self.after_finishing__auto_lock = True
        self.quiet = False
        self.tags = []


    def create_experiment_run(self):
        return self.experiment_run_class(self)


    def ensure_subfolder(self, subfolder_name):
        subfolder = self.path / subfolder_name / self.subexperiment_name
        subfolder.mkdir(parents=True, exist_ok=True)
        return subfolder


    def get_lock(self, lock_type=''):
        if lock_type == '':
            lock_type = self.default_lock_type
        return self.subfolder('locks') / ('locked_{}'.format(lock_type))


    def get_locks(self):
        return [lockfile.name for lockfile in self.subfolder('locks').glob('locked_*')]


    def subfolder_exists(self, subfolder):
        subfolder = self.path / subfolder / self.subexperiment_name
        return subfolder.exists()


    def validate(self):
        pass
