import os
import shutil
from pathlib import Path
from mbff.experiment.Exceptions import ExperimentFolderException
from mbff.utilities.LockablePath import LockablePath

class ExperimentDefinition(LockablePath):

    def __init__(self, experiments_folder, name):
        self.experiments_folder = experiments_folder
        self.name = name
        super().__init__(self.experiments_folder, self.name)
        self.default_lock_type = 'experiment'
        self.experiment_run_class = None
        self.exds_definition = None
        self.algorithm_run_configuration = {}
        self.algorithm_run_parameters = []
        self.save_algorithm_run_datapoints = False
        self.algorithm_run_log__stdout = True
        self.algorithm_run_log__file = False
        self.auto_lock_after_finishing = True
        self.quiet = False
        self.tags = []


    def create_experiment_run(self):
        return self.experiment_run_class(self)


    def validate(self):
        pass

