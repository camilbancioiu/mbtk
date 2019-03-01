from mbff.experiment.Exceptions import ExperimentFolderException

class ExperimentDefinition:

    def __init__(self, name, exds_definition, experiments_folder, algorithm_run_parameters, configuration):
        self.name = name
        self.exds_definition = exds_definition
        self.algorithm_run_parameters = algorithm_run_parameters
        self.tags = []
        self.folder = experiments_folder + '/' + self.name
        self.configuration = configuration


    def get_lock_filename(self, lock_type='experiment'):
        return '{}/locked_{}'.format(self.folder, lock_type)


    def folder_is_locked(self, lock_type='experiment'):
        return bool(Path(self.get_lock_filename(lock_type)).exists())


    def folder_exists(self):
        return bool(Path(self.folder).exists())


    def subfolder_exists(self, subfolder):
        return bool(Path(self.folder + '/' + subfolder).exists())


    def lock_folder(self, lock_type='experiment'):
        folder = self.folder
        if self.folder_is_locked(lock_type):
            pass
        else:
            with open(self.get_lock_filename(lock_type), 'w') as f:
                f.write('locked')


    def delete_folder(self):
        if not self.folder_exists():
            raise ExperimentFolderException(self, self.folder, 'Experiment folder {} does not exist, cannot delete it.'.format(self.name))
        if self.folder_is_locked():
            raise ExperimentFolderException(self, self.folder, 'Experiment folder {} is locked, cannot delete it.'.format(self.name))
            return
        shutil.rmtree(self.folder)


    def delete_subfolder(self, subfolder):
        if self.folder_is_locked():
            raise ExperimentFolderException(self, self.folder, 'Experiment folder is locked, cannot delete any subfolder.')
        if not self.subfolder_exists(subfolder):
            raise ExperimentFolderException(self, self.folder, 'Experiment subfolder {} does not exist.'.format(subfolder))
        shutil.rmtree(self.folder + '/' + subfolder)


    def unlock_folder(self, lock_type='exds'):
        folder = self.folder
        if not self.folder_is_locked(lock_type):
            pass
        else:
            os.remove(self.get_lock_filename(lock_type))


    def ensure_folder(self):
        path = Path('./' + self.folder)
        path.mkdir(parents=True, exist_ok=True)


    def ensure_subfolder(self, subfolder):
        path = Path('./' + self.folder + '/' + subfolder)
        path.mkdir(parents=True, exist_ok=True)
