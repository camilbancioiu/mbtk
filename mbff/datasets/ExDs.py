import random
import os
import shutil
from pathlib import Path
from mbff.datasets.DatasetMatrix import DatasetMatrix

class ExperimentalDatasetDefinition():
    """
    The definition of an ExperimentalDataset, which contains the information needed
    to build an experimental dataset (exds) from external source data, split it into
    training and testing samplesets, save it and analyze it later.

    :var name: The unique machine-name identifier of the exds.
    :var source: A class inheriting the DatasetSource class which has a
        .build() method that returns an annotated DatasetMatrix
        object created from external source data.
    :var source_args: A dictionary object which configures the DatasetSource class
        provided as the ``source`` property.
    :var training_subset_size: A real value representing the proportion of dataset
        rows allocated to training a model. By specifying a value less than 1.0,
        the dataset will be randomly split into two non-overlapping subsets of rows,
        one for training, the other for testing the model.
    :var trim_prob__feature: A tuple of two real values between 0.0 and 1.0,
        representing the minimum and maximum thresholds of allowed probability
        of a value within a feature of the dataset. Namely, if a value appears
        too few times in a feature (or too many times), then the feature is deemed
        uninformative and it is removed from the dataset.
    :var trim_prob__objective: A tuple of two real values between 0.0 and 1.0,
        representing the minimum and maximum thresholds of allowed probability
        of a value within an objective variable of the dataset. Namely, if too
        few or too many samples have a single value for an objective variable,
        then the objective variable is deemed uninformative and it is removed
        from the dataset.
    :var random_seed: An integer chosen arbitrarily, which controls the random
        selection of training/testing rows. Should be set once per exds definition
        and never changed (unless you know what you are doing).
    :var auto_lock_after_build: Determines whether the folder of the exds is
        automatically locked after building. Locking the folder of an exds will
        prevent accidental rebuilding or deleting.
    :var tags: A list of arbitrary strings, used to categorize this exds.

    """

    def __init__(self):
        self.name = ""
        self.source = None
        self.source_args = {}
        self.exds_folder = ""
        self.folder = ""
        self.training_subset_size = 1.0
        self.trim_prob__feature = (0.0, 1.0)
        self.trim_prob__objective = (0.0, 1.0)
        self.random_seed = 42
        self.auto_lock_after_build = True
        self.tags = []


    def setup(self):
        self.folder = self.exds_folder + '/' + self.name
        self.validate()


    def validate(self):
        pass


    def get_lock_filename(self, lock_type='exds'):
        return '{}/locked_{}'.format(self.folder, lock_type)


    def folder_is_locked(self, lock_type='exds'):
        return bool(Path(self.get_lock_filename(lock_type)).exists())


    def folder_exists(self):
        return bool(Path(self.folder).exists())


    def lock_folder(self, lock_type='exds'):
        folder = self.folder
        if self.folder_is_locked(lock_type):
            print('{}: Folder {} is already locked ({}).'.format(self.name, folder, lock_type))
        else:
            with open(self.get_lock_filename(lock_type), 'w') as f:
                f.write('locked')
                print('{}: Folder has been locked ({}).'.format(self.name, lock_type))


    def delete_folder(self):
        if not self.folder_exists():
            print('{}: ExDs folder does not exist.'.format(self.name))
            return
        if self.folder_is_locked():
            print('{}: ExDs folder is locked, cannot delete.'.format(self.name))
            return
        shutil.rmtree(self.folder)
        print('{}: ExDs folder deleted.'.format(self.name))


    def unlock_folder(self, lock_type='exds'):
        folder = self.folder
        if not self.folder_is_locked(lock_type):
            print('{}: Folder {} is not locked ({}).'.format(self.name, folder, lock_type))
        else:
            os.remove(self.get_lock_filename(lock_type))
            print('{}: Folder {} has been unlocked ({}).'.format(self.name, folder, lock_type))


class ExperimentalDataset():
    def __init__(self, definition):
        self.definition = definition
        self.matrix = None
        self.matrix_train = None
        self.matrix_test = None
        self.total_row_count = 0
        self.train_rows = None
        self.test_rows = None


class ExperimentalDatasetError(Exception):
    def __init__(self, exds_definition, message):
        self.exds_definition = exds_definition
        self.message = message
        super().__init__(self.message)

