import random
import os
import shutil
from pathlib import Path
from mbff.dataset.DatasetMatrix import DatasetMatrix
from mbff.dataset.Exceptions import ExperimentalDatasetFolderException
from mbff.utilities.LockablePath import LockablePath

class ExperimentalDatasetDefinition(LockablePath):
    """
    This class represents the definition of an ``ExperimentalDataset``, which
    contains the information needed to build an experimental dataset (exds)
    from external source data, split it into training and testing samplesets,
    save it to a specified folder and analyze it later. This would result in a
    proper ``ExperimentalDataset`` instance.

    Also provided are methods to "lock" and "unlock" the folder of an
    experimental dataset. Locking will prevent accidental overwriting or
    deleting of the folder of an exds.

    :var name: The unique machine-name identifier of the exds.
    :var source: A class inheriting ``DatasetSource``, to be instantiated,
        configured and used to generate a ``DatasetMatrix`` from an external
        source of data.
    :var source_configuration: A dictionary object which configures the
        DatasetSource class provided as the ``source`` property.
    :var exds_folder: The folder where the exds should create its own subfolder
        to save itself to, or where to load itself from. Can be thought of as
        the "experimental dataset repository".
    :var after_build__finalize_and_save: Determines whether the exds is
        finalized and saved immediately after building. If false, the exds will
        be built in memory only, and saving can be performed manually at a
        later time.
    :var after_save__auto_lock: Determines whether the folder of the exds is
        automatically locked after building. Locking the folder of an exds will
        prevent accidental rebuilding or deleting. Only relevant if
        `after_build__finalize_and_save` is `True`.
    :var tags: A list of arbitrary strings, used to categorize this exds.
    """

    def __init__(self, exds_folder, name):
        self.exds_folder = exds_folder
        self.name = name
        super().__init__(self.exds_folder, self.name)
        self.default_lock_type = 'exds'
        self.exds_class = None
        self.source = None
        self.source_configuration = {}
        self.options = { }
        self.after_build__finalize_and_save = True
        self.after_save__auto_lock = True
        self.tags = []


    def create_exds(self):
        return self.exds_class(self)


    def validate(self):
        pass


