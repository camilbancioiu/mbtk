import random
import os
import shutil
from pathlib import Path

from mbff.datasets.ExperimentalDatasetDefinition import ExperimentalDatasetDefinition
from mbff.datasets.DatasetMatrix import DatasetMatrix

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


