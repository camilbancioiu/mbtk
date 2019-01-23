import numpy
import scipy
import scipy.io
import pickle
from .. import utilities as util

class DatasetMatrix:
    def __init__(self, label):
        self.label = label
        self.X = None
        self.Y = None
        self.row_labels = []
        self.column_labels_X = []
        self.column_labels_Y = []


    def save(self, folder):
        folder += '/' + label
        util.ensure_folder(folder)

        util.save_matrix(folder, "X", self.X)
        util.save_matrix(folder, "Y", self.Y)

        with open(folder + '/row_labels.txt', mode='wt') as f:
            f.write('\n'.join(self.row_labels))

        with open(folder + '/column_labels_X.txt', mode='wt') as f:
            f.write('\n'.join(self.column_labels_X))

        with open(folder + '/column_labels_Y.txt', mode='wt') as f:
            f.write('\n'.join(self.column_labels_Y))


    def load(self, folder):
        folder += '/' + label

        self.X = util.load_matrix(folder, "X")
        self.Y = util.load_matrix(folder, "Y")

        with open(folder + '/row_labels.txt', mode='rt') as f:
            self.row_labels = map(str.stip, list(f))

        with open(folder + '/column_labels_X.txt', mode='rt') as f:
            self.column_labels_X = map(str.stip, list(f))

        with open(folder + '/column_labels_Y.txt', mode='rt') as f:
            self.column_labels_Y = map(str.stip, list(f))
