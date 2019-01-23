import random
import os
import shutil
import operator
from pathlib import Path
import numpy
import scipy
import scipy.io
import functools as F
import pickle
from collections import Counter

import dataset_rcv1v2 as rcv1v2
import utilities as util
import matplotlib.pyplot as Plotter
from mpprint import mpprint

lock_types = ['exds', 'ks_gamma']

def build_experimental_dataset(definition):
    mpprint("Building ExperimentalDataset \"{0}\"".format(definition.name))
    mpprint("Loading documents from industry {0}...".format(definition.industry))
    R = rcv1v2.fetch_rcv1_token(definition.industry)
    mpprint("RCV1 dataset industry stats before trimming: " + R.get_stats())
    mpprint("Trimming topics at frequencies {0}...".format(definition.trim_freqs))
    rcv1v2.trim_rcv1_token(R, definition.trim_freqs)
    mpprint("RCV1 dataset industry stats: " + R.get_stats())

    mpprint("Building experimental dataset for industry...")
    ExDs = ExperimentalDataset(definition)
    ExDs.set_source_RCV1Dataset(R)
    ExDs.perform_random_dataset_split()
    mpprint("Splitting experimental dataset at {0} train rows proportion...".format(definition.train_rows_proportion))
    #mpprint("Experimental dataset stats: " + ExDs.get_stats_string())
    test_dataset_split(R, ExDs.train_rows, ExDs.test_rows, ExDs.Xtrain, ExDs.Xtest, ExDs.Ytrain, ExDs.Ytest)
    ExDs.save()
    mpprint("Experimental dataset saved to folder {0}.".format(definition.folder))
    return ExDs

class ExperimentalDatasetDefinition():
    def __init__(self, name, industry, train_rows_proportion, trim_freqs, folder=None):
        self.name = name
        self.industry = industry
        if folder != None:
            self.folder = folder
        else:
            self.folder = 'ExperimentalDatasets/' + self.name
        self.train_rows_proportion = train_rows_proportion
        self.trim_freqs = trim_freqs
        self.tags = []
        self.random_seed = 42

    def __str__(self):
        return 'ExDs {}, Folder {}, Industry {}, Tags {}'.format(self.name, self.folder, self.industry, self.tags)

    def pack_as_dict(self):
        return {self.name : self}

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
        self.R = None
        self.row_count = 0
        self.train_rows = None
        self.test_rows = None
        self.X = self.Y = None
        self.Xtrain = self.Xtest = None
        self.Ytrain = self.Ytest = None
        self.Y_csc = self.Ytrain_csc = self.Ytest_csc = None
        self.matrix_names = ['X', 'Y', 'Xtrain', 'Ytrain', 'Xtest', 'Ytest']
        self.definition = definition
        self.train_rows_proportion = self.definition.train_rows_proportion
        self.topics = None

    def set_source_RCV1Dataset(self, R):
        self.R = R
        self.X = R.words_csr
        self.Y = R.topics_csr
        self.row_count = self.X.get_shape()[0]
        self.topics = list(self.R.topics)

    def perform_random_dataset_split(self):
        train_rows_count = int((self.row_count) * self.train_rows_proportion)
        rows = range(self.row_count)
        random.seed(self.definition.random_seed)
        shuffled_rows = random.sample(rows, len(rows))
        self.train_rows = shuffled_rows[0:train_rows_count]
        self.test_rows = shuffled_rows[train_rows_count:]
        self.update_dataset_split()
        self.ready = True

    def update_dataset_split(self):
        self.Xtrain = util.keep_matrix_rows(self.X, self.train_rows)
        self.Xtest = util.keep_matrix_rows(self.X, self.test_rows)
        self.Ytrain = util.keep_matrix_rows(self.Y, self.train_rows)
        self.Ytest = util.keep_matrix_rows(self.Y, self.test_rows)

        self.remove_empty_training_words()
        self.update_csc_versions()

    def update_csc_versions(self):
        self.Y_csc = self.Y.tocsc()
        self.Ytrain_csc = self.Ytrain.tocsc()
        self.Ytest_csc = self.Ytest.tocsc()

    def remove_empty_training_words(self):
        Xtrain_csc = self.Xtrain.tocsc()
        non_null_filter = (lambda i: numpy.sum(Xtrain_csc.getcol(i).toarray().ravel()) != 0)
        words_to_keep = list(filter(non_null_filter, range(self.Xtrain.get_shape()[1])))
        self.X = util.keep_matrix_columns(self.X, words_to_keep)
        self.Xtrain = util.keep_matrix_columns(self.Xtrain, words_to_keep)
        self.Xtest = util.keep_matrix_columns(self.Xtest, words_to_keep)
        self.row_count = self.X.get_shape()[0]

    def save(self):
        folder = self.definition.folder + '/data'
        path = Path('./' + folder)
        path.mkdir(parents=True, exist_ok=True)

        if self.definition.folder_is_locked():
            raise ExperimentalDatasetError(self.definition, 'ExDs folder is locked, cannot save.')

        for matrix_name in self.matrix_names:
            util.save_matrix(folder, matrix_name, getattr(self, matrix_name))
        numpy.savetxt(folder + '/train_rows_proportion.txt', numpy.array([self.train_rows_proportion]))
        simpleformat = '%.1u'
        numpy.savetxt(folder + '/train_rows.txt', self.train_rows, fmt=simpleformat)
        numpy.savetxt(folder + '/test_rows.txt', self.test_rows, fmt=simpleformat)
        util.save_list_to_file(list(self.topics), 'topic_list', folder=folder)

    def load(self):
        if self.definition.folder_exists():
            folder = self.definition.folder + '/data'
        else:
            raise ExperimentalDatasetError(self.definition, 'ExDs folder does not exist.')
        for matrix_name in self.matrix_names:
            matrix = util.load_matrix(folder, matrix_name)
            setattr(self, matrix_name, matrix)
        self.train_rows_proportion = numpy.loadtxt(folder + '/train_rows_proportion.txt').item()
        self.train_rows = numpy.loadtxt(folder + '/train_rows.txt')
        self.test_rows = numpy.loadtxt(folder + '/test_rows.txt')
        self.row_count = self.X.get_shape()[0]
        self.topics = util.load_list_from_file('topic_list', folder=folder)
        try:
            self.update_csc_versions()
        except:
            pass

    def get_Y_column(self, col):
        return self.Y_csc.getcol(col).transpose().toarray().ravel()

    def get_Ytrain_column(self, col):
        return self.Ytrain_csc.getcol(col).transpose().toarray().ravel()

    def get_Ytest_column(self, col):
        return self.Ytest_csc.getcol(col).transpose().toarray().ravel()

    def get_Y_count(self):
        return self.Y.get_shape()[1]


class ExperimentalDatasetError(Exception):
    def __init__(self, exds_definition, message):
        self.exds_definition = exds_definition
        self.message = message

## Tests
def test_dataset_split(dataset, train_rows, test_rows, Xtrain, Xtest, Ytrain, Ytest):
    X = dataset.words_csr
    Y = dataset.topics_csr

    total_rows_count = X.get_shape()[0]
    if total_rows_count != (len(train_rows) + len(test_rows)):
        raise Exception('Bad dataset split (1)')
    if len(set(train_rows)) != len(train_rows):
        raise Exception('Bad dataset split (2.1)')
    if len(set(test_rows)) != len(test_rows):
        raise Exception('Bad dataset split (2.2)')
    if len(set(train_rows).intersection(set(test_rows))) != 0:
        raise Exception('Bad dataset split (3)')
    if len(train_rows) != Xtrain.get_shape()[0]:
        raise Exception('Bad dataset split (4.1)')
    if len(train_rows) != Ytrain.get_shape()[0]:
        raise Exception('Bad dataset split (4.2)')
    if len(test_rows) != Xtest.get_shape()[0]:
        raise Exception('Bad dataset split (4.3)')
    if len(test_rows) != Ytest.get_shape()[0]:
        raise Exception('Bad dataset split (4.4)')

    total_cols_count = Ytrain.get_shape()[1]
    for i in range(total_cols_count):
        if Ytrain.getcol(i).toarray().ravel().any() == False:
            raise Exception('Empty Ytrain column: {0}'.format(i))

def test_save_load_experimental_dataset():
    R = rcv1v2.fetch_rcv1_token()
    rcv1v2.trim_rcv1_token(R)

    E = ExperimentalDataset()
    E.set_source_RCV1Dataset(R)
    E.set_train_rows_proportion(train_rows_proportion)
    test_dataset_split(R, E.train_rows, E.test_rows, E.Xtrain, E.Xtest, E.Ytrain, E.Ytest)

    E.save('test_experimental_dataset')
    
    V = ExperimentalDataset()
    V.load('test_experimental_dataset')
    test_dataset_split(R, V.train_rows, V.test_rows, V.Xtrain, V.Xtest, V.Ytrain, V.Ytest)

    def exc(s):
        raise Exception('failed i/o ({0})'.format(s))

    if E.train_rows_proportion != V.train_rows_proportion:
        exc(1)
    if not numpy.array_equal(E.train_rows, V.train_rows):
        exc(2.1)
    if not numpy.array_equal(E.test_rows, V.test_rows):
        exc(2.2)
    
    if not E.X.get_shape() == V.X.get_shape():
        exc(3.1)
    if not E.Y.get_shape() == V.Y.get_shape():
        exc(3.2)

    if not E.Xtrain.get_shape() == V.Xtrain.get_shape():
        exc(3.3)
    if not E.Ytrain.get_shape() == V.Ytrain.get_shape():
        exc(3.4)
    
    if not E.Xtest.get_shape() == V.Xtest.get_shape():
        exc(3.5)
    if not E.Ytest.get_shape() == V.Ytest.get_shape():
        exc(3.6)

    if (E.X != V.X).nnz != 0:
        exc(3.7)
    if (E.Y != V.Y).nnz != 0:
        exc(3.8)
    if (E.Xtrain != V.Xtrain).nnz != 0:
        exc(4.1)
    if (E.Xtest != V.Xtest).nnz != 0:
        exc(4.2)
    if (E.Ytrain != V.Ytrain).nnz != 0:
        exc(4.3)
    if (E.Ytest != V.Ytest).nnz != 0:
        exc(4.4)
