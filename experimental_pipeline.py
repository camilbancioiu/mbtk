import numpy
import scipy
import scipy.sparse
import scipy.io
import sklearn.naive_bayes
import time
import itertools
import pickle
from multiprocessing import Pool
from glob import glob
import sys

from datetime import datetime
from copy import deepcopy
from pathlib import Path
import utilities as util
from experimental_dataset import *
from experimental_results import *
import feature_selection as FS 
from mpprint import mpprint

class ExperimentDefinition():
    def __init__(self, name, exds_definition, parameters, config, folder=None, parameter_order=None):
        self.name = name
        self.exds_definition = exds_definition
        self.parameters = parameters
        self.config = config
        self.tags = []
        if folder != None:
            self.folder = folder
        else:
            self.folder = 'ExperimentalRuns/' + self.name
        if parameter_order != None:
            self.parameters = util.create_ordered_dict(parameter_order, self.parameters)

    def folder_is_locked(self):
        return Path(self.folder + '/locked').exists()

    def folder_exists(self):
        return Path(self.folder).exists()

    def lock_folder(self):
        folder = self.folder
        if self.folder_is_locked():
            print('{}: Folder {} is already locked.'.format(self.name, folder))
        else:
            with open(folder + '/locked', 'w') as f:
                f.write('locked')
                print('{}: Folder has been locked with file {}.'.format(self.name, folder + '/locked'))

    def delete_folder(self):
        if not self.folder_exists():
            print('{}: Experiment folder does not exist.'.format(self.name))
            return
        if self.folder_is_locked():
            print('{}: Experiment folder is locked, cannot delete.'.format(self.name))
            return
        shutil.rmtree(self.folder)
        print('{}: Experiment folder deleted.'.format(self.name))

    def unlock_folder(self):
        folder = self.folder
        if not self.folder_is_locked():
            print('{}: Folder {} is not locked.'.format(self.name, folder))
        else:
            os.remove(folder + '/locked')
            print('{}: Folder {} has been unlocked'.format(self.name, folder))

class Experiment():
    def __init__(self, definition):
        self.definition = definition
        self.ExDs = ExperimentalDataset(self.definition.exds_definition)
        self.start_time = None
        self.end_time = None
        self.duration = None
        self.parameter_configurations = None

    def prepare(self):
        self.ensure_folder()
        if self.definition.folder_is_locked():
            raise ExperimentError(self.definition, 'Experiment folder is locked, cannot start')
        self.delete_saved_runs()
        self.delete_subfolder('logs')
        self.delete_subfolder('samples')
        mpprint('Preparing experiment {}...'.format(self.definition.name))
        self.ExDs.load()
        mpprint('ExDs {} loaded.'.format(self.ExDs.definition.name))
        FS.prepare(self.definition)
        self.parameter_configurations = itertools.product(*self.definition.parameters.values())

    def ensure_folder(self):
        path = Path('./' + self.definition.folder)
        path.mkdir(parents=True, exist_ok=True)

    def ensure_subfolder(self, subfolder):
        path = Path('./' + self.definition.folder + '/' + subfolder)
        path.mkdir(parents=True, exist_ok=True)

    def run(self):
        self.start_time = time.time()
        self.print_header()
        self.prepare()

        self.ensure_subfolder('logs')

        run_parallelism = self.definition.config['run_parallelism']

        if run_parallelism > 1:
            with Pool(run_parallelism) as pool:
                util.consume(pool.map(self.run_algorithm, self.parameter_configurations))
        else:
            for parameter_configuration in self.parameter_configurations:
                print(parameter_configuration)
                self.run_algorithm(parameter_configuration)

        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.print_footer()

    def run_algorithm(self, parameter_configuration):
        parameter_configuration = dict(zip(self.definition.parameters.keys(), parameter_configuration))
        alg_run_name = self.get_algorithm_run_name(parameter_configuration)
        if not ("algrun_stdout" in self.definition.config):
            self.definition.config["algrun_stdout"] = "file"

        algrun_stdout = None
        if self.definition.config["algrun_stdout"] == "file":
            output_file_name = '{}/logs/{}.log'.format(self.definition.folder, alg_run_name)
            algrun_stdout = open(output_file_name, 'wt')
        if self.definition.config["algrun_stdout"] == "sys.stdout":
            algrun_stdout = sys.__stdout__

        sys.stdout = algrun_stdout
        algorithm_run = AlgorithmRun(alg_run_name, self.definition, self.ExDs, parameter_configuration)
        self.print_begin_run(algorithm_run)
        algorithm_run.run()
        self.print_end_run(algorithm_run)

        if not ("algrun_saving" in self.definition.config):
            self.definition.config['algrun_saving'] = 'save_samples'

        if self.definition.config['algrun_saving'] == 'save_full':
            self.save_algorithm_run(algorithm_run)

        if self.definition.config['algrun_saving'] == 'save_samples':
            self.save_algrun_sample(AlgorithmRunSample(algorithm_run))

        if self.definition.config['algrun_saving'] == 'save_full_and_samples':
            self.save_algorithm_run(algorithm_run)
            self.save_algrun_sample(AlgorithmRunSample(algorithm_run))

        sys.stdout = sys.__stdout__

    def save_algorithm_run(self, algorithm_run):
        self.ensure_subfolder('runs')
        filename = '{}/runs/{}.pickle'.format(self.definition.folder, algorithm_run.name)
        with open(filename, 'wb') as f:
            pickle.dump(algorithm_run, f)

    def get_algorithm_run_name(self, parameter_configuration):
        output = '{}_{}_T{}'.format(self.definition.name, parameter_configuration['algorithm'], parameter_configuration['target'])
        extra_parameter_names = [pn for pn in self.definition.parameters.keys() if pn not in ['algorithm', 'target']]
        for pn in extra_parameter_names:
            output += '_{}{}'.format(pn, parameter_configuration[pn])
        return output

    def map_over_saved_runs(self, function):
        folder = self.definition.folder + '/runs/'
        filenames = glob(folder + '*.pickle')
        results = []
        count = len(filenames)
        i = 0
        print()
        for filename in filenames:
            with open(filename, 'rb') as f:
                run = pickle.load(f)
                result = function(run)
                results.append(result)
                print("\rMapping over saved AlgorithmRuns... {}/{}".format(i, count), end='')
            i = i + 1
        print("\rMapping over saved AlgorithmRuns... {}/{}".format(i, count), end='')
        print()
        return results

    def load_saved_runs(self):
        folder = self.definition.folder + '/runs/'
        filenames = glob(folder + '*.pickle')
        algorithm_runs = []
        for filename in filenames:
            with open(filename, 'rb') as f:
                algorithm_runs.append(pickle.load(f))
        return algorithm_runs

    def generate_algrun_samples(self):
        samples = self.map_over_saved_runs(AlgorithmRunSample)
        self.ensure_subfolder('samples')
        filename = '{}/samples/algrun-samples.pickle'.format(self.definition.folder)
        with open(filename, 'wb') as f:
            pickle.dump(samples, f)

    # TODO Make this method safe for parallel runs,
    # if at all required. Maybe not needed, since the AlgRuns are run on separate processes,
    # but the parent Experiment object is single-process.
    def save_algrun_sample(self, sample):
        samples = self.load_algrun_samples()
        samples.append(sample)
        self.save_algrun_samples(samples)

    def load_algrun_samples(self):
        filename = '{}/samples/algrun-samples.pickle'.format(self.definition.folder)
        samples = []
        try:
            with open(filename, 'rb') as f:
                samples = pickle.load(f)
        except FileNotFoundError:
            samples = []
        return samples

    def save_algrun_samples(self, samples):
        self.ensure_subfolder('samples')
        filename = '{}/samples/algrun-samples.pickle'.format(self.definition.folder)
        with open(filename, 'wb') as f:
            samples = pickle.dump(samples, f)
        return samples

    def delete_saved_runs(self):
        self.delete_subfolder('runs')

    def delete_subfolder(self, subfolder):
        if self.definition.folder_is_locked():
            print('{}: Experiment folder is locked, cannot delete subfolder \'{}\'.'.format(self.definition.name, subfolder))
            return
        try:
            shutil.rmtree(self.definition.folder + '/' + subfolder)
        except:
            print('{}: Experiment subfolder \'{}\' doesn\'t exist.'.format(self.definition.name, subfolder))
        self.ensure_subfolder(subfolder)

    def print_header(self):
        mpprint('Experiment {} began at {}.'.format(self.definition.name, util.localftime(self.start_time)))
        mpprint('-------------------------')
        mpprint()

    def print_footer(self):
        mpprint('Experiment {} ended at {}.'.format(self.definition.name, util.localftime(self.end_time)))
        mpprint('Experiment duration: {}'.format(str(self.duration)))

    def print_begin_run(self, algorithm_run):
        mpprint('Beginning AlgorithmRun {}... '.format(algorithm_run.name))

    def print_end_run(self, algorithm_run):
        mpprint('AlgorithmRun complete. Duration {}.'.format(str(algorithm_run.duration)))



class AlgorithmRun():
    def __init__(self, name, experiment_definition, exds, parameter_configuration):
        self.parameter_configuration = parameter_configuration
        self.algorithm_key = self.parameter_configuration['algorithm']
        self.algorithm = FS.get_algorithm(self.algorithm_key)
        self.target = self.parameter_configuration['target']
        self.experiment_definition = experiment_definition
        self.start_time = self.end_time = 0

        self.ExDs = exds

        self.result = None
        self.classifier = sklearn.naive_bayes.BernoulliNB()
        self.classifier_evaluation = {}
        self.duration = 0.0

        self.name = name
    
    def run(self):
        self.start_time = time.time()
        self.selected_features = self.select_features()
        self.end_time = time.time()
        self.duration = (self.end_time - self.start_time) * 1000.0

        (reducedXtrain, reducedXtest) = self.reduce_dataset(self.selected_features)
        Ytrain = self.ExDs.get_Ytrain_column(self.target)
        Ytest = self.ExDs.get_Ytest_column(self.target)
        self.train_classifier(reducedXtrain, Ytrain)
        self.classifier_evaluation = self.test_classifier(reducedXtest, Ytest)

    def select_features(self):
        X = self.ExDs.X
        Y = self.ExDs.get_Y_column(self.target)
        dataset = {'X': X, 'Y': Y}
        selected_features = self.algorithm(dataset, self.parameter_configuration)
        return selected_features

    def reduce_dataset(self, selected_features):
        return (util.keep_matrix_columns(self.ExDs.Xtrain, selected_features),
                util.keep_matrix_columns(self.ExDs.Xtest, selected_features))

    def train_classifier(self, Xtrain, Ytrain):
        self.classifier.fit(Xtrain, Ytrain)

    def test_classifier(self, Xtest, Ytest):
        classification = self.classifier.predict(Xtest)
        return util.get_classifier_evaluation(Ytest, classification)

    def accuracy(self):
        TP = self.classifier_evaluation['TP']
        FP = self.classifier_evaluation['FP']
        TN = self.classifier_evaluation['TN']
        FN = self.classifier_evaluation['FN']
        return (TP + TN) * 1.0 / (TP + TN + FP + FN)

    def indexed_accuracy(self):
        return (self.parameter_configuration, self.accuracy())

    def todict(self, add_selected_features):
        d = {}
        d.update({
                'experiment' : self.experiment_definition.name,
                'exds': self.experiment_definition.exds_definition.name,
                'duration': self.duration
                })
        d.update(self.parameter_configuration)
        d.update(self.classifier_evaluation)
        if add_selected_features:
            d['selected_features'] = self.selected_features
        return d

    def csv_keys(parameter_keys, add_selected_features):
        keys = ['experiment', 'exds']
        keys.extend(parameter_keys)
        keys.append('duration')
        keys.extend(['TP', 'TN', 'FP', 'FN'])
        if add_selected_features:
            keys.append('selected_features')
        return keys

class AlgorithmRunSample():
    def __init__(self, algorithm_run):
        pc = algorithm_run.parameter_configuration
        self.target = pc['target']
        self.algorithm = pc['algorithm']
        self.Q = pc['Q']
        try:
            self.K = pc['K']
        except:
            pass
        self.TP = algorithm_run.classifier_evaluation['TP']
        self.FP = algorithm_run.classifier_evaluation['FP']
        self.TN = algorithm_run.classifier_evaluation['TN']
        self.FN = algorithm_run.classifier_evaluation['FN']
        self.duration = algorithm_run.duration
        self.result = algorithm_run.result
        self.accuracy = algorithm_run.accuracy()

class ExperimentError(Exception):
    def __init__(self, experiment_definition, message):
        self.experiment_definition = experiment_definition
        self.message = message
