import csv
import numpy
import scipy
import scipy.sparse
import scipy.io
import json
from pathlib import Path
import ast
import utilities as util



class ExperimentalSampleSet():
    def __init__(self, name=''):
        self.comparative_set = None
        self.samples = []
        self.name = name

    def fromComparativeSet(self, comparative_set):
        self.comparative_set = comparative_set
        self.name = self.comparative_set.identifier
        self.samples = []
        for crun in self.comparative_set.comparative_runs:
            for algorithm_run in crun.algorithm_runs:
                S = ExperimentalSample()
                S.fromAlgorithmRun(algorithm_run)
                self.samples.append(S)

    def tolist(self):
        return [sample.tolist() for sample in self.samples]

    def load(self, folder):
        self.samples = []
        fname = './{0}/{1}.csv'.format(folder, self.name)
        with open(fname, 'r', newline='') as f:
            def makesample(l):
                S = ExperimentalSample()
                S.fromlist(l)
                return S
            reader = csv.reader(f)
            self.samples = list(map(makesample, reader))
        # print('ESS loaded: ' + fname)

    def save(self, folder):
        path = Path('./' + folder)
        path.mkdir(parents=True, exist_ok=True)
        with open(folder + '/' + self.name + '.csv', 'w', newline='') as f:
            w = csv.writer(f, lineterminator='\n')
            w.writerows(self.tolist())

class ExperimentalSample():
    def __init__(self):
        self.algorithm_run = None

        self.algorithm_name     = ''
        self.Q                  = 0
        self.K                  = 0
        self.Tj                 = 0
        self.fs_duration        = 0
        self.cls_tp             = 0
        self.cls_tn             = 0
        self.cls_fp             = 0
        self.cls_fn             = 0
        self.cls_sensitivity    = 0
        self.cls_specificity    = 0
        self.cls_accuracy       = 0
        self.selected_features  = []

    def __str__(self):
        return '({algorithm_name}, Q={Q}, K={K}, Tj={Tj}, time={fs_duration}s, acc={cls_accuracy})'.format(**self.__dict__)

    def fromAlgorithmRun(self, algorithm_run):
        self.algorithm_run = algorithm_run

        self.algorithm_name     = self.algorithm_run.name
        self.Q                  = self.algorithm_run.parameters.get('Q', 0)
        self.K                  = self.algorithm_run.parameters.get('K', 0)
        self.Tj                 = self.algorithm_run.Tj
        self.fs_duration        = self.algorithm_run.feature_selection_duration
        self.cls_tp             = numpy.asscalar(self.algorithm_run.classifier_evaluation['TP'])
        self.cls_tn             = numpy.asscalar(self.algorithm_run.classifier_evaluation['TN'])
        self.cls_fp             = numpy.asscalar(self.algorithm_run.classifier_evaluation['FP'])
        self.cls_fn             = numpy.asscalar(self.algorithm_run.classifier_evaluation['FN'])
        self.selected_features  = self.algorithm_run.W
        self.update_classifier_metrics()

    def tolist(self):
        return [self.algorithm_name, 
                self.Q,
                self.K,
                self.Tj,
                self.fs_duration,
                self.cls_tp,
                self.cls_tn,
                self.cls_fp,
                self.cls_fn,
                self.selected_features]

    def fromlist(self, l):
        self.algorithm_name     = l[0]
        self.Q                  = int(l[1])
        self.K                  = int(l[2])
        self.Tj                 = int(l[3])
        self.fs_duration        = float(l[4])
        self.cls_tp             = int(l[5])
        self.cls_tn             = int(l[6])
        self.cls_fp             = int(l[7])
        self.cls_fn             = int(l[8])
        self.selected_features  = ast.literal_eval(l[9])
        self.update_classifier_metrics()

    def update_classifier_metrics(self):
        try:
            self.cls_sensitivity  = self.cls_tp / (self.cls_tp + self.cls_fn)
        except ZeroDivisionError:
            self.cls_sensitivity  = 0

        try:
            self.cls_specificity = self.cls_tn / (self.cls_tn + self.cls_fp)
        except ZeroDivisionError:
            self.cls_specificity = 0

        try:
            self.cls_accuracy = (self.cls_tp + self.cls_tn) / (self.cls_tp + self.cls_tn + self.cls_fp + self.cls_fn)
        except ZeroDivisionError:
            self.cls_accuracy = 0

