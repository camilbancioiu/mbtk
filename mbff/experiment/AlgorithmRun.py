import time
import numpy
from string import Template

class AlgorithmRun:

    def __init__(self, exds, configuration, parameters):
        self.configuration = configuration
        self.parameters = parameters
        self.ID = ''
        self.algorithm = self.configuration['algorithm']
        self.algorithm_name = '.'.join([self.algorithm.__module__, self.algorithm.__name__])
        self.exds = exds

        self.start_time = 0
        self.end_time = 0
        self.duration = 0.0

        self.selected_features = []
        self.classifier_class = self.configuration['classifier']
        self.classifier_classname = '.'.join([self.classifier_class.__module__, self.classifier_class.__name__])
        self.classifier_evaluation = {}

        self.datasetmatrix_train = None
        self.datasetmatrix_test = None
        self.samples_train = None
        self.samples_test = None
        self.objective_index = -1
        self.objective_train = None
        self.objective_test = None
        self.predictions = {}

        # self.label could be a Template string.
        self.label = self.configuration['label']
        if isinstance(self.label, Template):
            self.label = self.label.safe_substitute(self.parameters)


    def run(self):
        self.start_time = time.time()
        self.selected_features = self.algorithm(self.exds.matrix, self.parameters)
        self.end_time = time.time()
        self.duration = (self.end_time - self.start_time) * 1000.0

        self.classifier_evaluation = self.evaluate_classifier()


    def evaluate_classifier(self):
        self.datasetmatrix_train = self.exds.matrix_train.select_columns_X(self.selected_features)
        self.datasetmatrix_test = self.exds.matrix_test.select_columns_X(self.selected_features)
        self.objective_index = self.parameters['objective_index']

        self.samples_train = self.datasetmatrix_train.X
        self.samples_test = self.datasetmatrix_test.X
        self.objective_train = self.datasetmatrix_train.get_column_Y(self.objective_index)
        self.objective_test = self.datasetmatrix_test.get_column_Y(self.objective_index)

        self.classifier = self.classifier_class()
        self.classifier.fit(self.samples_train, self.objective_train)
        self.predictions = self.classifier.predict(self.samples_test)
        return self.evaluate_classifier_predictions(self.objective_test, self.predictions)


    def evaluate_classifier_predictions(self, expected, predictions):
        n_expected = numpy.logical_not(expected)
        n_predictions = numpy.logical_not(predictions)
        evaluation = {}
        evaluation['TP'] = numpy.sum(numpy.logical_and(expected, predictions)).item()
        evaluation['TN'] = numpy.sum(numpy.logical_and(n_expected, n_predictions)).item()
        evaluation['FP'] = numpy.sum(numpy.logical_and(n_expected, predictions)).item()
        evaluation['FN'] = numpy.sum(numpy.logical_and(expected, n_predictions)).item()
        return evaluation


