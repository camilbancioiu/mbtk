import time
import numpy

class AlgorithmRun:

    def __init__(self, label, exds, parameters):
        self.label = label
        self.parameters = parameters
        self.algorithm = self.parameters['algorithm']
        self.exds = exds

        self.start_time = 0
        self.end_time = 0
        self.duration = 0.0

        self.classifier_class = self.parameters['classifier_class']
        self.classifier_evaluation = {}

        self.datasetmatrix_train = None
        self.datasetmatrix_test = None
        self.samples_train = None
        self.samples_test = None
        self.objective_index = -1
        self.objective_train = None
        self.objective_test = None
        self.predictions = {}


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
        evaluation['TP'] = numpy.asscalar(numpy.sum(numpy.logical_and(expected, predictions)))
        evaluation['TN'] = numpy.asscalar(numpy.sum(numpy.logical_and(n_expected, n_predictions)))
        evaluation['FP'] = numpy.asscalar(numpy.sum(numpy.logical_and(n_expected, predictions)))
        evaluation['FN'] = numpy.asscalar(numpy.sum(numpy.logical_and(expected, n_predictions)))
        return evaluation


