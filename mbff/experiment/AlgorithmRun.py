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


    def run(self):
        self.start_time = time.time()
        self.selected_features = self.algorithm(self.exds.matrix, self.parameters)
        self.end_time = time.time()
        self.duration = (self.end_time - self.start_time) * 1000.0

        self.classifier_evaluation = self.evaluate_classifier()


    def evaluate_classifier(self):
        datasetmatrix_train = self.exds.matrix_train.select_columns_X(self.selected_features)
        datasetmatrix_test = self.exds.matrix_test.select_columns_X(self.selected_features)
        objective_index = self.parameters['objective_index']

        samples_train = datasetmatrix_train.X
        samples_test = datasetmatrix_test.X
        objective_train = datasetmatrix_train.get_column_Y(objective_index)
        objective_test = datasetmatrix_test.get_column_Y(objective_index)

        self.classifier = self.classifier_class()
        self.classifier.fit(samples_train, objective_train)
        predictions = self.classifier.predict(samples_test)
        return self.evaluate_classifier_predictions(objective_test, predictions)


    def evaluate_classifier_predictions(self, expected, predictions):
        n_expected = numpy.logical_not(expected)
        n_predictions = numpy.logical_not(predictions)
        evaluation = collections.OrderedDict()
        evaluation['TP'] = numpy.asscalar(numpy.sum(numpy.logical_and(expected, predictions)))
        evaluation['TN'] = numpy.asscalar(numpy.sum(numpy.logical_and(n_expected, n_predictions)))
        evaluation['FP'] = numpy.asscalar(numpy.sum(numpy.logical_and(n_expected, predictions)))
        evaluation['FN'] = numpy.asscalar(numpy.sum(numpy.logical_and(expected, n_predictions)))
        return evaluation


