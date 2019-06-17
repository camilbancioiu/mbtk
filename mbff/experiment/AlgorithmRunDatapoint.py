class AlgorithmRunDatapoint:
    def __init__(self, algorithm_run):
        self.parameters = self.pickleable_parameters(algorithm_run.parameters)
        self.label = algorithm_run.label
        self.ID = algorithm_run.ID
        self.algorithm_name = algorithm_run.algorithm_name
        self.selected_features = algorithm_run.selected_features
        self.start_time = algorithm_run.start_time
        self.end_time = algorithm_run.end_time
        self.duration = algorithm_run.duration


    def pickleable_parameters(self, parameters):
        newparameters = parameters.copy()
        for k, v in newparameters.items():
            if callable(v):
                newparameters[k] = str(v)



class AlgorithmAndClassifierRunDatapoint(AlgorithmRunDatapoint):
    def __init__(self, algorithm_run):
        super().__init__(algorithm_run)
        self.classifier_classname = algorithm_run.classifier_classname
        self.classifier_evaluation = algorithm_run.classifier_evaluation
