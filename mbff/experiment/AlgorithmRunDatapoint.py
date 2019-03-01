class AlgorithmRunDatapoint:
    def __init__(self, algorithm_run):
        self.parameters = algorithm_run.parameters
        self.label = self.parameters['label']
        self.ID = algorithm_run.ID
        self.algorithm_name = algorithm_run.algorithm_name
        self.start_time = algorithm_run.start_time
        self.end_time = algorithm_run.end_time
        self.duration = algorithm_run.duration
        self.classifier_classname = algorithm_run.classifier_classname
        self.classifier_evaluation = algorithm_run.evaluation
