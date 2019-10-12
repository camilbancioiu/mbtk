import humanize
from datetime import timedelta
import pprint


class AlgorithmRunDatapoint:
    def __init__(self, algorithm_run):
        self.parameters = self.pickleable_parameters(algorithm_run.parameters)
        self.ID = algorithm_run.ID
        self.algorithm_name = algorithm_run.algorithm_name
        self.selected_features = algorithm_run.selected_features
        self.start_time = algorithm_run.start_time
        self.end_time = algorithm_run.end_time
        self.duration = algorithm_run.duration


    def pickleable_parameters(self, parameters):
        newparameters = parameters.copy()
        for k, v in newparameters.items():
            if not callable(v):
                newparameters[k] = str(v)

        return newparameters


    def __str__(self):
        view = dict()
        view['ID'] = self.ID
        view['algorithm_name'] = self.algorithm_name
        view['parameters'] = pprint.pformat(self.parameters)
        view['select_features'] = pprint.pformat(self.selected_features, indent=2)
        view['duration'] = humanize.naturaldelta(timedelta(seconds=self.duration))
        view['exact_duration'] = self.duration
        format_string = (
            '{ID}\n'
            'Duration: {exact_duration:.2f}s ({duration})\n'
            '{algorithm_name}, with parameters:\n'
            '{parameters}\n')
        return format_string.format(**view)



class AlgorithmAndClassifierRunDatapoint(AlgorithmRunDatapoint):
    def __init__(self, algorithm_run):
        super().__init__(algorithm_run)
        self.classifier_classname = algorithm_run.classifier_classname
        self.classifier_evaluation = algorithm_run.classifier_evaluation
