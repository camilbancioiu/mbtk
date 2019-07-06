import pickle
from mbff.math.CITestResult import CITestResult


class DSeparationCITest:

    def __init__(self, datasetmatrix, parameters):
        self.parameters = parameters
        self.source_bn = self.parameters.get('source_bayesian_network', None)
        self.debug = self.parameters.get('ci_test_debug', False)
        self.ci_test_results = []


    def conditionally_independent(self, X, Y, Z):
        result = CITestResult()
        result.start_timing()
        independent = self.source_bn.conditionally_independent(X, Y, Z)
        result.end_timing()

        result.set_independent(independent, 0)
        result.set_variables(X, Y, Z)
        result.computed_d_separation = independent
        result.set_statistic('None', 0, dict())
        result.set_distribution('None', 0, dict())

        self.ci_test_results.append(result)

        if self.debug: print(result)

        return result.independent


    def end(self):
        save_path = self.parameters.get('ci_test_results_path__save', None)
        if save_path is not None:
            with save_path.open('wb') as f:
                pickle.dump(self.ci_test_results, f)
