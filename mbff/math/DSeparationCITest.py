from mbff.math.CITestResult import CITestResult


class DSeparationCITest:

    def __init__(self, datasetmatrix, parameters):
        self.source_bn = parameters.get('source_bayesian_network', None)
        self.ci_test_results = []


    def conditionally_independent(self, X, Y, Z):
        independent = self.source_bn.conditionally_independent(X, Y, Z)

        result = CITestResult()
        result.set_independent(independent, None)
        result.set_variables(X, Y, Z)
        result.computed_d_separation = independent

        self.ci_test_results.append(result)

        return result.independent
