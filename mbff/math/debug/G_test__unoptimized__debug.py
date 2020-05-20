from mbff.math.G_test__unoptimized import G_test


class G_test_debug(G_test):

    def __init__(self, datasetmatrix, parameters):
        super().__init__(datasetmatrix, parameters)
        self.debug = self.parameters.get('ci_test_debug', 0)


    def conditionally_independent(self, X, Y, Z):
        result = super().conditionally_independent(X, Y, Z)
        self.print_ci_test_result(result)
        return result


    def end(self):
        super().end()
        save_path = self.parameters.get('ci_test_results_path__save', None)
        if self.debug >= 1: print('CI test results saved to {}'.format(save_path))


    def print_ci_test_result(self, result):
        if self.debug >= 1:
            if result.accurate():
                if self.parameters.get('ci_test_results__print_accurate', True):
                    print(result)
            if not result.accurate():
                if self.parameters.get('ci_test_results__print_inaccurate', True):
                    print(result)
