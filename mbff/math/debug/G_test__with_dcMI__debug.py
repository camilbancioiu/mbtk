from mbff.math.G_test__with_dcMI import G_test


class G_test_debug(G_test):

    def __init__(self, datasetmatrix, parameters):
        super().__init__(datasetmatrix, parameters)
        self.debug = self.parameters.get('ci_test_debug', 0)


    def conditionally_independent(self, X, Y, Z):
        result = super().conditionally_independent(X, Y, Z)
        self.print_ci_test_result(result)
        return result


    def prepare_JHT(self):
        jht_load_path = self.parameters.get('ci_test_jht_path__load', None)
        super().prepare_JHT()
        if self.debug >= 1:
            print('Loaded the JHT from {}.'.format(jht_load_path))



    def get_joint_entropy_term(self, *variables):
        misses = self.JHT_misses

        H = super().get_joint_entropy_term(*variables)

        if self.debug >= 2:
            jht_key = self.create_flat_variable_set(*variables)
            if misses != self.JHT_misses:
                print('\tJHT miss and update: store H={:8.6f} for {}'.format(H, jht_key))
            else:
                print('\tJHT hit: found H={:8.6f} for {}'.format(H, jht_key))


    def end(self):
        super().end()
        if self.debug >= 1:
            save_path = self.parameters.get('ci_test_results_path__save', None)
            print('CI test results saved to {}'.format(save_path))

        if self.debug >= 1:
            jht_save_path = self.parameters.get('ci_test_jht_path__save', None)
            print('JHT has been saved to {}'.format(jht_save_path))


    def print_ci_test_result(self, result):
        if self.debug >= 1:
            if result.accurate():
                if self.parameters.get('ci_test_results__print_accurate', True):
                    print(result)
            if not result.accurate():
                if self.parameters.get('ci_test_results__print_inaccurate', True):
                    print(result)
