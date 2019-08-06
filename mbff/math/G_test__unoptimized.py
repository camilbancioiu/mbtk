import pickle

from mbff.math.CITestResult import CITestResult

import mbff.math.infotheory as infotheory
from mbff.math.PMF import PMF, CPMF
from mbff.math.Variable import JointVariables

from scipy.stats import chi2
import gc


class G_test:

    def __init__(self, datasetmatrix, parameters):
        self.parameters = parameters
        self.debug = self.parameters.get('ci_test_debug', 0)
        self.datasetmatrix = None
        self.matrix = None
        self.column_values = None
        self.N = None
        if datasetmatrix is not None:
            self.datasetmatrix = datasetmatrix
            self.matrix = self.datasetmatrix.X
            self.column_values = self.datasetmatrix.get_values_per_column('X')
            self.N = self.matrix.get_shape()[0]
        self.significance = parameters.get('ci_test_significance', 0)
        self.omega = parameters.get('omega', None)
        self.source_bn = parameters.get('source_bayesian_network', None)
        self.ci_test_name = '.'.join([self.__module__, self.__class__.__name__])
        self.parameters['ci_test_name'] = self.ci_test_name
        self.ci_test_results = []


    def conditionally_independent(self, X, Y, Z):
        # Load the actual variable instances (samples) from the
        # datasetmatrix.
        (VarX, VarY, VarZ) = self.load_variables(X, Y, Z)

        result = self.G_test_conditionally_independent(VarX, VarY, VarZ, X, Y, Z)

        if self.source_bn is not None:
            result.computed_d_separation = self.source_bn.d_separated(X, Z, Y)

        self.ci_test_results.append(result)
        self.print_ci_test_result(result)

        # Garbage collection required to deallocate variable instances.
        gc.collect()

        return result.independent


    def G_test_conditionally_independent(self, VarX, VarY, VarZ, X, Y, Z):
        result = CITestResult()
        result.start_timing()

        (PrXYcZ, PrXcZ, PrYcZ, PrZ) = self.calculate_cpmfs(VarX, VarY, VarZ)
        G = self.G_value(PrXYcZ, PrXcZ, PrYcZ, PrZ)
        DF = self.calculate_degrees_of_freedom(PrXYcZ, PrXcZ, PrYcZ, PrZ, X, Y, Z)

        p = chi2.cdf(G, DF)
        independent = None
        if p < self.significance:
            independent = True
        else:
            independent = False

        result.end_timing()
        result.index = len(self.ci_test_results)
        result.set_independent(independent, self.significance)
        result.set_variables(VarX, VarY, VarZ)
        result.set_statistic('G', G, dict())
        result.set_distribution('chi2', p, {'DoF': DF})

        result.extra_info = ' DoF {}'.format(DF)

        return result


    def G_value(self, PrXYcZ, PrXcZ, PrYcZ, PrZ):
        cMI = infotheory.conditional_mutual_information(PrXYcZ, PrXcZ, PrYcZ, PrZ, base='e')
        return 2 * self.N * cMI


    def load_variables(self, X, Y, Z):
        VarX = self.datasetmatrix.get_variable('X', X)
        VarY = self.datasetmatrix.get_variable('X', Y)
        if len(Z) == 0:
            VarZ = self.omega
        else:
            VarZ = self.datasetmatrix.get_variables('X', Z)

        return (VarX, VarY, VarZ)


    def calculate_degrees_of_freedom(self, PrXYcZ, PrXcZ, PrYcZ, PrZ, X, Y, Z):
        method = self.parameters.get('ci_test_dof_computation_method', 'structural')
        if method == 'structural':
            return self.calculate_degrees_of_freedom__structural(PrXYcZ, PrXcZ, PrYcZ, PrZ, X, Y, Z)
        elif method == 'rowcol':
            return self.calculate_degrees_of_freedom__rowcol(PrXYcZ, PrXcZ, PrYcZ, PrZ, X, Y, Z)
        elif method == 'rowcol_minus_zerocells':
            return self.calculate_degrees_of_freedom__rowcol_minus_zerocells(PrXYcZ, PrXcZ, PrYcZ, PrZ, X, Y, Z)


    def calculate_degrees_of_freedom__structural(self, PrXYcZ, PrXcZ, PrYcZ, PrZ, X, Y, Z):
        X_val = len(self.column_values[X])
        Y_val = len(self.column_values[Y])

        Z_val = 1
        for z in Z:
            Z_val *= len(self.column_values[z])
        DoF = (X_val - 1) * (Y_val - 1) * Z_val
        return DoF


    def calculate_degrees_of_freedom__rowcol(self, PrXYcZ, PrXcZ, PrYcZ, PrZ, X, Y, Z):
        DoF = 0
        for (z, pz) in PrZ.items():
            PrX = PrXcZ.given(z)
            PrY = PrYcZ.given(z)

            expected_dof_xycz = (len(PrX) - 1) * (len(PrY) - 1)
            DoF += expected_dof_xycz

        return DoF


    def calculate_degrees_of_freedom__rowcol_minus_zerocells(self, PrXYcZ, PrXcZ, PrYcZ, PrZ, X, Y, Z):
        DoF = 0
        for (z, pz) in PrZ.items():
            PrXY = PrXYcZ.given(z)
            PrX = PrXcZ.given(z)
            PrY = PrYcZ.given(z)

            expected_dof_xycz = (len(PrX) - 1) * (len(PrY) - 1)
            for pxy in PrXY.values():
                if pxy == 0:
                    expected_dof_xycz -= 1
            DoF += expected_dof_xycz

        return DoF


    def calculate_cpmfs(self, VarX, VarY, VarZ):
        PrXYcZ = CPMF(JointVariables(VarX, VarY), VarZ)
        PrXcZ = CPMF(VarX, VarZ)
        PrYcZ = CPMF(VarY, VarZ)
        PrZ = PMF(VarZ)

        return (PrXYcZ, PrXcZ, PrYcZ, PrZ)


    def end(self):
        save_path = self.parameters.get('ci_test_results_path__save', None)
        if save_path is not None:
            with save_path.open('wb') as f:
                pickle.dump(self.ci_test_results, f)
        if self.debug >= 1: print('CI test results saved to {}'.format(save_path))


    def print_ci_test_result(self, result):
        if self.debug >= 1:
            if result.accurate():
                if self.parameters.get('ci_test_results__print_accurate', True):
                    print(result)
            if not result.accurate():
                if self.parameters.get('ci_test_results__print_inaccurate', True):
                    print(result)
