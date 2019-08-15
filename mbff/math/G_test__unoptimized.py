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
        self.ci_test_counter = 0
        self.gc_collect_rate = self.parameters.get('ci_test_gc_collect_rate', 0)





    def conditionally_independent(self, X, Y, Z):
        result = self.G_test_conditionally_independent(X, Y, Z)

        if self.source_bn is not None:
            result.computed_d_separation = self.source_bn.d_separated(X, Z, Y)

        self.ci_test_results.append(result)
        self.print_ci_test_result(result)

        self.ci_test_counter += 1
        if self.gc_collect_rate != 0:
            if self.ci_test_counter % self.gc_collect_rate == 0:
                gc.collect()

        return result.independent


    def G_test_conditionally_independent(self, X, Y, Z):
        (VarX, VarY, VarZ) = self.load_variables(X, Y, Z)

        result = CITestResult()
        result.start_timing()

        (PrXYcZ, PrXcZ, PrYcZ, PrZ) = self.calculate_cpmfs(VarX, VarY, VarZ)
        G = self.G_value(PrXYcZ, PrXcZ, PrYcZ, PrZ)
        DoF = self.calculate_degrees_of_freedom(PrXYcZ, PrXcZ, PrYcZ, PrZ, X, Y, Z)

        p = chi2.cdf(G, DoF)
        independent = None
        if p < self.significance:
            independent = True
        else:
            independent = False

        result.end_timing()
        result.index = self.ci_test_counter + 1
        result.set_independent(independent, self.significance)
        result.set_variables(VarX, VarY, VarZ)
        result.set_statistic('G', G, dict())
        result.set_distribution('chi2', p, {'DoF': DoF})

        result.extra_info = ' DoF {}'.format(DoF)

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
        DoF = 0
        if method == 'structural':
            DoF = self.calculate_degrees_of_freedom__structural(X, Y, Z)
        elif method == 'rowcol':
            DoF = self.calculate_degrees_of_freedom__rowcol(PrXYcZ, PrXcZ, PrYcZ, PrZ)
        elif method == 'structural_minus_zerocells':
            DoF = self.calculate_degrees_of_freedom__structural_minus_zerocells(PrXYcZ, PrXcZ, PrYcZ, PrZ, X, Y)
        else:
            raise NotImplementedError

        return DoF


    def calculate_degrees_of_freedom__structural(self, X, Y, Z):
        X_val = len(self.column_values[X])
        Y_val = len(self.column_values[Y])

        Z_val = 1
        for z in Z:
            Z_val *= len(self.column_values[z])
        DoF = (X_val - 1) * (Y_val - 1) * Z_val
        return DoF


    def calculate_degrees_of_freedom__rowcol(self, PrXYcZ, PrXcZ, PrYcZ, PrZ):
        DoF = 0
        for (z, pz) in PrZ.items():
            PrX = PrXcZ.given(z)
            PrY = PrYcZ.given(z)

            expected_dof_xycz = (len(PrX) - 1) * (len(PrY) - 1)
            DoF += expected_dof_xycz

        return DoF


    def calculate_degrees_of_freedom__structural_minus_zerocells(self, PrXYcZ, PrXcZ, PrYcZ, PrZ, X, Y):
        DoF = 0
        X_val = len(self.column_values[X])
        Y_val = len(self.column_values[Y])
        for (z, pz) in PrZ.items():
            PrXY = PrXYcZ.given(z)

            expected_dof_xycz = (X_val - 1) * (Y_val - 1)
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


    def create_flat_variable_set(self, *variables):
        variable_set = set()
        for variable in variables:
            if isinstance(variable, int):
                # `variable` is a single variable ID, not a set or list of IDs
                variable_set.add(variable)
            else:
                # `variable` is a set or list of IDs
                variable_set.update(variable)

        return frozenset(variable_set)


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
