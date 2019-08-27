import pickle

from mbff.math.CITestResult import CITestResult

import mbff.math.infotheory as infotheory
from mbff.math.PMF import PMF, CPMF
from mbff.math.Variable import JointVariables
from mbff.math.Exceptions import InsufficientSamplesForCITest

from scipy.stats import chi2
import gc


class G_test:

    def __init__(self, datasetmatrix, parameters):
        self.parameters = parameters
        self.debug = self.parameters.get('ci_test_debug', 0)
        self.ci_test_name = '.'.join([self.__module__, self.__class__.__name__])
        self.parameters['ci_test_name'] = self.ci_test_name

        self.datasetmatrix = None
        self.matrix = None
        self.column_values = None
        self.N = None
        if datasetmatrix is not None:
            self.datasetmatrix = datasetmatrix
            self.matrix = self.datasetmatrix.X
            self.column_values = self.datasetmatrix.get_values_per_column('X')
            self.N = self.matrix.get_shape()[0]
        self.omega = self.parameters.get('omega', None)
        self.source_bn = self.parameters.get('source_bayesian_network', None)

        self.significance = self.parameters.get('ci_test_significance', 0)
        DoF_calculator_class = self.parameters['ci_test_dof_calculator_class']
        self.DoF_calculator = DoF_calculator_class(self)

        self.ci_test_counter = 0
        self.ci_test_results = []
        self.gc_collect_rate = self.parameters.get('ci_test_gc_collect_rate', 0)


    def conditionally_independent(self, X, Y, Z):
        self.DoF_calculator.reset()
        result = self.G_test_conditionally_independent(X, Y, Z)

        if self.source_bn is not None:
            result.computed_d_separation = self.source_bn.d_separated(X, Z, Y)

        self.ci_test_results.append(result)
        self.print_ci_test_result(result)

        self.ci_test_counter += 1
        if self.gc_collect_rate != 0:
            if self.ci_test_counter % self.gc_collect_rate == 0:
                gc.collect()

        if result.insufficient_samples:
            raise InsufficientSamplesForCITest(result)

        return result.independent


    def G_test_conditionally_independent(self, X, Y, Z):
        (VarX, VarY, VarZ) = self.load_variables(X, Y, Z)

        result = CITestResult()
        result.start_timing()

        (PrXYcZ, PrXcZ, PrYcZ, PrZ) = self.calculate_cpmfs(VarX, VarY, VarZ)

        self.DoF_calculator.set_context_variables(X, Y, Z)
        if self.DoF_calculator.requires_pmfs:
            pmfs = self.calculate_pmfs(VarX, VarY, VarZ)
            self.DoF_calculator.set_context_pmfs(*pmfs)
        if self.DoF_calculator.requires_cpmfs:
            self.DoF_calculator.set_context_cpmfs(PrXYcZ, PrXcZ, PrYcZ, PrZ)
        DoF = self.DoF_calculator.calculate_DoF(X, Y, Z)

        if not self.sufficient_samples(DoF):
            result.end_timing()
            result.index = self.ci_test_counter + 1
            result.set_insufficient_samples()
            result.set_variables(VarX, VarY, VarZ)
            result.extra_info = ' DoF {}'.format(DoF)
            return result

        G = self.G_value(PrXYcZ, PrXcZ, PrYcZ, PrZ)
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


    def sufficient_samples(self, DoF):
        custom_criterion = self.parameters.get('ci_test_sufficient_samples_criterion', None)
        if custom_criterion is None:
            return 5 * DoF < self.N
        else:
            return custom_criterion(self, DoF)


    def load_variables(self, X, Y, Z):
        VarX = self.datasetmatrix.get_variable('X', X)
        VarY = self.datasetmatrix.get_variable('X', Y)
        if len(Z) == 0:
            VarZ = self.omega
        else:
            VarZ = self.datasetmatrix.get_variables('X', Z)

        return (VarX, VarY, VarZ)


    def calculate_cpmfs(self, VarX, VarY, VarZ):
        PrXYcZ = CPMF(JointVariables(VarX, VarY), VarZ)
        PrXcZ = CPMF(VarX, VarZ)
        PrYcZ = CPMF(VarY, VarZ)
        PrZ = PMF(VarZ)

        return (PrXYcZ, PrXcZ, PrYcZ, PrZ)


    def calculate_pmfs(self, VarX, VarY, VarZ):
        PrXYZ = PMF(JointVariables(VarX, VarY, VarZ))
        PrXZ = PMF(JointVariables(VarX, VarZ))
        PrYZ = PMF(JointVariables(VarY, VarZ))
        PrZ = PMF(VarZ)

        return (PrXYZ, PrXZ, PrYZ, PrZ)


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
