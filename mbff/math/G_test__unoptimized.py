import pickle

from mbff.math.CITestResult import CITestResult

import mbff.math.infotheory as infotheory
from mbff.math.PMF import PMF, CPMF, PMFInfo
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

        self.PMF_info = dict()
        self.prepare_PMF_info()


    def prepare_PMF_info(self):
        pmf_info_load_path = self.parameters.get('ci_test_pmf_info_path__load', None)
        if pmf_info_load_path is not None and pmf_info_load_path.exists():
            if self.debug >= 1: print('Loading the PMF info from {} ...'.format(pmf_info_load_path))
            with pmf_info_load_path.open('rb') as f:
                self.PMF_info = pickle.load(f)
            if self.debug >= 1: print('PMF info loaded.')


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
            if self.parameters.get('ci_test_dof_computation_method', 'structural') == 'cached_joint_pmf_info':
                self.cache_variable_pmf_infos__XY(VarX, VarY, X, Y)
        else:
            VarZ = self.datasetmatrix.get_variables('X', Z)
            if self.parameters.get('ci_test_dof_computation_method', 'structural') == 'cached_joint_pmf_info':
                self.cache_variable_pmf_infos__XYZ(VarX, VarY, VarZ, X, Y, Z)

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
        elif method == 'cached_joint_pmf_info':
            DoF = self.calculate_degrees_of_freedom__cached_joint_pmf_info(X, Y, Z)
        else:
            raise NotImplementedError

        if DoF < 1:
            raise InsufficientSamplesForCITest("Calculated degrees of freedom is less than 1 for {} ⊥ {} | {}: {} due to too many empty cells in the contingency table.".format(X, Y, Z, DoF))

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


    def calculate_degrees_of_freedom__cached_joint_pmf_info(self, X, Y, Z):
        # Expected count of values for X, when none are missing.
        X_xp = len(self.column_values[X])
        # Expected count of values for Y, when none are missing.
        Y_xp = len(self.column_values[Y])

        Z_observed = None
        if len(Z) > 0:
            z = self.create_flat_variable_set(Z)
            infoZ = self.PMF_info[z]
            Z_observed = infoZ.observed_nonzero_cells
        else:
            Z_observed = 1

        xyz = self.create_flat_variable_set(X, Y, Z)
        xz = self.create_flat_variable_set(X, Z)
        yz = self.create_flat_variable_set(Y, Z)

        infoXYZ = self.PMF_info[xyz]
        infoXZ = self.PMF_info[xz]
        infoYZ = self.PMF_info[yz]

        X_0 = Z_observed * X_xp - infoXZ.observed_nonzero_cells
        Y_0 = Z_observed * Y_xp - infoYZ.observed_nonzero_cells
        XY_0 = Z_observed * X_xp * Y_xp - infoXYZ.observed_nonzero_cells - infoXYZ.observed_zero_cells

        sum_of_products_of_observed_XcZ_and_YcZ = (-1) * (XY_0 - (X_0 * Y_xp) - (Y_0 * X_xp))

        term1 = Z_observed * (X_xp * Y_xp - X_xp - Y_xp + 1)
        term2 = (-1) * X_xp * Y_0
        term3 = (-1) * Y_xp * X_0
        term4 = X_0
        term5 = Y_0
        term6 = sum_of_products_of_observed_XcZ_and_YcZ
        DoF = term1 + term2 + term3 + term4 + term5 + term6

        return DoF


    def cache_variable_pmf_infos__XYZ(self, VarX, VarY, VarZ, X, Y, Z):
        PrXYZ = PMF(JointVariables(VarX, VarY, VarZ))
        PrXZ = PMF(JointVariables(VarX, VarZ))
        PrYZ = PMF(JointVariables(VarY, VarZ))
        PrZ = PMF(VarZ)
        self.cache_pmf_infos__XYZ(PrXYZ, PrXZ, PrYZ, PrZ, X, Y, Z)


    def cache_variable_pmf_infos__XY(self, VarX, VarY, X, Y):
        PrXY = PMF(JointVariables(VarX, VarY))
        PrX = PMF(VarX)
        PrY = PMF(VarY)
        self.cache_pmf_infos__XY(PrXY, PrX, PrY, X, Y)


    def cache_pmf_infos__XYZ(self, PrXYZ, PrXZ, PrYZ, PrZ, X, Y, Z):
        if PrXYZ is not None:
            xyz = self.create_flat_variable_set(X, Y, Z)
            self.cache_pmf_info(xyz, PrXYZ)

        if PrXZ is not None:
            xz = self.create_flat_variable_set(X, Z)
            self.cache_pmf_info(xz, PrXZ)

        if PrYZ is not None:
            yz = self.create_flat_variable_set(Y, Z)
            self.cache_pmf_info(yz, PrYZ)

        if PrZ is not None:
            z = self.create_flat_variable_set(Z)
            self.cache_pmf_info(z, PrZ)


    def cache_pmf_infos__XY(self, PrXY, PrX, PrY, X, Y):
        if PrXY is not None:
            xy = self.create_flat_variable_set(X, Y)
            self.cache_pmf_info(xy, PrXY)

        if PrX is not None:
            x = self.create_flat_variable_set(X)
            self.cache_pmf_info(x, PrX)

        if PrY is not None:
            y = self.create_flat_variable_set(Y)
            self.cache_pmf_info(y, PrY)


    def cache_pmf_info(self, key, pmf):
        if key not in self.PMF_info:
            info = PMFInfo()
            info.expected_total_cells = 1
            for variable_index in key:
                info.expected_total_cells *= len(self.column_values[variable_index])
            info.observed_total_cells = len(pmf)
            info.observed_nonzero_cells = len(list(filter(None, pmf.values())))
            info.observed_zero_cells = info.observed_total_cells - info.observed_nonzero_cells
            info.observed_missing_cells = info.expected_total_cells - info.observed_total_cells
            self.PMF_info[key] = info


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

        pmf_info_save_path = self.parameters.get('ci_test_pmf_info_path__save', None)
        if pmf_info_save_path is not None:
            with pmf_info_save_path.open('wb') as f:
                pickle.dump(self.PMF_info, f)
        if self.debug >= 1: print('PMF info has been saved to {}'.format(pmf_info_save_path))


    def print_ci_test_result(self, result):
        if self.debug >= 1:
            if result.accurate():
                if self.parameters.get('ci_test_results__print_accurate', True):
                    print(result)
            if not result.accurate():
                if self.parameters.get('ci_test_results__print_inaccurate', True):
                    print(result)
