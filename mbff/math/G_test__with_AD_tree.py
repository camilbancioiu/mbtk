import time
import pickle

from mbff.math.CITestResult import CITestResult
from mbff.math.PMF import PMF, CPMF

import mbff.math.infotheory as infotheory
import mbff.math.G_test__unoptimized

import mbff.structures.ADTree

from scipy.stats import chi2


class G_test(mbff.math.G_test__unoptimized.G_test):

    def __init__(self, datasetmatrix, parameters):
        super().__init__(datasetmatrix, parameters)

        self.AD_tree_build_start_time = 0
        self.AD_tree_build_end_time = 0
        self.AD_tree_build_duration = 0.0
        self.AD_tree = None
        self.N = None

        self.prepare_AD_tree()


    def prepare_AD_tree(self):
        preloaded_AD_tree = self.parameters.get('ci_test_ad_tree_preloaded', None)
        if preloaded_AD_tree is not None:
            # The AD-tree might have already been loaded in memory (by the
            # experiment script, by a previous G-test instance or any other
            # way), and we've been passed a reference to it. We simply use that
            # reference from now on.
            self.use_preloaded_AD_tree(preloaded_AD_tree)
        else:
            # There is no preloaded AD tree available, so we try to load
            # it ourselves from the path provided as a parameter; if that
            # fails, start building the entire AD-tree for the given
            # datasetmatrix.
            adtree_load_path = self.parameters.get('ci_test_ad_tree_path__load', None)
            if adtree_load_path is not None and adtree_load_path.exists():
                self.load_AD_tree(adtree_load_path)
            else:
                self.build_AD_tree()
                self.save_AD_tree()

        # By definition, an empty query passed to the AD-tree returns the total
        # number of rows in the dataset.
        self.N = self.AD_tree.query_count(dict())


    def use_preloaded_AD_tree(self, preloaded_AD_tree):
        self.AD_tree = preloaded_AD_tree


    def load_AD_tree(self, adtree_load_path):
        with adtree_load_path.open('rb') as f:
            self.AD_tree = pickle.load(f)


    def build_AD_tree(self):
        ADTreeClass = self.parameters['ci_test_ad_tree_class']
        leaf_list_threshold = self.parameters['ci_test_ad_tree_leaf_list_threshold']
        self.AD_tree_build_start_time = time.time()
        self.AD_tree = ADTreeClass(self.matrix, self.column_values, leaf_list_threshold)
        self.AD_tree_build_end_time = time.time()
        self.AD_tree_build_duration = self.AD_tree_build_end_time - self.AD_tree_build_start_time


    def save_AD_tree(self):
        adtree_save_path = self.parameters.get('ci_test_ad_tree_path__save', None)
        if adtree_save_path is not None:
            with adtree_save_path.open('wb') as f:
                pickle.dump(self.AD_tree, f)


    def G_test_conditionally_independent(self, X, Y, Z):
        result = CITestResult()
        result.start_timing()

        (PrXYcZ, PrXcZ, PrYcZ, PrZ) = self.calculate_pmf_from_AD_tree(X, Y, Z)

        self.DoF_calculator.set_context_variables(X, Y, Z)
        if self.DoF_calculator.requires_cpmfs:
            self.DoF_calculator.set_context_cpmfs(PrXYcZ, PrXcZ, PrYcZ, PrZ)
        DoF = self.DoF_calculator.calculate_DoF(X, Y, Z)

        if not self.sufficient_samples(DoF):
            result.end_timing()
            result.index = self.ci_test_counter + 1
            result.set_insufficient_samples()
            result.set_variables(X, Y, Z)
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
        result.set_variables(X, Y, Z)
        result.set_statistic('G', G, dict())
        result.set_distribution('chi2', p, {'DoF': DoF})

        result.extra_info = ' DoF {}'.format(DoF)

        return result


    def G_value(self, PrXYcZ, PrXcZ, PrYcZ, PrZ):
        cMI = infotheory.conditional_mutual_information(PrXYcZ, PrXcZ, PrYcZ, PrZ, base='e')
        return 2 * self.N * cMI


    def calculate_pmf_from_AD_tree(self, X, Y, Z):
        if len(Z) == 0:
            PrXY = self.AD_tree.make_pmf(sorted([X, Y]))
            if [X, Y] != sorted([X, Y]):
                new_probabilities = dict()
                for key, p in PrXY.items():
                    new_key = tuple(reversed(key))
                    new_probabilities[new_key] = p
                PrXY.probabilities = new_probabilities

            PrX = self.AD_tree.make_pmf([X])
            PrY = self.AD_tree.make_pmf([Y])
            PrXYcZ = self.make_omega_cpmf_from_pmf(PrXY)
            PrXcZ = self.make_omega_cpmf_from_pmf(PrX)
            PrYcZ = self.make_omega_cpmf_from_pmf(PrY)
            PrZ = self.make_omega_pmf()

            if self.DoF_calculator.requires_pmfs:
                self.DoF_calculator.set_context_pmfs(PrXY, PrX, PrY, None)

        else:
            Z = sorted(list(Z))
            PrZ = self.AD_tree.make_pmf(Z)
            (PrXYcZ, PrXYZ) = self.make_cpmf_PrXYcZ(X, Y, Z, PrZ)
            (PrXcZ, PrXZ) = self.make_cpmf_PrXcZ(X, Z, PrZ)
            (PrYcZ, PrYZ) = self.make_cpmf_PrXcZ(Y, Z, PrZ)

            if self.DoF_calculator.requires_pmfs:
                self.DoF_calculator.set_context_pmfs(PrXYZ, PrXZ, PrYZ, PrZ)

        return (PrXYcZ, PrXcZ, PrYcZ, PrZ)


    def make_cpmf_PrXYcZ(self, X, Y, Z, PrZ=None):
        if PrZ is None:
            PrZ = self.AD_tree.make_pmf(list(Z))

        unsorted_variables = [X, Y] + Z
        joint_variables = sorted(unsorted_variables)
        index = {var: joint_variables.index(var) for var in joint_variables}

        PrXYZ = self.AD_tree.make_pmf(joint_variables)

        PrXYcZ = CPMF(None, None)

        for joint_key, joint_p in PrXYZ.items():
            zkey = tuple([joint_key[index[zvar]] for zvar in Z])
            varkey = tuple([joint_key[index[var]] for var in unsorted_variables if var not in Z])
            if len(zkey) == 1:
                zkey = zkey[0]
            try:
                pmf = PrXYcZ.conditional_probabilities[zkey]
            except KeyError:
                pmf = PMF(None)
                PrXYcZ.conditional_probabilities[zkey] = pmf
            try:
                pmf.probabilities[varkey] = joint_p / PrZ.p(zkey)
            except ZeroDivisionError:
                pass

        return (PrXYcZ, PrXYZ)


    def make_cpmf_PrXcZ(self, X, Z, PrZ=None):
        if PrZ is None:
            PrZ = self.AD_tree.make_pmf(list(Z))

        unsorted_variables = [X] + Z
        joint_variables = sorted(unsorted_variables)
        index = {var: joint_variables.index(var) for var in joint_variables}

        PrXZ = self.AD_tree.make_pmf(joint_variables)

        PrXcZ = CPMF(None, None)

        for joint_key, joint_p in PrXZ.items():
            zkey = tuple([joint_key[index[zvar]] for zvar in Z])
            varkey = [joint_key[index[var]] for var in unsorted_variables if var not in Z][0]
            if len(zkey) == 1:
                zkey = zkey[0]
            try:
                pmf = PrXcZ.conditional_probabilities[zkey]
            except KeyError:
                pmf = PMF(None)
                PrXcZ.conditional_probabilities[zkey] = pmf
            try:
                pmf.probabilities[varkey] = joint_p / PrZ.p(zkey)
            except ZeroDivisionError:
                pass

        return (PrXcZ, PrXZ)
