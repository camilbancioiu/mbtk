import time
import pickle
import gc

from mbff.math.CITestResult import CITestResult
from mbff.math.PMF import PMF, CPMF

import mbff.math.infotheory as infotheory
import mbff.math.G_test__unoptimized

import mbff.structures.ADTree

from scipy.stats import chi2



class G_test(mbff.math.G_test__unoptimized.G_test):

    def __init__(self, datasetmatrix, parameters):
        super().__init__(datasetmatrix, parameters)

        self.matrix = self.datasetmatrix.X

        self.AD_tree_build_start_time = 0
        self.AD_tree_build_end_time = 0
        self.AD_tree_build_duration = 0.0
        self.AD_tree = None
        self.N = None

        self.prepare_AD_tree()


    def prepare_AD_tree(self):
        adtree_load_path = self.parameters.get('ci_test_ad_tree_path__load', None)
        if adtree_load_path is not None and adtree_load_path.exists():
            self.load_AD_tree(adtree_load_path)
        else:
            self.build_AD_tree()

        self.N = self.AD_tree.query_count(dict())


    def build_AD_tree(self):
        if self.debug >= 1: print("Building the AD-tree...")
        leaf_list_threshold = self.parameters['ci_test_ad_tree_leaf_list_threshold']
        self.AD_tree_build_start_time = time.time()
        self.AD_tree = mbff.structures.ADTree.ADTree(self.matrix, self.column_values, leaf_list_threshold, self.debug)
        self.AD_tree_build_end_time = time.time()
        self.AD_tree_build_duration = self.AD_tree_build_end_time - self.AD_tree_build_start_time
        if self.debug >= 1: print("AD-tree built in {:>10.4f}s".format(self.AD_tree_build_duration))

        adtree_save_path = self.parameters.get('ci_test_ad_tree_path__save', None)
        if adtree_save_path is not None:
            with adtree_save_path.open('wb') as f:
                pickle.dump(self.AD_tree, f)
        if self.debug >= 1: print("AD-tree saved to", adtree_save_path)


    def load_AD_tree(self, adtree_load_path):
        if self.debug >= 1: print("Loading the AD-tree from {} ...".format(adtree_load_path))
        with adtree_load_path.open('rb') as f:
            self.AD_tree = pickle.load(f)

        self.AD_tree.debug = self.debug - 1
        if self.AD_tree.debug >= 1:
            self.AD_tree.debug_prepare__querying()
        if self.debug >= 1: print('AD-tree loaded.')


    def conditionally_independent(self, X, Y, Z):
        result = self.G_test_conditionally_independent(X, Y, Z)

        if self.source_bn is not None:
            result.computed_d_separation = self.source_bn.d_separated(X, Z, Y)

        self.ci_test_results.append(result)

        if self.debug >= 1: print(result)
        if self.debug >= 1: print()

        # Garbage collection required to deallocate variable instances.
        gc.collect()

        return result.independent


    def G_test_conditionally_independent(self, X, Y, Z):
        result = CITestResult()
        result.start_timing()

        G = self.G_value(X, Y, Z)
        DF = self.calculate_degrees_of_freedom(X, Y, Z)

        p = chi2.cdf(G, DF)
        independent = None
        if p < self.significance:
            independent = True
        else:
            independent = False

        result.end_timing()
        result.index = len(self.ci_test_results)
        result.set_independent(independent, self.significance)
        result.set_variables(X, Y, Z)
        result.set_statistic('G', G, dict())
        result.set_distribution('chi2', p, {'DoF': DF})

        if self.AD_tree.debug >= 1:
            result.extra_info = (
                '\nAD-Tree:'
                ' total of {n_pmf} contingency tables; currently {n_pmf_ll} contingency tables from leaf-lists;'
                ' queries {n_queries},'
                ' of which leaf-list queries {n_queries_ll}'
            ).format(**self.AD_tree.__dict__)

        return result


    def G_value(self, X, Y, Z):
        (PrXYcZ, PrXcZ, PrYcZ, PrZ) = self.calculate_pmf_from_AD_tree(X, Y, Z)
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
        else:
            Z = sorted(list(Z))
            PrZ = self.AD_tree.make_pmf(Z)
            PrXYcZ = self.make_cpmf_PrXYcZ(X, Y, Z, PrZ)
            PrXcZ = self.make_cpmf_PrXcZ(X, Z, PrZ)
            PrYcZ = self.make_cpmf_PrYcZ(Y, Z, PrZ)
        return (PrXYcZ, PrXcZ, PrYcZ, PrZ)


    def make_omega_cpmf_from_pmf(self, pmf):
        cpmf = CPMF(None, None)
        cpmf.conditional_probabilities[1] = pmf
        return cpmf


    def make_omega_pmf(self):
        pmf = PMF(None)
        pmf.probabilities[1] = 1.0
        return pmf


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

        return PrXYcZ


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

        return PrXcZ


    def make_cpmf_PrYcZ(self, Y, Z, PrZ=None):
        return self.make_cpmf_PrXcZ(Y, Z, PrZ)
