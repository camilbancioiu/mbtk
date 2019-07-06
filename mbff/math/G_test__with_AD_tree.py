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
        self.column_values = self.datasetmatrix.get_values_per_column('X')

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

        adtree_save_path = self.parameters.get('ci_test_ad_tree_path__save', None)
        if adtree_save_path is not None:
            with adtree_save_path.open('wb') as f:
                pickle.dump(self.AD_tree, f)


    def build_AD_tree(self):
        if self.debug: print("Building the AD-tree...")
        leaf_list_threshold = self.parameters['ci_test_ad_tree_leaf_list_threshold']
        self.AD_tree_build_start_time = time.time()
        self.AD_tree = mbff.structures.ADTree.ADTree(self.matrix, self.column_values, leaf_list_threshold, (self.debug, self.debug))
        self.AD_tree_build_end_time = time.time()
        self.AD_tree_build_duration = self.AD_tree_build_end_time - self.AD_tree_build_start_time
        if self.debug: print("AD-tree built in {:>10.4f}s".format(self.AD_tree_build_duration))


    def load_AD_tree(self, adtree_load_path):
        if self.debug: print("Loading the AD-tree from {} ...".format(adtree_load_path))
        with adtree_load_path.open('rb') as f:
            self.AD_tree = pickle.load(f)

        self.AD_tree.debug = self.debug
        self.AD_tree.debug_to_stdout = self.debug
        if self.AD_tree.debug:
            self.AD_tree.debug_prepare__querying()
        if self.debug: print('AD-tree loaded.')


    def conditionally_independent(self, X, Y, Z):
        result = self.G_test_conditionally_independent(X, Y, Z)

        if self.source_bn is not None:
            result.computed_d_separation = self.source_bn.d_separated(X, Z, Y)

        self.ci_test_results.append(result)

        if self.debug: print(result)
        if self.debug: print()

        # Garbage collection required to deallocate variable instances.
        gc.collect()

        return result.independent


    def G_test_conditionally_independent(self, X, Y, Z):
        result = CITestResult()
        result.start_timing()

        G = self.G_value(X, Y, Z)
        DF = self.calculate_degrees_of_freedom(X, Y)

        p = chi2.cdf(G, DF)
        independent = None
        if p < self.significance:
            independent = True
        else:
            independent = False

        result.end_timing()
        result.set_independent(independent, self.significance)
        result.set_variables(X, Y, Z)
        result.set_statistic('G', G, dict())
        result.set_distribution('chi2', p, {'DoF': DF})

        if self.AD_tree.debug:
            result.extra_info = (
                '\nAD-Tree:'
                ' CPMFs {n_cpmf},'
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
            PrXY = self.AD_tree.make_cpmf([X, Y], list())
            PrX = self.AD_tree.make_cpmf([X], list())
            PrY = self.AD_tree.make_cpmf([Y], list())
            PrXYcZ = self.make_cpmf_from_pmf(PrXY)
            PrXcZ = self.make_cpmf_from_pmf(PrX)
            PrYcZ = self.make_cpmf_from_pmf(PrY)
            PrZ = self.make_omega_pmf()
        else:
            PrXYcZ = self.AD_tree.make_cpmf([X, Y], list(Z))
            PrXcZ = self.AD_tree.make_cpmf([X], list(Z))
            PrYcZ = self.AD_tree.make_cpmf([Y], list(Z))
            PrZ = self.AD_tree.make_cpmf(list(Z), list())
        return (PrXYcZ, PrXcZ, PrYcZ, PrZ)


    def make_cpmf_from_pmf(self, pmf):
        cpmf = CPMF(None, None)
        cpmf.conditional_probabilities[1] = pmf
        return cpmf


    def make_omega_pmf(self):
        pmf = PMF(None)
        pmf.probabilities[1] = 1.0
        return pmf


    def calculate_degrees_of_freedom(self, X, Y):
        X_val = len(self.column_values[X])
        Y_val = len(self.column_values[Y])
        return (X_val - 1) * (Y_val - 1)
