import time
import math
import numpy

from mbff.math.Variable import Variable, JointVariables, IndexVariable, validate_variable_instances_lengths
from mbff.math.CITestResult import CITestResult
from mbff.math.Exceptions import VariableInstancesOfUnequalCount
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
        self.build_AD_tree()


    def build_AD_tree(self):
        print("Building the AD-tree...")
        self.AD_tree_build_start_time = time.time()
        self.AD_tree = mbff.structures.ADTree.ADTree(self.matrix, self.column_values)
        self.AD_tree_build_end_time = time.time()
        self.AD_tree_build_duration = self.AD_tree_build_end_time - self.AD_tree_build_start_time

        self.N = self.AD_tree.query_count(dict())

        import gc; gc.collect()

        print("AD-tree built in {:>10.4f}s".format(self.AD_tree_build_duration))


    def conditionally_independent(self, X, Y, Z):
        result = self.G_test_conditionally_independent(X, Y, Z)

        if not self.source_bn is None:
            result.computed_d_separation = self.source_bn.d_separated(X, Z, Y)

        self.ci_test_results.append(result)

        # Garbage collection required to deallocate variable instances.
        import gc; gc.collect()

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
