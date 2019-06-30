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
        self.N = self.matrix.get_shape()[0]
        self.column_values = self.datasetmatrix.get_values_per_column('X')

        self.JMI_cache = dict()


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
        HXYZ = self.get_joint_entropy_term(X, Y, Z)
        HXZ = self.get_joint_entropy_term(X, Z)
        HYZ = self.get_joint_entropy_term(Y, Z)
        HZ = self.get_joint_entropy_term(Z)
        cMI = HYZ + HXZ - HXYZ - HZ
        return 2 * self.N * cMI


    def get_joint_entropy_term(self, *variables):
        variable_set = self.create_flat_variable_set(*variables)
        jmi_cache_key = frozenset(variable_set)

        try:
            H = self.JMI_cache[jmi_cache_key]
            if self.debug: print('JMI cache hit: found H={:8.6f} for {}'.format(H, jmi_cache_key))
        except KeyError:
            joint_variables = self.datasetmatrix.get_variables('X', variable_set)
            joint_variables.load_instances()
            pmf = PMF(joint_variables)
            H = - pmf.expected_value(lambda v, p: math.log(p))
            self.JMI_cache[jmi_cache_key] = H
            if self.debug: print('JMI cache miss and update: store H={:8.6f} for {}'.format(H, jmi_cache_key))

        return H


    def create_flat_variable_set(self, *variables):
        variable_set = set()
        for variable in variables:
            if isinstance(variable, int):
                # `variable` is a single variable ID, not a set or list of IDs
                variable_set.add(variable)
            else:
                # `variable` is a set or list of IDs
                variable_set.update(variable)

        return variable_set


    def calculate_degrees_of_freedom(self, X, Y):
        X_val = len(self.column_values[X])
        Y_val = len(self.column_values[Y])
        return (X_val - 1) * (Y_val - 1)
