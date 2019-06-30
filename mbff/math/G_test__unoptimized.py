import math
import numpy

from mbff.math.Variable import Variable, JointVariables, IndexVariable, validate_variable_instances_lengths
from mbff.math.CITestResult import CITestResult
from mbff.math.Exceptions import VariableInstancesOfUnequalCount

import mbff.math.infotheory as infotheory

from scipy.stats import chi2


class G_test:

    def __init__(self, datasetmatrix, parameters):
        self.parameters = parameters
        self.debug = self.parameters.get('debug', False)
        self.datasetmatrix = datasetmatrix
        self.significance = parameters.get('ci_test_significance', 0)
        self.omega = parameters.get('omega', None)
        self.source_bn = parameters.get('source_bayesian_network', None)
        self.ci_test_results = []


    def conditionally_independent(self, X, Y, Z):
        # Load the actual variable instances (samples) from the
        # datasetmatrix.
        (VarX, VarY, VarZ) = self.load_variables(X, Y, Z)
        result = self.G_test_conditionally_independent(VarX, VarY, VarZ)

        if not self.source_bn is None:
            result.computed_d_separation = self.source_bn.d_separated(X, Z, Y)

        self.ci_test_results.append(result)

        # Garbage collection required to deallocate variable instances.
        import gc; gc.collect()

        return result.independent


    def G_test_conditionally_independent(self, VarX, VarY, VarZ):
        result = CITestResult()
        result.start_timing()

        G = self.G_value(VarX, VarY, VarZ)
        DF = self.calculate_degrees_of_freedom(VarX, VarY)

        p = chi2.cdf(G, DF)
        independent = None
        if p < self.significance:
            independent = True
        else:
            independent = False

        result.end_timing()
        result.set_independent(independent, self.significance)
        result.set_variables(VarX, VarY, VarZ)
        result.set_statistic('G', G, dict())
        result.set_distribution('chi2', p, {'DoF': DF})

        return result


    def G_value(self, VarX, VarY, VarZ):
        validate_variable_instances_lengths([VarX, VarY, VarZ])

        N = len(VarX.instances)

        (PrXYcZ, PrXcZ, PrYcZ, PrZ) = infotheory.calculate_pmf_for_cmi(VarX, VarY, VarZ)
        cMI = infotheory.conditional_mutual_information(PrXYcZ, PrXcZ, PrYcZ, PrZ, base='e')
        return 2 * N * cMI


    def load_variables(self, X, Y, Z):
        VarX = self.datasetmatrix.get_variable('X', X)
        VarY = self.datasetmatrix.get_variable('X', Y)
        if len(Z) == 0:
            VarZ = self.omega
        else:
            VarZ = self.datasetmatrix.get_variables('X', Z)

        VarX.load_instances()
        VarY.load_instances()
        VarZ.load_instances()

        return (VarX, VarY, VarZ)


    def calculate_degrees_of_freedom(self, VarX, VarY):
        return (len(VarX.values) - 1) * (len(VarY.values) - 1)
