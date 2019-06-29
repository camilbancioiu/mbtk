import math
import numpy

from mbff.math.Variable import Variable, JointVariables, IndexVariable, validate_variable_instances_lengths
from mbff.math.CITestResult import CITestResult
from mbff.math.Exceptions import VariableInstancesOfUnequalCount

import mbff.math.infotheory as infotheory

from scipy.stats import chi2


def ci_test_builder(datasetmatrix, parameters):
    significance = parameters['ci_test_significance']
    omega = parameters['omega']
    ci_test_results = parameters.get('ci_test_results', list())
    source_bn = parameters.get('source_bayesian_network', None)

    def conditionally_independent(X, Y, Z):
        # Load the actual variable instances (samples) from the
        # datasetmatrix.
        (VarX, VarY, VarZ) = load_variables(X, Y, Z, datasetmatrix, omega)
        result = G_test_conditionally_independent(significance, VarX, VarY, VarZ)

        if not source_bn is None:
            result.computed_d_separation = source_bn.d_separated(X, Z, Y)

        ci_test_results.append(result)

        # Garbage collection required to deallocate variable instances.
        import gc; gc.collect()

        return result.independent

    return conditionally_independent



def load_variables(X, Y, Z, datasetmatrix, omega):
    VarX = datasetmatrix.get_variable('X', X)
    VarY = datasetmatrix.get_variable('X', Y)
    if len(Z) == 0:
        VarZ = omega
    else:
        VarZ = datasetmatrix.get_variables('X', Z)

    VarX.load_instances()
    VarY.load_instances()
    VarZ.load_instances()

    return (VarX, VarY, VarZ)




def G_test_conditionally_independent(significance, VarX, VarY, VarZ):
    G = G_value(VarX, VarY, VarZ)
    DF = calculate_degrees_of_freedom(VarX, VarY)

    p = chi2.cdf(G, DF)
    independent = None
    if p < significance:
        independent = True
    else:
        independent = False

    result = CITestResult()
    result.set_independent(independent, significance)
    result.set_variables(VarX, VarY, VarZ)
    result.set_statistic('G', G, dict())
    result.set_distribution('chi2', p, {'DoF': DF})

    return result



def G_value(VarX, VarY, VarZ):
    validate_variable_instances_lengths([VarX, VarY, VarZ])

    N = len(VarX.instances)

    (PrXYcZ, PrXcZ, PrYcZ, PrZ) = infotheory.calculate_pmf_for_cmi(VarX, VarY, VarZ)
    cMI = infotheory.conditional_mutual_information(PrXYcZ, PrXcZ, PrYcZ, PrZ, base='e')
    return 2 * N * cMI



def calculate_degrees_of_freedom(VarX, VarY):
    return (len(VarX.values) - 1) * (len(VarY.values) - 1)
