import math
import numpy

from mbff.math.Variable import Variable, JointVariables, IndexVariable, validate_variable_instances_lengths
from mbff.math.CITestResult import CITestResult
from mbff.math.Exceptions import VariableInstancesOfUnequalCount

import mbff.math.infotheory as infotheory

from scipy.stats import chi2


def G_test_conditionally_independent(significance, X, Y, Z):
    G = G_value__unoptimized(X, Y, Z)
    DF = calculate_degrees_of_freedom(X, Y)

    p = chi2.cdf(G, DF)
    independent = None
    if p < significance:
        independent = True
    else:
        independent = False

    result = CITestResult()
    result.set_independent(independent, significance)
    result.set_variables(X, Y, Z)
    result.set_statistic('G', G, dict())
    result.set_distribution('chi2', p, {'DoF': DF})

    return result



def G_value__unoptimized(X, Y, Z):
    validate_variable_instances_lengths([X, Y, Z])

    N = len(X.instances)

    (PrXYcZ, PrXcZ, PrYcZ, PrZ) = infotheory.calculate_pmf_for_cmi(X, Y, Z)
    cMI = infotheory.conditional_mutual_information(PrXYcZ, PrXcZ, PrYcZ, PrZ, base='e')
    return 2 * N * cMI



def calculate_degrees_of_freedom(X, Y):
    return (len(X.values) - 1) * (len(Y.values) - 1)
