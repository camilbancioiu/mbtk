import math
import numpy

from mbff.math.Variable import Variable, Omega, JointVariables, IndexVariable, validate_variable_instances_lengths
from mbff.math.Exceptions import VariableInstancesOfUnequalCount

import mbff.math.infotheory as infotheory

from scipy.stats import chi2


def G_test_conditionally_independent(significance, X, Y, Z=None):
    G = G_value__unoptimized(X, Y, Z)
    DF = calculate_degrees_of_freedom(X, Y)

    p = chi2.cdf(G, DF)
    if p < significance:
        return True
    else:
        return False



def G_value__unoptimized(X, Y, Z=None):
    # If Z is none, use the Universe as the conditioning variable.
    if Z is None:
        # Raise an exception if one of the Variables has a different number of
        # instances than the rest, except Z.
        validate_variable_instances_lengths([X, Y])
        Z = Omega(len(X.instances))
    else:
        # Raise an exception if one of the Variables has a different number of
        # instances than the rest.
        validate_variable_instances_lengths([X, Y, Z])

    N = len(X.instances)

    (PrXYcZ, PrXcZ, PrYcZ, PrZ) = infotheory.calculate_pmf_for_cmi(X, Y, Z)
    cMI = infotheory.conditional_mutual_information(PrXYcZ, PrXcZ, PrYcZ, PrZ, base='e')
    return 2 * N * cMI



def calculate_degrees_of_freedom(X, Y):
    return (len(X.values) - 1) * (len(Y.values) - 1)
