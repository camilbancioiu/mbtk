import math
import numpy

from mbff.math.Variable import Variable, Omega, JointVariables, IndexVariable, validate_variable_instances_lengths
from mbff.math.PMF import PMF, CPMF
from mbff.math.Exceptions import VariableInstancesOfUnequalCount


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

    (PrXYcZ, PrXcZ, PrYcZ, PrZ) = calculate_pmf_for_cmi(X, Y, Z)
    cMI = conditional_mutual_information(PrXYcZ, PrXcZ, PrYcZ, PrZ, base='e')
    return 2 * N * cMI



def calculate_degrees_of_freedom(X, Y):
    return (len(X.values) - 1) * (len(Y.values) - 1)



def conditional_mutual_information(PrXYcZ, PrXcZ, PrYcZ, PrZ, base=2):
    logarithm = create_logarithm_function(base)
    cMI = 0.0
    for (z, pz) in PrZ.items():
        for (x, pxcz) in PrXcZ.given(z).items():
            for (y, pycz) in PrYcZ.given(z).items():
                pxycz = PrXYcZ.given(z).p(x, y)
                if pxycz == 0 or pxcz == 0 or pycz == 0:
                    continue
                else:
                    cMI += pz * pxycz * logarithm(pxycz / (pxcz * pycz))
    return cMI



def calculate_pmf_for_cmi(X, Y, Z):
    PrXYcZ = CPMF(JointVariables(X, Y), Z)
    PrXcZ = CPMF(X, Z)
    PrYcZ = CPMF(Y, Z)
    PrZ = PMF(Z)

    return (PrXYcZ, PrXcZ, PrYcZ, PrZ)



def create_logarithm_function(base):
    if base == 'e':
        return math.log
    elif base == 2:
        return math.log2
    elif base == 10:
        return math.log10
    else:
        return lambda x: math.log(x, base)

