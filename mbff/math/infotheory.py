import math

from mbff.math.PMF import PMF, CPMF
from mbff.math.Variable import JointVariables


def mutual_information(PrXY, PrX, PrY, base=2):
    logarithm = create_logarithm_function(base)
    MI = 0.0
    for (x, px) in PrX.items():
        for (y, py) in PrY.items():
            pxy = PrXY.p(x, y)
            if pxy == 0 or px == 0 or py == 0:
                continue
            else:
                pMI = pxy * logarithm(pxy / (px * py))
                MI += pMI
    return MI



def calculate_pmf_for_mi(X, Y):
    PrXY = PMF(JointVariables(X, Y))
    PrX = PMF(X)
    PrY = PMF(Y)

    return (PrXY, PrX, PrY)



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
                    pcMI = pz * pxycz * logarithm(pxycz / (pxcz * pycz))
                    cMI += pcMI
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
