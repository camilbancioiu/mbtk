import math
from typing import Callable, Union

from mbtk.math.PMF import PMF, CPMF
from mbtk.math.Variable import Variable, JointVariables


def mutual_information(
    PrXY: PMF,
    PrX: PMF,
    PrY: PMF,
    base=2,
) -> float:

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



def conditional_mutual_information(
    PrXYcZ: CPMF,
    PrXcZ: CPMF,
    PrYcZ: CPMF,
    PrZ: PMF,
    base: float = 2,
) -> float:

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



def calculate_pmf_for_mi(X: Variable, Y: Variable) -> tuple[PMF, PMF, PMF]:
    PrXY = PMF(JointVariables(X, Y))
    PrX = PMF(X)
    PrY = PMF(Y)

    return (PrXY, PrX, PrY)



def calculate_pmf_for_cmi(
    X: Variable,
    Y: Variable,
    Z: Union[Variable, JointVariables],
) -> tuple[CPMF, CPMF, CPMF, PMF]:

    PrXYcZ = CPMF(JointVariables(X, Y), Z)
    PrXcZ = CPMF(X, Z)
    PrYcZ = CPMF(Y, Z)
    PrZ = PMF(Z)

    return (PrXYcZ, PrXcZ, PrYcZ, PrZ)



def create_logarithm_function(base: Union[str, float]) -> Callable[[float], float]:
    if base == 'e':
        return math.log
    elif base == 2:
        return math.log2
    elif base == 10:
        return math.log10
    else:
        assert isinstance(base, float)
        return lambda x: math.log(x, float(base))
