from typing import Union

from mbtk.math.PMF import OmegaPMF, OmegaCPMF, PMF, CPMF
from mbtk.math.Variable import JointVariables
import mbtk.math.infotheory as infotheory


class CMICalculator:

    def __init__(self, datasetmatrix, parameters):
        self.parameters = parameters

        self.datasetmatrix = None
        self.matrix = None
        self.column_values = None
        self.N = None
        if datasetmatrix is not None:
            self.datasetmatrix = datasetmatrix
            self.matrix = self.datasetmatrix.X
            self.column_values = self.datasetmatrix.get_values_per_column('X')
            self.N = self.matrix.get_shape()[0]
        self.omega = self.parameters.get('omega', None)
        self.source_bn = self.parameters.get('source_bayesian_network', None)
        self.pmf_source = self.parameters.get('heuristic_pmf_source', 'datasetmatrix')
        if self.source_bn is None:
            self.pmf_source = 'datasetmatrix'


    def compute(self, X: int, Y: int, Z: Union[set[int], list[int]]) -> float:
        Zl = list(Z)
        assert isinstance(Zl, list)

        PrZ: PMF
        PrXcZ: CPMF
        PrYcZ: CPMF
        PrXYcZ: CPMF

        if self.pmf_source == 'datasetmatrix':
            PrXYcZ, PrXcZ, PrYcZ, PrZ = self.make_pmfs_from_datasetmatrix(X, Y, Zl)
        else:
            PrXYcZ, PrXcZ, PrYcZ, PrZ = self.make_pmfs_from_source_bn(X, Y, Zl)

        cMI = infotheory.conditional_mutual_information(
            PrXYcZ,
            PrXcZ,
            PrYcZ,
            PrZ,
            base='e')

        return cMI


    def make_pmfs_from_source_bn(self, X: int, Y: int, Zl: list[int]) -> tuple[CPMF, CPMF, CPMF, PMF]:
        bn = self.source_bn
        if len(Zl) == 0:
            PrXY = bn.create_partial_joint_pmf((X, Y))
            PrX = PrXY.sum_over(Y)
            PrY = PrXY.sum_over(X)
            PrZ = OmegaPMF()
            PrXYcZ = OmegaCPMF(PrXY)
            PrXcZ = OmegaCPMF(PrX)
            PrYcZ = OmegaCPMF(PrY)
        else:
            PrXYZ = bn.create_partial_joint_pmf(tuple([X, Y] + Zl))
            PrXZ = PrXYZ.sum_over(Y)
            PrYZ = PrXYZ.sum_over(X)
            PrZ = PrXZ.sum_over(X)
            PrXYcZ = PrXYZ.condition_on(PrZ)
            PrXcZ = PrXZ.condition_on(PrZ)
            PrYcZ = PrYZ.condition_on(PrZ)

        return (PrXYcZ, PrXcZ, PrYcZ, PrZ)


    def make_pmfs_from_datasetmatrix(self, X: int, Y: int, Zl: list[int]) -> tuple[CPMF, CPMF, CPMF, PMF]:
        PrZ: PMF
        PrXcZ: CPMF
        PrYcZ: CPMF
        PrXYcZ: CPMF

        (VarX, VarY, VarZ) = self.load_variables(X, Y, Zl)
        if len(Zl) == 0:
            PrXY = PMF(JointVariables(VarX, VarY))
            PrX = PMF(VarX)
            PrY = PMF(VarY)
            PrZ = OmegaPMF()
            PrXYcZ = OmegaCPMF(PrXY)
            PrXcZ = OmegaCPMF(PrX)
            PrYcZ = OmegaCPMF(PrY)

        else:
            PrXYZ = PMF(JointVariables(VarX, VarY, VarZ))
            PrXZ = PMF(JointVariables(VarX, VarZ))
            PrYZ = PMF(JointVariables(VarY, VarZ))
            PrZ = PMF(VarZ)

            PrXcZ = PrXZ.condition_on(PrZ)
            PrYcZ = PrYZ.condition_on(PrZ)
            PrXYcZ = PrXYZ.condition_on(PrZ)

        return (PrXYcZ, PrXcZ, PrYcZ, PrZ)


    def load_variables(self, X, Y, Z):
        VarX = self.datasetmatrix.get_variable('X', X)
        VarY = self.datasetmatrix.get_variable('X', Y)
        if len(Z) == 0:
            VarZ = self.omega
        else:
            VarZ = self.datasetmatrix.get_variables('X', Z)

        return (VarX, VarY, VarZ)
