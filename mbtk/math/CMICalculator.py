from typing import Union

from mbtk.math.PMF import OmegaPMF, OmegaCPMF
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
        self.source_bn = self.parameters.get('source_bayesian_network', None)

        # TODO implement datasetmatrix access as well;
        # currently, the CMICalculator only supports
        # extracting probability distributions from a source
        # Bayesian network.
        if self.source_bn is None:
            raise NotImplementedError


    def compute(self, X: int, Y: int, Z: Union[set[int], list[int]]) -> float:
        Zs = list(Z)
        assert isinstance(Zs, list)

        bn = self.source_bn
        print(f'computing I({X}; {Y} | {Z})')

        if len(Z) == 0:
            PrXY = bn.create_partial_joint_pmf((X, Y))
            PrX = PrXY.sum_over(Y)
            PrY = PrXY.sum_over(X)
            PrZ = OmegaPMF()
            PrXYcZ = OmegaCPMF(PrXY)
            PrXcZ = OmegaCPMF(PrX)
            PrYcZ = OmegaCPMF(PrY)
        else:
            PrXYZ = bn.create_partial_joint_pmf(tuple([X, Y] + Zs))
            PrXZ = PrXYZ.sum_over(Y)
            PrYZ = PrXYZ.sum_over(X)
            PrZ = PrXZ.sum_over(X)
            PrXYcZ = PrXYZ.condition_on(PrZ)
            PrXcZ = PrXZ.condition_on(PrZ)
            PrYcZ = PrYZ.condition_on(PrZ)

        cMI = infotheory.conditional_mutual_information(
            PrXYcZ,
            PrXcZ,
            PrYcZ,
            PrZ,
            base='e')

        print(f'I({X}; {Y} | {Z}) = {cMI}')

        return cMI
