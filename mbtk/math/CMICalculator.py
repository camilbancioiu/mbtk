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


    def compute(self, X, Y, Z):
        bn = self.source_bn
        self.DoF_calculator.reset()

        if len(Z) == 0:
            PrXY = bn.
