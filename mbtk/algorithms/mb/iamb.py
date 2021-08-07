class AlgorithmIAMB:

    def __init__(self, datasetmatrix, parameters):
        self.parameters = parameters
        self.debug = self.parameters.get('algorithm_debug', 0)
        self.datasetmatrix = datasetmatrix

        # The 'target' is the index of a column in datasetmatrix.X.
        self.target = parameters['target']

        # The set of all variables; it is provided as
        # parameters['all_variables'], or alternatively it is read directly from
        # the provided datasetmatrix.
        self.U = None
        try:
            self.U = set(self.parameters['all_variables'])
        except KeyError:
            self.U = set(range(self.datasetmatrix.X.get_shape()[1]))

        # self.CITest is the test of conditional independence. It must have a
        # method called `conditionally_independent` that receives three
        # arguments: the variable indices X and Y and a set of indices Z. The
        # test must return True if X and Y are conditionally independent given
        # Z and False otherwise. In case the test cannot be performed, the
        # exception CannotPerformCITestException should be thrown, which is
        # handled by the algorithm as it sees fit.
        ci_test_class = self.parameters['ci_test_class']
        self.CITest = ci_test_class(self.datasetmatrix, self.parameters)

        # self.Dep is the correlation heuristic. It must have a method called
        # `compute` that receives three arguments: the variable indices X and Y
        # and a set of indices Z. The method must return a rational number.
        correlation_heuristic_class = self.parameters['correlation_heuristic_class']
        self.Dep = correlation_heuristic_class(self.datasetmatrix, self.parameters)


    def select_features(self):
        """Alias of self.discover_mb()"""
        return self.discover_mb()


    def discover_mb(self):
        """
        Run IPC-MB to discover the Markov boundary of the variable defined
        in the preconfigured parameters.
        """
        markov_boundary = sorted(list(self.IAMB(self.target)))
        self.CITest.end()
        return markov_boundary


    def IAMB(self, T):
        """
        The main function of the IAMB algorithm. See the IAMB article for
        details.
        """
        if self.debug >= 1: print('Begin IAMB')

        return []
