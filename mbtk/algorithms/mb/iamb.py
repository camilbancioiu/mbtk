import functools

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

        # self.CI is an alias only
        self.CI = self.CITest.conditionally_independent

        # self.dep_heuristic is the correlation heuristic. It must have a
        # method called `compute` that receives three arguments: the variable
        # indices X and Y and a set of indices Z. The method must return a
        # rational number.
        correlation_heuristic_class = self.parameters['correlation_heuristic_class']
        self.dep_heuristic = correlation_heuristic_class(self.datasetmatrix, self.parameters)


    def select_features(self):
        """Alias of self.discover_mb()"""
        return self.discover_mb()


    def discover_mb(self):
        """
        Run IAMB to discover the Markov boundary of the variable defined
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

        CMB = self.compute_candidate_Markov_boundary()
        MB = self.remove_false_positives_from_Markov_boundary(CMB)

        return MB


    def compute_candidate_Markov_boundary(self):
        possible_candidates = self.U - {self.target}

        CMB = set()

        while True:
            changed_CMB = False
            if self.debug >= 1: print('possible candidates', possible_candidates)
            candidate = self.highest_correlated_candidate(possible_candidates, CMB)
            if self.debug >= 1: print('candidate', candidate)
            if not self.CI(self.target, candidate, CMB):
                CMB.add(candidate)
                possible_candidates.remove(candidate)
                changed_CMB = True
                if self.debug >= 1: print('candidate added', candidate)
            if not changed_CMB:
                if self.debug >= 1: print('breaking')
                break

        return CMB


    def remove_false_positives_from_Markov_boundary(self, CMB):
        false_positives = set()
        for candidate in CMB:
            MB = CMB - false_positives - {candidate}
            if self.debug >= 1: print('current MB', MB)
            if self.debug >= 1: print('candidate', candidate)
            if self.CI(self.target, candidate, MB):
                if self.debug >= 1: print('candidate false positive', candidate)
                false_positives.add(candidate)

        MB = CMB - false_positives
        return MB


    def highest_correlated_candidate(self, possible_candidates, CMB):
        dep = functools.partial(self.dep, CMB)
        return max(possible_candidates, key=dep)

    def dep(self, conditioning_set, candidate):
        return self.dep_heuristic.compute(self.target, candidate, conditioning_set)
