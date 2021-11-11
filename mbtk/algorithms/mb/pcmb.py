import itertools
import functools


class AlgorithmPCMB:
    """
    Implementation of the PCMB algorithm.

    Note that this algorithm will only operate on datasetmatrix.X and will
    ignore datasetmatrix.Y.
    """

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
        self.CITest = self.parameters['ci_test_class'](self.datasetmatrix, self.parameters)

        # self.CI is an alias
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
        Run PCMB to discover the Markov boundary of the variable defined
        in the preconfigured parameters.
        """
        markov_boundary = sorted(list(self.PCMB(self.target)))
        self.CITest.end()
        return markov_boundary


    def PCMB(self, T):
        if self.debug >= 1: print('Begin PCMB')
        self.SepSetCache = dict()
        self.SepSetZ = dict()

        PC = self.GetPC(T)
        print('PC', PC)
        MB = PC
        print('MB', MB)

        for Y in PC:
            for X in self.GetPC(Y):
                if X not in PC:
                    Z = self.SepSetZ[X]
                    if not self.CI(self.target, X, Z):
                        raise Exception("PCMB error")
                    if not self.CI(self.target, X, Z | {Y}):
                        MB.add(X)
                    print('MB', MB)

        return MB


    def GetPC(self, T):
        PC = set()
        PCD = self.GetPCD(T)
        for X in PCD:
            if T in self.GetPCD(X):
                PC.add(X)
        return PC


    def GetPCD(self, T):
        PCD = set()
        CandidatePCD = self.U - {T}

        iteration = 0

        while True:
            print('iteration', iteration)
            for X in CandidatePCD:
                self.SepSetCache[X] = self.strongest_separator(PCD, X)
            false_positives = self.find_false_candidates(CandidatePCD)
            print('CandidatePCD with false positives', CandidatePCD)
            CandidatePCD = CandidatePCD - false_positives
            print('CandidatePCD without false positives', CandidatePCD)

            candidate = self.highest_correlated_candidate_sep(CandidatePCD)
            PCD.add(candidate)
            CandidatePCD.remove(candidate)

            for X in PCD:
                self.SepSetCache[X] = self.strongest_separator(PCD - {X}, X)
            false_positives = self.find_false_candidates(PCD)
            print('PCD with false positives', PCD)
            PCD = PCD - false_positives
            print('PCD without false positives', PCD)

            if len(false_positives) == 0:
                break

            iteration += 1


        print(f'GetPCD({T}) = {PCD}')
        return PCD


    def find_false_candidates(self, PCD):
        false_positives = set()
        for X in PCD:
            Z = self.SepSetCache[X]
            if self.CI(self.target, X, Z):
                false_positives.add(X)
                self.SepSetZ[X] = Z
        return false_positives


    def strongest_separator(self, PCD, candidate):
        subsets = powerset(PCD)
        dep_subset = functools.partial(self.dep, candidate)
        return min(subsets, key=dep_subset)


    def highest_correlated_candidate_sep(self, candidates):
        return min(candidates, key=self.dep_with_separator)


    def dep_with_separator(self, candidate):
        conditioning_set = self.SepSetCache[candidate]
        return self.dep(candidate, conditioning_set)


    def dep(self, candidate, conditioning_set):
        return self.dep_heuristic.compute(
            self.target,
            candidate,
            conditioning_set)


def powerset(iterable):
    s = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(len(s) + 1)
    )
