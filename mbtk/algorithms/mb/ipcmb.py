import itertools
from mbtk.math.Exceptions import InsufficientSamplesForCITest
from mbtk.algorithms.mb.setcache import SetCache

from pprint import pprint


class AlgorithmIPCMB:
    """
    Implementation of the IPC-MB algorithm.

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


    def select_features(self):
        """Alias of self.discover_mb()"""
        return self.discover_mb()


    def discover_mb(self):
        """
        Run IPC-MB to discover the Markov boundary of the variable defined
        in the preconfigured parameters.
        """
        markov_boundary = sorted(list(self.IPCMB(self.target)))
        self.CITest.end()
        return markov_boundary


    def IPCMB(self, T):
        """
        The main function of the IPC-MB algorithm. See the IPC-MB article for
        details.
        """
        if self.debug >= 1: print('Begin IPCMB')
        self.SepSetCache = SetCache()
        CandidatePC_T = self.RecognizePC(T, self.U - {T})

        PC = set()
        CandidateSpouses = SetCache()
        for X in CandidatePC_T:
            CandidatePC_X = self.RecognizePC(X, self.U - {X})
            if T in CandidatePC_X:
                PC.add(X)
                new_spouses = CandidatePC_X - {T}
                CandidateSpouses.add(new_spouses, T, X)

        # For development purposes, AlgorithmIPCMB.IPCMB() can be used to discover only
        # the parents and children of a node, instead of the entire Markov blanket.
        pc_only = self.parameters.get('pc_only', False)
        if pc_only:
            if self.debug >= 1: print('Returning only PC: {}'.format(PC))
            return PC
        else:
            if self.debug >= 2: print('\tCurrent PC before adding spouses: {}'.format(PC))

        if self.debug >= 2: print('\tCurrent SepSetCache before adding spouses:')
        if self.debug >= 2: pprint(self.SepSetCache.cache)

        MB = PC.copy()
        for X in PC:
            if self.debug >= 2: print('\tIterating over PC to find spouses: {}'.format(X))
            for Y in CandidateSpouses.get(T, X):
                if self.debug >= 2: print('\t\tIterating over candidate spouses if {} were a child: {}'.format(X, Y))
                if Y not in MB:
                    if self.debug >= 2: print('\t\t\tTesting if {} ⊥ {} |  ( {} ∪ {{ {} }} ):'.format(T, Y, self.SepSetCache.get(T, Y), X))
                    separation_set = self.SepSetCache.get(T, Y)
                    try:
                        if not self.CITest.conditionally_independent(T, Y, separation_set | {X}):
                            MB.add(Y)
                            if self.debug >= 2: print('\t\t\t\tFalse, adding {} to the MB, which becomes {}'.format(Y, MB))
                        else:
                            if self.debug >= 2: print('\t\t\t\tTrue, thus {} is not a spouse of {}'.format(Y, T))
                    except InsufficientSamplesForCITest:
                        continue

        if self.debug >= 1: print('Returning the entire MB: {}'.format(MB))
        return MB


    def RecognizePC(self, T, AdjacentNodes):
        """
        The RecognizePC function of the IPC-MB algorithm. It returns the parents
        and children of the given target variable, found among the list
        AdjacentNodes, received as argument. See the IPC-MB article for details.
        """
        NonPC = set()
        CutSetSize = 0
        if self.debug >= 2: print()
        if self.debug >= 2: print('Begin RecognizePC')
        while True:
            if self.debug >= 2: print()
            if self.debug >= 2: print('CutSetSize {}'.format(CutSetSize))
            if self.debug >= 2: print('Target {}, AdjacentNodes {}'.format(T, AdjacentNodes))
            for X in AdjacentNodes:
                if self.debug >= 2: print('\tIterating over X ∈ AdjacentNodes: {}'.format(X))
                for Z in itertools.combinations(AdjacentNodes - {X}, CutSetSize):
                    Z = set(Z)
                    if self.debug >= 2: print('\t\tIterating over possible conditioning sets Z: {}'.format(Z))
                    if self.debug >= 2: print('\t\tTesting {} ̩⊥ {} | {}: '.format(T, X, Z))
                    try:
                        if self.CITest.conditionally_independent(X, T, Z):
                            if self.debug >= 2: print('\t\t\tTrue')
                            NonPC.add(X)
                            if self.debug >= 2: print('\t\t\tNonPC is currently {}'.format(NonPC))
                            self.SepSetCache.add(Z, T, X)
                            break
                        else:
                            if self.debug >= 2: print('\t\t\tFalse')
                            if self.debug >= 2: print('\t\t\tNonPC remains {}'.format(NonPC))
                    except InsufficientSamplesForCITest:
                        NonPC.add(X)
                        continue
                if not self.SepSetCache.contains(T, X):
                    self.SepSetCache.add(set(), T, X)
            AdjacentNodes = AdjacentNodes - NonPC
            NonPC = set()
            CutSetSize += 1
            if len(AdjacentNodes) <= CutSetSize:
                break
        if self.debug >= 2: print()
        if self.debug >= 2: print('RecognizePC result: {}'.format(AdjacentNodes))
        return AdjacentNodes
