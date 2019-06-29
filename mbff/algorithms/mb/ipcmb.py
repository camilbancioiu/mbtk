import itertools

from pprint import pprint


def algorithm_IPCMB(datasetmatrix, parameters):
    """
    Implementation of the IPC-MB algorithm.

    Note that this algorithm will only operate on datasetmatrix.X and will
    ignore datasetmatrix.Y.
    """

    # The 'target' is the index of a column in datasetmatrix.X.
    target = parameters['target']

    # The set of all variables; it is provided as
    # parameters['all_variables'], or alternatively it is read directly from
    # the provided datasetmatrix.
    U = None
    try:
        U = set(parameters['all_variables'])
    except KeyError:
        U = set(range(datasetmatrix.X.get_shape()[1]))

    # Special cache managed by IPCMB. This has nothing to do with AD-trees, dcMI or
    # JMI tables.
    SepSetCache = SetCache()

    # This is the test of conditional independence. It is a class that must be
    # instantiated. The class must have a method called
    # `conditionally_independent` that receives three arguments: the variable
    # indices X and Y and a set of indices Z. The test must return True if X
    # and Y are conditionally independent given Z and False otherwise. In case
    # the test cannot be performed, the exception CannotPerformCITestException
    # should be thrown, which is handled by the algorithm as it sees fit.
    ci_test = parameters['ci_test_class'](datasetmatrix, parameters)

    debug = parameters.get('debug', False)

    # The RecognizePC function of the IPC-MB algorithm. It returns the parents
    # and children of the given target variable, found among the list
    # AdjacentNodes, received as argument. See the IPC-MB article for details.
    def RecognizePC(T, AdjacentNodes):
        NonPC = set()
        CutSetSize = 0
        if debug: print()
        if debug: print('Begin RecognizePC')
        while True:
            if debug: print()
            if debug: print('CutSetSize {}'.format(CutSetSize))
            if debug: print('Target {}, AdjacentNodes {}'.format(T, AdjacentNodes))
            for X in AdjacentNodes:
                if debug: print('\tIterating over X ∈ AdjacentNodes: {}'.format(X))
                for Z in itertools.combinations(AdjacentNodes - {X}, CutSetSize):
                    Z = set(Z)
                    if debug: print('\t\tIterating over possible conditioning sets Z: {}'.format(Z))
                    if debug: print('\t\tTesting {} ̩⊥ {} | {}: '.format(T, X, Z))
                    if ci_test.conditionally_independent(X, T, Z):
                        if debug: print('\t\t\tTrue')
                        NonPC.add(X)
                        if debug: print('\t\t\tNonPC is currently {}'.format(NonPC))
                        SepSetCache.add(Z, T, X)
                        break
                    else:
                        if debug: print('\t\t\tFalse')
                        if debug: print('\t\t\tNonPC remains {}'.format(NonPC))
                if not SepSetCache.contains(T, X):
                    SepSetCache.add(set(), T, X)
            AdjacentNodes = AdjacentNodes - NonPC
            NonPC = set()
            CutSetSize += 1
            if len(AdjacentNodes) <= CutSetSize:
                break
        if debug: print()
        if debug: print('RecognizePC result: {}'.format(AdjacentNodes))
        return AdjacentNodes

    # The main function of the IPC-MB algorithm. See the IPC-MB article for
    # details.
    def IPCMB(T):
        if debug: print('Begin IPCMB')
        CandidatePC_T = RecognizePC(T, U - {T})

        PC = set()
        CandidateSpouses = SetCache()
        for X in CandidatePC_T:
            CandidatePC_X = RecognizePC(X, U - {X})
            if T in CandidatePC_X:
                PC.add(X)
                new_spouses = CandidatePC_X - {T}
                CandidateSpouses.add(new_spouses, T, X)

        # For development purposes, algorithm_IPCMB() can be used to discover only
        # the parents and children of a node, instead of the entire Markov blanket.
        pc_only = parameters.get('pc_only', False)
        if pc_only:
            if debug: print('Returning only PC: {}'.format(PC))
            return PC
        else:
            if debug: print('\tCurrent PC before adding spouses: {}'.format(PC))

        if debug: print('\tCurrent SepSetCache before adding spouses:')
        if debug: pprint(SepSetCache.cache)

        MB = PC.copy()
        for X in PC:
            if debug: print('\tIterating over PC to find spouses: {}'.format(X))
            for Y in CandidateSpouses.get(T, X):
                if debug: print('\t\tIterating over candidate spouses if {} were a child: {}'.format(X, Y))
                if Y not in MB:
                    if debug: print('\t\t\tTesting if {} ⊥ {} |  ( {} ∪ {{ {} }} ):'.format(T, Y, SepSetCache.get(T, Y), X))
                    separation_set = SepSetCache.get(T, Y)
                    if not ci_test.conditionally_independent(T, Y, separation_set | {X}):
                        MB.add(Y)
                        if debug: print('\t\t\t\tFalse, adding {} to the MB, which becomes {}'.format(Y, MB))
                    else:
                        if debug: print('\t\t\t\tTrue, thus {} is not a spouse of {}'.format(Y, T))

        if debug: print('Returning the entire MB: {}'.format(MB))
        return MB

    selected_features = sorted(list(IPCMB(target)))
    return selected_features



class SetCache:

    def __init__(self):
        self.cache = dict()


    def add(self, cset, *elements):
        self.cache[frozenset(elements)] = cset


    def get(self, *elements):
        return self.cache[frozenset(elements)]


    def contains(self, *elements):
        return frozenset(elements) in self.cache
