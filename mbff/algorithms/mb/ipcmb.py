import itertools


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
    print("U =", U)

    # Special cache managed by IPCMB. This has nothing to do with AD-trees, dcMI or
    # JMI tables.
    SepSetCache = dict()

    # This is the test of conditional independence. It is a function that must
    # be returned by the CI test builder provided as a parameter (the test
    # builder is itself a function). The test of conditional independence is a
    # function that receives three arguments: the variable indices X and Y and
    # a set of indices Z. The test must return True if X and Y are
    # conditionally independent given Z and False otherwise. In case the test
    # cannot be performed, the exception CannotPerformCITestException should be
    # thrown, which is handled by the algorithm as it sees fit.
    conditionally_independent = parameters['ci_test_builder'](datasetmatrix, parameters)

    # The RecognizePC function of the IPC-MB algorithm. It returns the parents
    # and children of the given target variable, found among the list
    # AdjacentNodes, received as argument. See the IPC-MB article for details.
    def RecognizePC(T, AdjacentNodes):
        print("AdjacentNodes", AdjacentNodes)
        NonPC = set()
        CutSetSize = 0
        while True:
            for X in AdjacentNodes:
                for Z in itertools.combinations(AdjacentNodes - {X}, CutSetSize):
                    Z = set(Z)
                    if conditionally_independent(X, T, Z):
                        NonPC = NonPC | {X}
                        SepSetCache[(T, X)] = Z
                        break
            AdjacentNodes = AdjacentNodes - NonPC
            NonPC = set()
            CutSetSize += 1
            if len(AdjacentNodes) <= CutSetSize:
                break
        return AdjacentNodes

    # The main function of the IPC-MB algorithm. See the IPC-MB article for
    # details.
    def IPCMB(T):
        CandidatePC_T = RecognizePC(T, U - {T})
        print('CandidatePC_T =', CandidatePC_T)

        PC = set()
        CandidateSpouses = dict()
        for X in CandidatePC_T:
            CandidatePC_X = RecognizePC(X, U - {X})
            print('CandidatePC_X =', CandidatePC_X)
            if T in CandidatePC_X:
                PC = PC | {X}
                CandidateSpouses[(T, X)] = CandidatePC_X - {T}  # Note: added " - {T}" even if not in the article

        MB = PC

        for X in PC:
            for Y in CandidateSpouses[(T, X)]:
                if Y not in MB:
                    if not conditionally_independent(T, Y, SepSetCache[(T, Y)] | {X}):
                        MB = MB + {Y}

        return MB
    
    selected_features = list(IPCMB(target))
    print(selected_features)

    return selected_features

