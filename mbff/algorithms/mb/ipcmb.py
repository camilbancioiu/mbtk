import itertools


# All the features.
U = set()

# Special cache managed by IPCMB. This has nothing to do with AD-trees, dcMI or
# JMI tables.
SepSetCache = dict()



def conditionally_independent(X, Y, Z):
    pass



def RecognizePC(T, AdjacentNodes):
    NonPC = set()
    CutSetSize = 0
    while True:
        for X in AdjacentNodes:
            for Z in itertools.combinations(AdjacentNodes - set(X), CutSetSize):
                if conditionally_independent(X, T, Z):
                    NonPC = NonPC + set(X)
                    SepSetCache[(T, X)] = Z
                    break
        AdjacentNodes = AdjacentNodes - NonPC
        NonPC = set()
        CutSetSize += 1
        if len(AdjacentNodes) <= CutSetSize:
            break
    return AdjacentNodes



def IPCMB(T):
    CandidatePC_T = RecognizePC(T, U - set(T))

    PC = set()
    CandidateSpouses = dict()
    for X in CandidatePC_T:
        CandidatePC_X = RecognizePC(X, U - set(X))
        if T in CandidatePC_X:
            PC = PC + set(X)
            CandidateSpouses[(T, X)] = CandidatePC_X


    MB = PC

    for X in PC:
        for Y in CandidateSpouses[(T, X)]:
            if Y not in MB:
                if not conditionally_independent(T, Y, SepSetCache[(T, Y)] + set(X)):
                    MB = MB + set(Y)

    return MB




