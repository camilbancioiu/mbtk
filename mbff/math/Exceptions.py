class BayesianNetworkNotFinalizedError(Exception):

    def __init__(self, bn, attempt):
        self.bayesian_network = bn
        self.message = "BayesianNetwork not finalized. " + attempt
        super().__init__(self.message)



class BayesianNetworkFinalizedError(Exception):

    def __init__(self, bn, attempt):
        self.bayesian_network = bn
        self.message = "BayesianNetwork already finalized. " + attempt
        super().__init__(self.message)




