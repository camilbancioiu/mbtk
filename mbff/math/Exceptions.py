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



class VariableInstancesOfUnequalCount(Exception):

    def __init__(self, variables):
        self.variables = variables
        self.variableIDs = [var.ID for var in self.variables]
        self.variableEnum = ", ".join(["ID {} (named {}, length {})".format(var.ID, var.name, len(var.instances())) for var in variables])
        self.message = "Variables {} have unequal numbers of instances".format(self.variableEnum)


    def __str__(self):
        return self.message
