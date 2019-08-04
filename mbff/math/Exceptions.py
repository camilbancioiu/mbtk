class InsufficientSamplesForCITest(Exception):

    def __init__(self, ci_test_result):
        self.message = "Insufficient samples to compute the test of conditional independence"
        self.ci_test_result = ci_test_result


    def __str__(self):
        return self.message



class VariableInstancesOfUnequalCount(Exception):

    def __init__(self, variables):
        self.variables = variables
        self.variableIDs = [var.ID for var in self.variables]
        self.variableEnum = ", ".join(["ID {} (named {}, length {})".format(var.ID, var.name, len(var.instances())) for var in variables])
        self.message = "Variables {} have unequal numbers of instances".format(self.variableEnum)


    def __str__(self):
        return self.message
