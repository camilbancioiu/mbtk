class InsufficientSamplesForCITest(Exception):

    def __init__(self, result):
        self.message = "Insufficient samples for CI test"
        self.result = result


    def __str__(self):
        return self.message + '\n' + self.result



class VariableInstancesOfUnequalCount(Exception):

    def __init__(self, variables):
        self.variables = variables
        self.variableIDs = [var.ID for var in self.variables]
        self.variableEnum = ", ".join(["ID {} (named {}, length {})".format(var.ID, var.name, len(var.instances())) for var in variables])
        self.message = "Variables {} have unequal numbers of instances".format(self.variableEnum)


    def __str__(self):
        return self.message
