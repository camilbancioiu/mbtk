class Variable:

    def __init__(self, name):
        self.name = name
        self.values = []
        self.value_count = len(self.values)
        self.properties = {}
        self.pmd = None


    def __str__(self):
        return 'Variable "{}" with values {}'.format(self.name, self.values)



class ProbabilityMassDistribution:

    def __init__(self, variable_name):
        self.variable_name = variable_name
        self.probabilities = {}
        self.conditioning_set = []
        self.properties = {}


    def __str__(self):
        if self.conditioning_set is None:
            return "ProbabilityMassDistribution for variable {}, unconditional".format(self.variable_name)
        else:
            return "ProbabilityMassDistribution for variable {}, conditioned on {}".format(self.variable_name, self.conditioning_set)


class BayesianNetworkModel:

    def __init__(self, name):
        self.name = name
        self.variables = {}
        self.properties = {}
