class Variable:

    def __init__(self, name):
        self.name = name
        self.values = []
        self.value_count = len(self.values)
        self.properties = {}
        self.probdist = None


    def __str__(self):
        return 'Variable "{}" with values {}'.format(self.name, self.values)



class ProbabilityDistribution:

    def __init__(self, variable_name):
        self.variable_name = variable_name
        self.variable = None
        self.probabilities = {}
        self.conditioning_set = None
        self.properties = {}


    def __str__(self):
        if self.conditioning_set is None:
            return "ProbabilityMassDistribution for variable {}, unconditional".format(self.variable_name)
        else:
            return "ProbabilityMassDistribution for variable {}, conditioned on {}".format(self.variable_name, self.conditioning_set)



class BayesianNetwork:

    def __init__(self, name):
        self.name = name
        self.variables = {}
        self.properties = {}
