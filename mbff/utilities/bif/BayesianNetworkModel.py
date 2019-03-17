import warnings
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import pomegranate

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


    def to_pomegranate_distribution(self, conditioning_distributions=None):
        if self.conditioning_set is None:
            probability_table = dict(zip(self.variable.values, self.probabilities['<unconditional>']))
            distribution = pomegranate.DiscreteDistribution(probability_table)
            # distribution.bake()
            return distribution
        else:
            probability_table = []
            conditioning_values_instances = self.probabilities.keys()
            for conditioning_values_instance in conditioning_values_instances:
                for i, value in enumerate(self.variable.values):
                    values_instance = list(conditioning_values_instance) + [value]
                    conditional_probabilities = self.probabilities[conditioning_values_instance]
                    values_instance_probability = conditional_probabilities[i]
                    probability_table_row = values_instance + [values_instance_probability]
                    probability_table.append(probability_table_row)
            probability_table = sorted(probability_table)
            distribution = pomegranate.ConditionalProbabilityTable(probability_table, conditioning_distributions)
            # distribution.bake()
            return distribution


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


    def create_pomegranate_probability_distributions(self):
        pomegranate_probability_distributions = {}
        for variable in self.variables.values():
            if variable.probdist.conditioning_set is None:
                pomegranate_probability_distributions[variable.name] = variable.probdist.to_pomegranate_distribution()

        must_repeat = True
        while must_repeat == True:
            must_repeat = False
            for variable in self.variables.values():
                if not variable.probdist.conditioning_set is None:
                    conditioning_variables = variable.probdist.conditioning_set
                    try:
                        conditioning_distributions = []
                        for conditioning_variable in conditioning_variables:
                            conditioning_distributions.append(pomegranate_probability_distributions[conditioning_variable])
                        pomegranate_probability_distributions[variable.name] = variable.probdist.to_pomegranate_distribution(conditioning_distributions)
                    except KeyError:
                        must_repeat = True
                        continue

        return pomegranate_probability_distributions





