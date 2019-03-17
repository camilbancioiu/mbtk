from lark import Transformer

from mbff.utilities.bif.BayesianNetwork import *

def get_transformer_chain():
    basic = BIFTransformerBasic()
    variables = BIFTransformerVariables()
    probabilities = BIFTransformerProbabilities()
    network = BIFTransformerNetwork()
    return basic * variables * probabilities * network


class BIFTransformerBasic(Transformer):
    def identifier(self, items):
        return str(items[0])

    def string_value(self, items):
        return str(items[0])

    def property_value(self, items):
        return str(items[0])

    def property(self, items):
        return (items[0], items[1])

    def property_list(self, items):
        return [('properties', dict(items))]



class BIFTransformerVariables(Transformer):

    def variable_identifier(self, items):
        return [('variable_name', str(items[0]))]

    def variable_value_count(self, items):
        return ('value_count', int(items[0]))

    def variable_value_list(self, items):
        return ('value_list', list(items))

    def variable_type_def(self, items):
        return items

    def variable_block(self, items):
        attributes = dict([item for sublist in items for item in sublist])
        variable = Variable(attributes['variable_name'])
        variable.values = attributes['value_list']
        variable.value_count = len(variable.values)
        variable.properties = attributes['properties']
        return variable

    def variable_list(self, items):
        return [item[0][1] for item in items]



class BIFTransformerProbabilities(Transformer):
    probabilities_list = list

    def probability_value(self, items):
        return float(items[0])

    def variable_value_tuple(self, items):
        return items[0][1]

    def variable_value_probabilities__simple(self, items):
        return [('<unconditional>', items[0])]

    def variable_value_probabilities__cond(self, items):
        return items

    def variable_value_probabilities__cond_line(self, items):
        return (tuple(items[0]), items[1])

    def conditioning_set(self, items):
        return [('conditioning_set', items[0])]

    def variable_value_probabilities(self, items):
        return [('probabilities', dict(items[0]))]

    def probability_block(self, items):
        attributes = dict([item for sublist in items for item in sublist])
        pd = ProbabilityDistribution(attributes['variable_name'])
        pd.conditioning_set = attributes.get('conditioning_set', None)
        pd.probabilities = attributes['probabilities']
        pd.properties = attributes['properties']

        return pd


class BIFTransformerNetwork(Transformer):

    def network_name(self, items):
        return [('network_name', str(items[0]))]

    def network_block(self, items):
        attributes = dict([item for sublist in items for item in sublist])
        return attributes

    def network_definition(self, items):
        bn = BayesianNetwork('')
        for item in items:
            if isinstance(item, dict):
                bn.name = item['network_name']
                bn.properties = item.get('properties', {})
            if isinstance(item, Variable):
                bn.variables[item.name] = item

        for item in items:
            if isinstance(item, ProbabilityDistribution):
                variable = bn.variables[item.variable_name]
                pd = item
                variable.probdist = pd
                pd.variable = variable

        return bn



