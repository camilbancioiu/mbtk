from collections import OrderedDict
from lark import Transformer

from mbff.structures.BayesianNetwork import BayesianNetwork, VariableNode, ProbabilityDistributionOfVariableNode


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
        variable = VariableNode(attributes['variable_name'])
        variable.values = attributes['value_list']
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
        return [('<unconditioned>', items[0])]


    def variable_value_probabilities__cond(self, items):
        return items


    def variable_value_probabilities__cond_line(self, items):
        return (tuple(items[0]), items[1])


    def conditioning_variable_names(self, items):
        return [('conditioning_variable_names', items[0])]


    def variable_value_probabilities(self, items):
        return [('probabilities', dict(items[0]))]


    def probability_block(self, items):
        attributes = dict([item for sublist in items for item in sublist])
        pd = ProbabilityDistributionOfVariableNode(attributes['variable_name'])
        pd.probabilities = attributes['probabilities']
        conditioning_variable_names = attributes.get('conditioning_variable_names', [])
        pd.conditioning_variable_nodes = OrderedDict.fromkeys(conditioning_variable_names)
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
        # Firstly, gather 'network_name' and all the Variables.
        for item in items:
            if isinstance(item, dict):
                bn.name = item['network_name']
                bn.properties = item.get('properties', {})
            if isinstance(item, VariableNode):
                bn.variable_nodes[item.name] = item

        # Secondly, add references from VariableNodes to ProbabilityDistributionOfVariableNode and vice-versa.
        for item in items:
            if isinstance(item, ProbabilityDistributionOfVariableNode):
                variable = bn.variable_nodes[item.variable_name]
                pd = item
                variable.probdist = pd
                pd.variable = variable

                for varname in pd.conditioning_variable_nodes.keys():
                    pd.conditioning_variable_nodes[varname] = bn.variable_nodes[varname]

        return bn
