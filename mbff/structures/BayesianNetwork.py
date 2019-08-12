import itertools
import operator
import random
import copy

import numpy

from collections import OrderedDict

from mbff.math.PMF import PMF
from mbff.structures.Exceptions import BayesianNetworkNotFinalizedError


def finalization_required(func):
    def wrapper_guard_finalized(*args, **kwargs):
        instance = args[0]
        if instance.finalized is False:
            raise BayesianNetworkNotFinalizedError(instance, "Cannot call method {}()".format(func.__name__))
        else:
            return func(*args, **kwargs)
    return wrapper_guard_finalized


class BayesianNetwork:

    def __init__(self, name):
        self.name = name
        self.variable_nodes = {}
        self.properties = {}
        self.variable_nodes__sampling_order = []
        self.variable_node_names__sampling_order = []
        self.graph = None
        self.finalized = False


    def variable_node_names(self):
        return list(sorted(self.variable_nodes.keys()))


    def variable_nodes_index(self, varname):
        return self.variable_node_names().index(varname)


    @finalization_required
    def sample(self, as_list=False, values_as_indices=False):
        sample = {}

        if len(self.variable_nodes__sampling_order) == 0:
            # If no optimal sampling order for VariableNodes has been specified,
            # then sample all VariableNodes in no specific order, but enabling
            # recursive sampling of conditioning VariableNodes. This is not a fast
            # process.
            while len(sample) < len(self.variable_nodes):
                for varname, variable in self.variable_nodes.items():
                    sample = variable.sample(sample, recursive=True)

        else:
            # An optimal sampling order has been specified for VariableNodes. This
            # means that recursive sampling can be disabled without causing any
            # exceptions when calling VariableNode.sample(). Disabling recursive
            # sampling makes everything much faster.
            for variable in self.variable_nodes__sampling_order:
                sample = variable.sample(sample, recursive=False)

        if values_as_indices:
            sample = self.sample_values_to_indices(sample)

        if as_list:
            sample = self.sample_as_list(sample)
        return sample


    @finalization_required
    def sample_as_list(self, sample):
        return [sample[varname] for varname in self.variable_node_names()]


    @finalization_required
    def samples(self, n=1, as_list=False, values_as_indices=False):
        return [self.sample(as_list, values_as_indices) for i in range(n)]


    @finalization_required
    def sample_matrix(self, n=1, dtype=None):
        if dtype is None:
            dtype = numpy.int8
        samples = self.samples(n, as_list=True, values_as_indices=True)
        return numpy.asarray(list(samples), dtype=dtype)


    def sample_values_to_indices(self, sample):
        sample_with_indices = {}
        for varname, value in sample.items():
            sample_with_indices[varname] = self.variable_nodes[varname].values.index(value)

        return sample_with_indices


    @finalization_required
    def create_joint_pmf(self, values_as_indices=True):
        pmf = PMF(None)
        pmf.probabilities = self.joint_values_and_probabilities(values_as_indices=values_as_indices)
        return pmf


    @finalization_required
    def joint_values_and_probabilities(self, values_as_indices=True, joint_vp=dict(), current_value=dict(), current_probability=1.0):
        if len(current_value) == len(self.variable_nodes__sampling_order):
            if values_as_indices:
                current_value = self.sample_values_to_indices(current_value)
            current_value_as_list = self.sample_as_list(current_value)
            current_value_tuple = tuple(current_value_as_list)
            joint_vp[current_value_tuple] = current_probability
            return joint_vp

        current_variable = self.variable_nodes__sampling_order[len(current_value)]
        for value in current_variable.values:
            conditioning_values = current_variable.get_conditioning_values_from_partial_sample(current_value)
            if len(conditioning_values) == 0:
                conditioning_values = '<unconditioned>'
            current_value_probability = current_probability * current_variable.probability_of_value(value, conditioning_values)
            current_value[current_variable.name] = value
            self.joint_values_and_probabilities(values_as_indices, joint_vp, current_value, current_value_probability)

        del current_value[current_variable.name]
        return joint_vp


    def total_possible_values_count(self):
        count = 1
        for varnode in self.variable_nodes.values():
            count *= len(varnode.values)
        return count


    def finalize(self):
        for variable in self.variable_nodes.values():
            variable.probdist.finalize()

        if len(self.variable_node_names__sampling_order) == 0:
            self.variable_node_names__sampling_order = self.detect_optimal_variable_sampling_order()

        for varname in self.variable_node_names__sampling_order:
            self.variable_nodes__sampling_order.append(self.variable_nodes[varname])

        for ID, varname in enumerate(self.variable_node_names()):
            self.variable_nodes[varname].ID = ID

        self.graph_d = self.as_directed_graph()
        self.graph_u = self.as_undirected_graph()

        self.finalized = True


    def detect_optimal_variable_sampling_order(self):
        optimal_sampling_order = []
        variable_names = self.variable_node_names()

        while len(variable_names) > 0:
            for varname in variable_names:
                variable = self.variable_nodes[varname]
                conditioning_varnames = variable.probdist.conditioning_variable_nodes.keys()
                if len(conditioning_varnames) == 0 or set(conditioning_varnames).issubset(optimal_sampling_order):
                    optimal_sampling_order.append(variable.name)
            variable_names = [varname for varname in variable_names if varname not in optimal_sampling_order]

        return optimal_sampling_order


    def as_directed_graph(self):
        graph = {}
        for node in self.variable_nodes.values():
            cond_nodes = node.probdist.conditioning_variable_nodes.values()
            if len(cond_nodes) > 0:
                for cnode in cond_nodes:
                    try:
                        graph[cnode.ID].append(node.ID)
                    except KeyError:
                        graph[cnode.ID] = [node.ID]
        for i in graph:
            graph[i] = sorted(graph[i])

        for node in self.variable_nodes.values():
            if node.ID not in graph:
                graph[node.ID] = []

        return graph


    def as_undirected_graph(self):
        graph = {}
        for node in self.variable_nodes.values():
            cond_nodes = node.probdist.conditioning_variable_nodes.values()
            if len(cond_nodes) > 0:
                for cnode in cond_nodes:
                    try:
                        if node.ID not in graph[cnode.ID]:
                            graph[cnode.ID].append(node.ID)
                    except KeyError:
                        graph[cnode.ID] = [node.ID]
                    try:
                        if cnode.ID not in graph[node.ID]:
                            graph[node.ID].append(cnode.ID)
                    except KeyError:
                        graph[node.ID] = [cnode.ID]
        for i in graph:
            graph[i] = sorted(graph[i])

        for node in self.variable_nodes.values():
            if node.ID not in graph:
                graph[node.ID] = []

        return graph


    def from_directed_graph(self, graph):
        self.graph_d = graph.copy()
        self.graph_u = {}
        all_nodes = set()
        for node, descendants in self.graph_d.items():
            all_nodes.add(node)
            if len(descendants) > 0:
                for descendant in descendants:
                    all_nodes.add(descendant)
                    # For each directed connection in self.graph_d from `node`
                    # to `descendant`, add two connections in self.graph_u: one
                    # from `node` to `descendant`, and one from `descendant` to
                    # `node`.
                    try:
                        if descendant not in self.graph_u[node]:
                            self.graph_u[node].append(descendant)
                    except KeyError:
                        self.graph_u[node] = [descendant]
                    try:
                        if node not in self.graph_u[descendant]:
                            self.graph_u[descendant].append(node)
                    except KeyError:
                        self.graph_u[descendant] = [node]

        for node in all_nodes:
            try:
                self.graph_u[node] = sorted(self.graph_u[node])
            except KeyError:
                self.graph_u[node] = []

            try:
                self.graph_d[node] = sorted(self.graph_d[node])
            except KeyError:
                self.graph_d[node] = []


    def conditionally_independent(self, x, y, conditioning_set):
        return self.d_separated(x, conditioning_set, y)


    def d_separated(self, x, separators, y):
        if isinstance(separators, int):
            separators = [separators]
        paths = self.find_all_undirected_paths(x, y)
        for path in paths:
            if not self.is_path_blocked_by_nodes(path, separators):
                return False
        return True


    def is_path_blocked_by_nodes(self, path, conditioning_nodes):
        for i, node in enumerate(path):
            if i == 0 or i == (len(path) - 1):
                continue
            descendants = self.graph_d[node]
            is_collider = self.is_node_collider(path, i)
            is_conditioned_on = node in conditioning_nodes
            descendants_in_conditioning_nodes = set(descendants) & set(conditioning_nodes)
            has_descendants_in_conditioning_nodes = (len(descendants_in_conditioning_nodes) > 0)

            if is_collider:
                if not is_conditioned_on and not has_descendants_in_conditioning_nodes:
                    return True
            if not is_collider:
                if is_conditioned_on:
                    return True

        return False


    def is_node_collider(self, path, i):
        node_prev = path[i - 1]
        node_this = path[i]
        node_next = path[i + 1]

        if (node_this in self.graph_d[node_prev]) and (node_this in self.graph_d[node_next]):
            return True
        else:
            return False


    def find_all_directed_paths(self, start, end):
        return self.find_all_paths(self.graph_d, start, end)


    def find_all_undirected_paths(self, start, end, path=[]):
        return self.find_all_paths(self.graph_u, start, end)


    def find_all_paths(self, graph, start, end, path=[]):
        path = path + [start]
        if start == end:
            return [path]
        if start not in graph:
            return []
        paths = []
        for node in graph[start]:
            if node not in path:
                newpaths = self.find_all_paths(graph, node, end, path)
                for newpath in newpaths:
                    paths.append(newpath)
        return paths




class VariableNode:
    """
    Class representing a VariableNode in a :class:`BayesianNetwork`.

    The `VariableNode` class is tightly bound to the :class:`ProbabilityDistributionOfVariableNode` class.

    The VariableNode can be *sampled*, which means that we can request a random
    value to be produced by this variable, according to its probability
    distribution. The produced sample also contains the values produced when
    the *conditioning VariableNodes* were sampled as well, a required step before
    sampling the current VariableNode. Sampling of conditioning VariableNodes happens
    *recursively*, i.e. if a conditioning VariableNode has a conditioning VariableNode
    of its own, it will be sampled as well.

    :var str name: The name of the VariableNode. Must be unique within a
        BayesianNetwork instance.
    :var list(str) values: The list of possible values this VariableNode can take.
        The values (categories) must be strings.
    :var dict properties: A dictionary containing metadata about this VariableNode
        (e.g. any properties read from a BIF file).
    :var ProbabilityDistributionOfVariableNode probdist: The :class:`ProbabilityDistributionOfVariableNode`
        object that describes the probability distribution of this VariableNode.
    """

    def __init__(self, name):
        self.ID = -1
        self.name = name
        self.values = []
        self.properties = {}
        self.probdist = None


    def sample(self, partial_sample={}, recursive=True):
        """
        Produce a random sample that respects the probability distribution of
        the VariableNode.

        Returns the sample as a dictionary which also contains the values of
        the conditioning VariableNodes (they had to be sampled too, after all).
        """

        # Don't do anything if this VariableNode has already been sampled.
        if self.name in partial_sample:
            return partial_sample

        # Determine whether we need to sample any conditioning VariableNodes or
        # not. Conditioning VariableNodes need to be sampled and the sample value
        # must be added to `partial_sample`, unless these VariableNodes have
        # already been sampled (in which case `partial_sample` already contains
        # their values).
        if len(self.probdist.conditioning_variable_nodes) == 0:
            # This VariableNode is not conditioned by any other.
            conditioning_values = '<unconditioned>'
        else:
            # If recursive sampling of conditioning VariableNodes is enabled, then
            # iterate over not-yet-sampled conditioning VariableNodes and sample them now.
            # If recursive sampling is NOT enabled, then ignore this step,
            # because BayesianNetwork.sample() should already know the proper
            # sampling order which ensures all conditioning VariableNodes are
            # sampled before a conditioned VariableNode.
            if recursive:
                # This VariableNode is conditioned by other VariableNodes. We need to
                # determine which of the conditioning VariableNodes have been sampled
                # already, and which have not.
                unsampled_conditioning_variables = self.get_unsampled_conditioning_variables(partial_sample)
                # Recursively sample all unsampled conditioning VariableNodes.
                for unsampled_variable in unsampled_conditioning_variables:
                    partial_sample = unsampled_variable.sample(partial_sample, recursive=True)
            conditioning_values = self.get_conditioning_values_from_partial_sample(partial_sample)

        value_index = self.probdist.sample(conditioning_values)
        value = self.values[value_index]
        partial_sample[self.name] = value
        return partial_sample


    def probability_of_value(self, value, conditioning_values):
        value_index = self.values.index(value)
        value_probability = self.probdist.probabilities[conditioning_values][value_index]
        return value_probability


    def get_unsampled_conditioning_variables(self, partial_sample):
        """
        Find out what VariableNodes that are in our conditioning set have not yet
        been sampled, thus not added to partial_sample.
        """
        sampled_variable_names = set(partial_sample.keys())
        conditioning_variable_names = set(self.probdist.conditioning_variable_nodes.keys())
        unsampled_conditioning_variable_names = list(conditioning_variable_names - sampled_variable_names)

        unsampled_conditioning_variables = []
        for varname in unsampled_conditioning_variable_names:
            unsampled_conditioning_variables.append(varname)

        return unsampled_conditioning_variables


    def get_conditioning_values_from_partial_sample(self, partial_sample):
        conditioning_values = []
        for varname in self.probdist.conditioning_variable_nodes.keys():
            conditioning_values.append(partial_sample[varname])
        return tuple(conditioning_values)


    def __str__(self):
        return 'VariableNode "{}" with values {}'.format(self.name, self.values)



class ProbabilityDistributionOfVariableNode:
    """
    Class representing the probability distribution of a VariableNode in a BayesianNetwork.

    The `ProbabilityDistributionOfVariableNode` class is tightly bound to the :class:`VariableNode` class.
    """

    def __init__(self, var):
        self.variable_name = ''
        self.variable = None
        if isinstance(var, str):
            self.variable_name = var
            self.variable = None
        if isinstance(var, VariableNode):
            self.variable = var
            self.variable_name = self.variable.name
        self.probabilities = {}
        self.probabilities_with_indexed_conditioning = {}
        self.conditioning_variable_nodes = OrderedDict()
        self.cummulative_probabilities = OrderedDict()
        self.properties = {}


    def __eq__(self, other):
        """
        Two ProbabilityDistributionOfVariableNode objects are equal iff:

        * their probabilities tables are identical
        * their conditioning variables give identical values
        """
        eq_probabilities = self.probabilities == other.probabilities
        eq_conditioning_values = True
        if len(self.conditioning_variable_nodes) != len(other.conditioning_variable_nodes):
            eq_conditioning_values = False
        else:
            for ourVar, otherVar in zip(self.conditioning_variable_nodes, other.conditioning_variable_nodes):
                eq_probabilities = eq_probabilities and (ourVar.values == otherVar.values)

        return eq_probabilities and eq_conditioning_values


    def finalize(self):
        self.cummulative_probabilities = self.create_cummulative_probabilities(self.probabilities)
        if '<unconditioned>' not in self.probabilities:
            self.probabilities_with_indexed_conditioning = self.create_probabilities_with_indexed_conditioning()


    def create_cummulative_probabilities(self, probabilities):
        cummulative_probabilities = OrderedDict()
        for key, probs in probabilities.items():
            cummulative_probabilities[key] = list(itertools.accumulate(probs, func=operator.add))
        return cummulative_probabilities


    def create_probabilities_with_indexed_conditioning(self):
        prob_indexed = {}
        for conditioning_values in self.probabilities:
            indexed_conditioning_values = []
            for i, value in enumerate(conditioning_values):
                cond_variable = list(self.conditioning_variable_nodes.values())[i]
                value_index = cond_variable.values.index(value)
                indexed_conditioning_values.append(value_index)
            prob_indexed[tuple(indexed_conditioning_values)] = self.probabilities[conditioning_values]

        return prob_indexed


    def copy(self):
        # Instantiate a new ProbabilityDistributionOfVariableNode that references the same
        # VariableNode object.
        new = ProbabilityDistributionOfVariableNode(self.variable)
        # Create a deep copy of the probabilities and of the cummulative
        # probabilities, which are both dicts that map tuples of strings to lists
        # of floats.
        new.probabilities = copy.deepcopy(self.probabilities)
        new.cummulative_probabilities = copy.deepcopy(self.cummulative_probabilities)
        # Create a shallow copy of the OrderedDict containing the conditioning
        # VariableNodes. The new OrderedDict will thus reference the same instances
        # of VariableNode.
        new.conditioning_variable_nodes = self.conditioning_variable_nodes.copy()
        # Create a deep copy of the properties dict.
        new.properties = copy.deepcopy(self.properties)
        return new


    def sample(self, conditioning_values='<unconditioned>'):
        roll = random.random()
        cummulative_probabilities = self.cummulative_probabilities[conditioning_values]
        index = self.sample_roll(roll, cummulative_probabilities)
        return index


    def sample_roll(self, roll, cummulative_probabilities):
        for index, prob in enumerate(cummulative_probabilities):
            if roll <= prob:
                return index
            else:
                continue


    def __str__(self):
        if len(self.conditioning_variable_nodes) == 0:
            return "ProbabilityMassDistribution for variable {}, unconditioned".format(self.variable_name)
        else:
            return "ProbabilityMassDistribution for variable {}, conditioned on {}".format(self.variable_name, self.conditioning_variable_names)
