import itertools
import operator
import random
import copy

import numpy

from collections import OrderedDict

from mbff.dataset.Exceptions import BayesianNetworkFinalizedError, BayesianNetworkNotFinalizedError


def finalization_required(func):
    def wrapper_guard_finalized(*args, **kwargs):
        instance = args[0]
        if instance.finalized == False:
            raise BayesianNetworkNotFinalizedError(instance, "Cannot call method {}()".format(func.__name__))
        else:
            return func(*args, **kwargs)
    return wrapper_guard_finalized


class BayesianNetwork:

    def __init__(self, name):
        self.name = name
        self.variables = {}
        self.properties = {}
        self.variables__sampling_order = []
        self.variable_names__sampling_order = []
        self.finalized = False


    def variable_names(self):
        return list(sorted(self.variables.keys()))


    def variable_index(self, varname):
        return self.variable_names().index(varname)


    @finalization_required
    def sample(self, as_list=False, values_as_indices=False):
        sample = {}

        if len(self.variables__sampling_order) == 0:
            # If no optimal sampling order for Variables has been specified,
            # then sample all Variables in no specific order, but enabling
            # recursive sampling of conditioning Variables. This is not a fast
            # process.
            while len(sample) < len(self.variables):
                for varname, variable in self.variables.items():
                    sample = variable.sample(sample, recursive=True)

        else:
            # An optimal sampling order has been specified for Variables. This
            # means that recursive sampling can be disabled without causing any
            # exceptions when calling Variable.sample(). Disabling recursive
            # sampling makes everything much faster.
            for variable in self.variables__sampling_order:
                sample = variable.sample(sample, recursive=False)

        if values_as_indices:
            sample = self.sample_values_to_indices(sample)

        if as_list:
            sample = self.sample_as_list(sample)
        return sample


    @finalization_required
    def sample_as_list(self, sample):
        return [sample[varname] for varname in self.variable_names()]


    @finalization_required
    def samples(self, n=1, as_list=False, values_as_indices=False):
        return [self.sample(as_list, values_as_indices) for i in range(n)]


    @finalization_required
    def sample_matrix(self, n=1):
        samples = self.samples(n, as_list=True, values_as_indices=True)
        return numpy.asarray(list(samples))


    def sample_values_to_indices(self, sample):
        sample_with_indices = {}
        for varname, value in sample.items():
            sample_with_indices[varname] = self.variables[varname].values.index(value)

        return sample_with_indices


    def finalize(self):
        for variable in self.variables.values():
            variable.probdist.finalize()

        if len(self.variable_names__sampling_order) == 0:
            self.variable_names__sampling_order = self.detect_optimal_variable_sampling_order()

        for varname in self.variable_names__sampling_order:
            self.variables__sampling_order.append(self.variables[varname])

        self.finalized = True


    def detect_optimal_variable_sampling_order(self):
        optimal_sampling_order = []
        variable_names = self.variable_names()

        while len(variable_names) > 0:
            for varname in variable_names:
                variable = self.variables[varname]
                conditioning_varnames = variable.probdist.conditioning_variables.keys()
                if len(conditioning_varnames) == 0 or set(conditioning_varnames).issubset(optimal_sampling_order):
                    optimal_sampling_order.append(variable.name)
            variable_names = [varname for varname in variable_names if varname not in optimal_sampling_order]

        return optimal_sampling_order





class Variable:
    """
    Class representing a Variable (Node) in a :class:`BayesianNetwork`.

    The `Variable` class is tightly bound to the :class:`ProbabilityDistribution` class.

    The Variable can be *sampled*, which means that we can request a random
    value to be produced by this variable, according to its probability
    distribution. The produced sample also contains the values produced when
    the *conditioning Variables* were sampled as well, a required step before
    sampling the current Variable. Sampling of conditioning Variables happens
    *recursively*, i.e. if a conditioning Variable has a conditioning Variable
    of its own, it will be sampled as well.

    :var str name: The name of the Variable. Must be unique within a
        BayesianNetwork instance.
    :var list(str) values: The list of possible values this Variable can take.
        The values (categories) must be strings.
    :var dict properties: A dictionary containing metadata about this Variable
        (e.g. any properties read from a BIF file).
    :var ProbabilityDistribution probdist: The :class:`ProbabilityDistribution`
        object that describes the probability distribution of this Variable.
    """

    def __init__(self, name):
        self.name = name
        self.values = []
        self.properties = {}
        self.probdist = None


    def sample(self, partial_sample={}, recursive=True):
        """
        Produce a random sample that respects the probability distribution of
        the Variable.

        Returns the sample as a dictionary which also contains the values of
        the conditioning Variables (they had to be sampled too, after all).
        """

        # Don't do anything if this Variable has already been sampled.
        if self.name in partial_sample:
            return partial_sample

        # Determine whether we need to sample any conditioning Variables or
        # not. Conditioning Variables need to be sampled and the sample value
        # must be added to `partial_sample`, unless these Variables have
        # already been sampled (in which case `partial_sample` already contains
        # their values).
        if len(self.probdist.conditioning_variables) == 0:
            # This Variable is not conditioned by any other.
            conditioning_values = '<unconditioned>'
        else:
            # If recursive sampling of conditioning Variables is enabled, then
            # iterate over not-yet-sampled conditioning Variables and sample them now.
            # If recursive sampling is NOT enabled, then ignore this step,
            # because BayesianNetwork.sample() should already know the proper
            # sampling order which ensures all conditioning Variables are
            # sampled before a conditioned Variable.
            if recursive:
                # This Variable is conditioned by other Variables. We need to
                # determine which of the conditioning Variables have been sampled
                # already, and which have not.
                unsampled_conditioning_variables = self.get_unsampled_conditioning_variables(partial_sample)
                # Recursively sample all unsampled conditioning Variables.
                for unsampled_variable in unsampled_conditioning_variables:
                    partial_sample = unsampled_variable.sample(partial_sample, recursive=True)
            conditioning_values = self.get_conditioning_values_from_partial_sample(partial_sample)

        value_index = self.probdist.sample(conditioning_values)
        value = self.values[value_index]
        partial_sample[self.name] = value
        return partial_sample


    def get_unsampled_conditioning_variables(self, partial_sample):
        """
        Find out what Variables that are in our conditioning set have not yet
        been sampled, thus not added to partial_sample.
        """
        sampled_variable_names = set(partial_sample.keys())
        conditioning_variable_names = set(self.probdist.conditioning_variables.keys())
        unsampled_conditioning_variable_names = list(conditioning_variable_names - sampled_variable_names)

        unsampled_conditioning_variables = []
        for varname in unsampled_conditioning_variable_names:
            unsampled_variable = self.probdist.conditioning_variables[varname]
            unsampled_conditioning_variables.append(varname)

        return unsampled_conditioning_variables


    def get_conditioning_values_from_partial_sample(self, partial_sample):
        conditioning_values = []
        for varname in self.probdist.conditioning_variables.keys():
            conditioning_values.append(partial_sample[varname])
        return tuple(conditioning_values)


    def __str__(self):
        return 'Variable "{}" with values {}'.format(self.name, self.values)



class ProbabilityDistribution:
    """
    Class representing the probability distribution of a Variable in a BayesianNetwork.

    The `ProbabilityDistribution` class is tightly bound to the :class:`Variable` class.
    """

    def __init__(self, var):
        self.variable_name = ''
        self.variable = None
        if isinstance(var, str):
            self.variable_name = var
            self.variable = None
        if isinstance(var, Variable):
            self.variable = var
            self.variable_name = self.variable.name
        self.probabilities = {}
        self.conditioning_variables = OrderedDict()
        self.cummulative_probabilities = OrderedDict()
        self.properties = {}


    def __eq__(self, other):
        """
        Two ProbabilityDistribution objects are equal iff:

        * their probabilities tables are identical
        * their conditioning variables give identical values
        """
        eq_probabilities = self.probabilities == other.probabilities
        eq_conditioning_values = True
        if len(self.conditioning_variables) != len(other.conditioning_variables):
            eq_conditioning_values = False
        else:
            for ourVar, otherVar in zip(self.conditioning_variables, other.conditioning_variables):
                eq_probabilities = eq_probabilities and (ourVar.values == otherVar.values)

        return eq_probabilities and eq_conditioning_values


    def finalize(self):
        self.cummulative_probabilities = self.create_cummulative_probabilities(self.probabilities)


    def create_cummulative_probabilities(self, probabilities):
        cummulative_probabilities = OrderedDict()
        for key, probs in probabilities.items():
            cummulative_probabilities[key] = list(itertools.accumulate(probs, func=operator.add))
        return cummulative_probabilities


    def copy(self):
        # Instantiate a new ProbabilityDistribution that references the same
        # Variable object.
        new = ProbabilityDistribution(self.variable)
        # Create a deep copy of the probabilities and of the cummulative
        # probabilities, which are both dicts that map tuples of strings to lists
        # of floats.
        new.probabilities = copy.deepcopy(self.probabilities)
        new.cummulative_probabilities = copy.deepcopy(self.cummulative_probabilities)
        # Create a shallow copy of the OrderedDict containing the conditioning
        # Variables. The new OrderedDict will thus reference the same instances
        # of Variable.
        new.conditioning_variables = self.conditioning_variables.copy()
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
        if len(self.conditioning_variables) == 0:
            return "ProbabilityMassDistribution for variable {}, unconditioned".format(self.variable_name)
        else:
            return "ProbabilityMassDistribution for variable {}, conditioned on {}".format(self.variable_name, self.conditioning_variable_names)



