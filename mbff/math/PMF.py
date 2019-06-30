import numpy
from collections import Counter
from mbff.utilities import functions as util


class PMF:

    def __init__(self, variable):
        self.variable = variable
        if not variable is None:
            self.total_count = len(self.variable.instances)
            self.value_counts = self.count_values()
            self.probabilities = self.normalize_counts()
        else:
            self.value_counts = dict()
            self.probabilities = dict()
            self.total_count = 0


    def __str__(self):
        return str(self.probabilities)


    def items(self):
        return self.probabilities.items()


    def count_instance(self, value, count=1):
        try:
            self.value_counts[value] += count
        except KeyError:
            self.value_counts[value] = count
        self.total_count += count


    def values(self):
        return self.probabilities.values()


    def p(self, *args):
        key = process_pmf_key(args)

        try:
            return self.probabilities[key]
        except KeyError:
            return 0.0


    def count_values(self):
        counter = Counter(self.variable.instances)
        return counter


    def normalize_counts(self, update_probabilities=False):
        normalized_counts = dict()
        for value, count in self.value_counts.items():
            normalized_counts[value] = (count * 1.0) / self.total_count

        if update_probabilities:
            self.probabilities = normalized_counts

        return normalized_counts


    def expected_value(self, f):
        return sum([p * f(v, p) for (v, p) in self.probabilities.items()])



class CPMF(PMF):

    def __init__(self, variable, given):
        super().__init__(variable)
        self.conditional_probabilities = dict()

        if not (variable is None) and not (given is None):
            self.conditioning_variable = given
            self.conditional_probabilities = self.count_values_conditionally()
            self.normalize_conditional_counts()


    def __str__(self):
        output = ''
        for key, pmf in self.conditional_probabilities.items():
            output += str(key) + ':\n'
            for pmfkey, cp in pmf.items():
                output += '\t{}: {}\n'.format(pmfkey, cp)
        return output


    def given(self, *args):
        key = process_pmf_key(args)

        try:
            return self.conditional_probabilities[key]
        except KeyError:
            return PMF(None)


    def count_values_conditionally(self):
        conditional_counts = {}
        for (v, cv) in zip(self.variable.instances, self.conditioning_variable.instances):
            try:
                conditional_counts[cv].count_instance(v)
            except KeyError:
                conditional_counts[cv] = PMF(None)
                conditional_counts[cv].count_instance(v)
        return conditional_counts


    def normalize_conditional_counts(self):
        for cv, pmf in self.conditional_probabilities.items():
            pmf.normalize_counts(update_probabilities=True)



def process_pmf_key(key):
    # If the key is a tuple or list, flatten it.
    key = util.flatten(key)
    # Convert the key to a tuple, in case it is a list.
    key = tuple(key)
    # If the key only has a single element, take that as the key.
    if len(key) == 1:
        key = key[0]
    return key
