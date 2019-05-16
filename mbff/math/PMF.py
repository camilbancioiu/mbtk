import numpy
from collections import Counter


class PMF:

    def __init__(self, variable):
        self.variable = variable
        if not variable is None:
            self.total_count = len(self.variable.instances)
            self.value_counts = self.count_values()
            self.probabilities = self.normalize_counts()
        else:
            self.value_counts = {}
            self.probabilities = {}
            self.total_count = 0


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
        if len(args) == 1:
            key = args[0]
        else:
            key = tuple(args)

        try:
            return self.probabilities[key]
        except KeyError:
            return 0.0


    def count_values(self):
        counter = Counter(self.variable.instances)
        return counter


    def normalize_counts(self, update_probabilities=False):
        normalized_counts = {}
        for value, count in self.value_counts.items():
            normalized_counts[value] = (count * 1.0) / self.total_count

        if update_probabilities:
            self.probabilities = normalized_counts

        return normalized_counts


    def expected_value(self, f):
        return sum([p * f(v, p) for (v, p) in self.probabilities.items()])



class CPMF(PMF):

    def __init__(self, variable, conditioning_variable):
        super().__init__(variable)

        self.conditioning_variable = conditioning_variable
        self.conditional_probabilities = self.count_values_conditionally()
        self.normalize_conditional_counts()


    def given(self, *args):
        if len(args) == 1:
            key = args[0]
        else:
            key = tuple(args)

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



