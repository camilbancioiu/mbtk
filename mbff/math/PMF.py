import itertools
from collections import Counter
from mbff.utilities import functions as util


class PMF:

    def __init__(self, variable):
        self.tolerance_pdiff = 1e-10
        self.variable = variable
        if variable is not None:
            self.total_count = len(self.variable)
            self.value_counts = self.count_values()
            self.probabilities = self.normalize_counts()
        else:
            self.value_counts = dict()
            self.probabilities = dict()
            self.total_count = 0


    def __str__(self):
        output = ''
        keys = sorted(self.probabilities.keys())
        for key in keys:
            prob = self.probabilities[key]
            output += '{}: {}\n'.format(key, prob)
        return output


    def __len__(self):
        return len(self.probabilities)


    def __getitem__(self, key):
        return self.p(key)


    def items(self):
        return self.probabilities.items()


    def count_instance(self, value, count=1):
        try:
            self.value_counts[value] += count
        except KeyError:
            self.value_counts[value] = count
        self.total_count += count


    def keys(self):
        return self.probabilities.keys()


    def values(self):
        return self.probabilities.values()


    def remove_zeros(self):
        keys_to_delete = list()
        for key, p in self.probabilities.items():
            if p == 0.0:
                keys_to_delete.append(key)
        for key in keys_to_delete:
            del self.probabilities[key]


    def p(self, *args):
        key = process_pmf_key(args)

        try:
            return self.probabilities[key]
        except KeyError:
            return 0.0


    def count_values(self):
        counter = Counter(self.variable.instances())
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


    def min_instance_count_for_accuracy(self):
        nonzero_probabilities = filter(None, self.values())
        return round(1 / min(nonzero_probabilities))


    def create_instances_list(self, n=1):
        sorted_keys = sorted(self.keys(), key=lambda k: self.probabilities[k], reverse=True)
        instances = itertools.repeat(None, 0)

        running_instance_count = n

        for key in sorted_keys:
            probability = self.probabilities[key]
            value_instance_count = max(1, round(n * probability))
            if value_instance_count > running_instance_count:
                value_instance_count = running_instance_count
            running_instance_count -= value_instance_count
            instances = itertools.chain(instances, itertools.repeat(key, value_instance_count))
            if running_instance_count == 0:
                break

        if running_instance_count > 0:
            extra_instances = self.create_instances_list(running_instance_count)
            instances = itertools.chain(instances, extra_instances)

        return instances


    def __eq__(self, other):
        selfkeyset = set(self.probabilities.keys())
        otherkeyset = set(other.probabilities.keys())
        keys_equal = selfkeyset == otherkeyset
        if keys_equal is False:
            return False
        for key in (selfkeyset | otherkeyset):
            selfp = self.p(key)
            otherp = other.p(key)
            if abs(selfp - otherp) > self.tolerance_pdiff:
                return False
        return True



class CPMF(PMF):

    def __init__(self, variable, given):
        super().__init__(variable)
        self.conditional_probabilities = dict()
        self.tolerance_pdiff = 1e-10

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


    def __len__(self):
        return len(self.conditional_probabilities)


    def __getitem__(self, key):
        return self.given(key)


    def keys(self):
        return self.conditional_probabilities.keys()


    def items(self):
        return self.conditional_probabilities.items()


    def given(self, *args):
        key = process_pmf_key(args)

        try:
            return self.conditional_probabilities[key]
        except KeyError:
            return PMF(None)


    def count_values_conditionally(self):
        conditional_counts = {}
        for (v, cv) in zip(self.variable.instances(), self.conditioning_variable.instances()):
            try:
                conditional_counts[cv].count_instance(v)
            except KeyError:
                conditional_counts[cv] = PMF(None)
                conditional_counts[cv].count_instance(v)
        return conditional_counts


    def normalize_conditional_counts(self):
        for cv, pmf in self.conditional_probabilities.items():
            pmf.normalize_counts(update_probabilities=True)


    def __eq__(self, other):
        selfkeyset = set(self.conditional_probabilities.keys())
        otherkeyset = set(other.conditional_probabilities.keys())
        keys_equal = (selfkeyset == otherkeyset)
        if keys_equal is False:
            return False
        for key in self.conditional_probabilities:
            selfpmf = self.given(key)
            otherpmf = other.given(key)
            if selfpmf != otherpmf:
                return False
        return True



def process_pmf_key(key):
    # If the key is a tuple or list, flatten it.
    key = util.flatten(key)
    # Convert the key to a tuple, in case it is a list.
    key = tuple(key)
    # If the key only has a single element, take that as the key.
    if len(key) == 1:
        key = key[0]
    return key



def cpmf_diff(A, B):
    import mbff.utilities.colors as col

    output = ""
    pdiff_threshold = 1e-10

    condkeysUnion = sorted(set(list(A.keys()) + list(B.keys())))

    jointkeysUnion = []
    for cpmf in [A, B]:
        for condkey in condkeysUnion:
            jointkeys = list(cpmf.given(condkey).keys())
            jointkeysUnion.extend(jointkeys)
    jointkeysUnion = sorted(set(jointkeysUnion))

    width_condkey = max([len(str(condkey)) for condkey in condkeysUnion])
    width_jointkey = max([len(str(jointkey)) for jointkey in jointkeysUnion])

    formatString = (
        '| {:>' + str(width_condkey) + '} |'
        ' {:>' + str(width_jointkey) + '} |'
        ' {:>8.6f} |'
        ' {:>8.6f} |'
        '\n'
    )
    row_width = 4 + width_condkey + 3 + width_jointkey + 3 + 8 + 3 + 8

    for condkey in condkeysUnion:
        XYA = A.given(condkey)
        XYB = B.given(condkey)

        output += row_width * '-' + '\n'
        for jointkey in jointkeysUnion:
            xyap = XYA.p(jointkey)
            xybp = XYB.p(jointkey)

            row = formatString.format(str(condkey), str(jointkey), xyap, xybp)
            if abs(xyap - xybp) > pdiff_threshold:
                row = col.red(row)

            output += row
            output += row_width * '-' + '\n'

        output += '\n'

    return output



def pmf_diff(A, B):
    import mbff.utilities.colors as col

    output = ""
    pdiff_threshold = 1e-10

    jointkeysUnion = sorted(set(list(A.keys()) + list(B.keys())))

    width_jointkey = max([len(str(jointkey)) for jointkey in jointkeysUnion])

    formatString = (
        '| {:>' + str(width_jointkey) + '} |'
        ' {:>8.6f} |'
        ' {:>8.6f} |'
        '\n'
    )
    row_width = 4 + width_jointkey + 3 + 8 + 3 + 8

    output += row_width * '-' + '\n'
    for jointkey in jointkeysUnion:
        ap = A.p(jointkey)
        bp = B.p(jointkey)

        row = formatString.format(str(jointkey), ap, bp)
        if abs(ap - bp) > pdiff_threshold:
            row = col.red(row)

        output += row
        output += row_width * '-' + '\n'

    return output
