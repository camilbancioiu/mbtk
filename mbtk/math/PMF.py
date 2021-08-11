from __future__ import annotations

import itertools
from typing import cast
from collections import Counter

from mbtk.math.Variable import Variable
from mbtk.utilities import functions as util
import numpy


class PMF:
    tolerance_pdiff: float
    variable: Variable
    variableIDs: tuple[int]

    def __init__(self, variable):
        self.tolerance_pdiff = 1e-10
        self.variable = variable
        if variable is not None:
            self.total_count = len(self.variable)
            self.value_counts = self.count_values()
            self.probabilities = self.normalize_counts()
            self.variableIDs = tuple(variable.IDs())
        else:
            self.value_counts = dict()
            self.probabilities = dict()
            self.total_count = 0
            self.variableIDs = tuple()


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


    def IDs(self, *varIDs: int) -> tuple[int]:
        if len(varIDs) == 0:
            return self.variableIDs
        self.variableIDs = cast(tuple[int], tuple(varIDs))
        return self.variableIDs


    def sum_over(self, ID):
        index = self.variableIDs.index(ID)
        new_probabilities = dict()

        for key, value in self.probabilities.items():
            new_key = process_pmf_key(self.remove_from_key(key, index))
            try:
                new_probabilities[new_key] += value
            except KeyError:
                new_probabilities[new_key] = value

        new_pmf = PMF(None)
        new_pmf.probabilities = new_probabilities
        new_pmf.variableIDs = self.remove_from_key(self.variableIDs, index)

        return new_pmf


    def remove_from_key(self, key, index):
        return tuple(key[:index] + key[index + 1:])


    def count_values(self):
        instances = self.variable.instances()
        if isinstance(instances, numpy.ndarray):
            return dict(zip(*numpy.unique(instances, return_counts=True)))
        return Counter(instances)


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
        nonzero_probabilities = list(filter(None, self.values()))
        min_nonzero_probability = min(nonzero_probabilities)
        min_instance_count = round(1 / min_nonzero_probability)
        return min_instance_count


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


    def condition_on(self, cond_pmf: PMF) -> CPMF:
        cond_vars = cond_pmf.IDs()

        if len(cond_vars) == 0:
            raise ValueError('empty conditioning set')

        cpmf = CPMF(None, None)

        joint_IDs = self.IDs()
        joint_index = {var: joint_IDs.index(var) for var in joint_IDs}
        non_cond_vars = [ID for ID in joint_IDs if ID not in cond_vars]

        for joint_key, joint_p in self.items():
            cond_key = tuple(joint_key[joint_index[cvID]] for cvID in cond_vars)
            if len(cond_key) == 1:
                cond_key = cond_key[0]

            non_cond_key = tuple([joint_key[joint_index[ncvID]] for ncvID in non_cond_vars])

            try:
                pmf = cpmf.conditional_probabilities[cond_key]
            except KeyError:
                pmf = PMF(None)
                cpmf.conditional_probabilities[cond_key] = pmf

            try:
                pmf.probabilities[non_cond_key] = joint_p / cond_pmf.p(cond_key)
            except ZeroDivisionError:
                pass

        return cpmf


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

    def __init__(self, variable, given, initpmf=True):
        if initpmf:
            super().__init__(variable)
        else:
            self.tolerance_pdiff = 1e-10
            self.variable = variable
        self.probabilities = None
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
                pmf = conditional_counts[cv]
            except KeyError:
                pmf = PMF(None)
                conditional_counts[cv] = pmf
            # Avoiding the call to pmf.count_instance() for efficiency,
            # inlining it here instead.
            try:
                pmf.value_counts[v] += 1
            except KeyError:
                pmf.value_counts[v] = 1
            pmf.total_count += 1
        return conditional_counts


    def normalize_conditional_counts(self):
        for cv, pmf in self.conditional_probabilities.items():
            pmf.normalize_counts(update_probabilities=True)


    def __eq__(self, other: object) -> bool:
        assert isinstance(other, CPMF)
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



class OmegaPMF(PMF):

    def __init__(self):
        super().__init__(None)
        self.probabilities[1] = 1.0



class OmegaCPMF(CPMF):

    def __init__(self, pmf: PMF):
        super().__init__(None, None)
        self.conditional_probabilities[1] = pmf



def make_cpmf_PrXYcZ(
    X: int,
    Y: int,
    Z: list[int],
    PrXYZ: PMF,
    PrZ: PMF,
) -> CPMF:
    joint_variables = [X, Y] + Z
    index = {var: joint_variables.index(var) for var in joint_variables}

    PrXYcZ = CPMF(None, None)

    for joint_key, joint_p in PrXYZ.items():
        zkey = tuple([joint_key[index[zvar]] for zvar in Z])
        varkey = tuple([joint_key[index[var]] for var in joint_variables if var not in Z])
        if len(zkey) == 1:
            zkey = zkey[0]
        try:
            pmf = PrXYcZ.conditional_probabilities[zkey]
        except KeyError:
            pmf = PMF(None)
            PrXYcZ.conditional_probabilities[zkey] = pmf
        try:
            pmf.probabilities[varkey] = joint_p / PrZ.p(zkey)
        except ZeroDivisionError:
            pass

    return PrXYcZ



def make_cpmf_PrXcZ(
    X: int,
    Z: list[int],
    PrXZ: PMF,
    PrZ: PMF,
) -> CPMF:
    joint_variables = [X] + Z
    index = {var: joint_variables.index(var) for var in joint_variables}

    PrXcZ = CPMF(None, None)

    for joint_key, joint_p in PrXZ.items():
        zkey = tuple([joint_key[index[zvar]] for zvar in Z])
        varkey = [joint_key[index[var]] for var in joint_variables if var not in Z][0]
        if len(zkey) == 1:
            zkey = zkey[0]
        try:
            pmf = PrXcZ.conditional_probabilities[zkey]
        except KeyError:
            pmf = PMF(None)
            PrXcZ.conditional_probabilities[zkey] = pmf
        try:
            pmf.probabilities[varkey] = joint_p / PrZ.p(zkey)
        except ZeroDivisionError:
            pass

    return PrXcZ



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
    import mbtk.utilities.colors as col

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
    import mbtk.utilities.colors as col

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
