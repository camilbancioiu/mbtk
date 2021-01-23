import sys
from collections import Counter, defaultdict
from itertools import combinations
from operator import attrgetter
from pprint import pprint
from argparse import ArgumentParser
import math


def make_features(count):
    A = ord('A')
    return set([chr(i + A) for i in range(count)])



class Test:

    def __init__(self, x, y, z):
        self.X = x
        self.Y = y

        if self.X > self.Y:
            self.X = y
            self.Y = x

        self.Z = set(z)

        self.XZ = set(self.Z | {self.X})
        self.YZ = set(self.Z | {self.Y})
        self.XYZ = set(self.Z | {self.X} | {self.Y})

    def __str__(self):
        z = ' '.join(sorted(self.Z))
        Hxz = '(' + ', '.join(sorted(self.XZ)) + ')'
        Hyz = '(' + ', '.join(sorted(self.YZ)) + ')'
        Hxyz = '(' + ', '.join(sorted(self.XYZ)) + ')'
        Hz = '(' + ', '.join(sorted(self.Z)) + ')'
        return '{} ⊥ {} | {} \t=> {} + {} - {} - {}'.format(self.X, self.Y, z, Hxz, Hyz, Hxyz, Hz)


    def as_bits(self, features):
        x = as_bits(features, list(self.X))
        y = as_bits(features, list(self.Y))
        z = as_bits(features, sorted(self.Z))
        Hxz = as_bits(features, sorted(self.XZ))
        Hyz = as_bits(features, sorted(self.YZ))
        Hxyz = as_bits(features, sorted(self.XYZ))
        Hz = as_bits(features, sorted(self.Z))
        return '{} ⊥ {} | {} \t=> {} + {} - {} - {}'.format(x, y, z, Hxz, Hyz, Hxyz, Hz)


    def joint_entropy_terms(self):
        return map(frozenset, [self.XZ, self.YZ, self.XYZ, self.Z])


def as_bits(features, subset):
    bits = list()

    for feature in features:
        if feature in subset:
            bits.append('1')
        else:
            bits.append('0')

    bits.reverse()
    return ''.join(bits)


def count_tests_of_cond_set_size(tests, cond_set_size):
    count = 0
    for test in tests:
        if len(test.Z) == cond_set_size:
            count += 1
    return count


def cond_set_sizes(features):
    return range(len(features) - 1)


def iter_tests(features):
    for cond_set_size in cond_set_sizes(features):
        for Z in sorted(combinations(features, cond_set_size)):
            test_pairs = features - set(Z)
            for X, Y in sorted(combinations(test_pairs, 2)):
                yield Test(X, Y, Z)


def make_jht_index_summary(jht_index):
    index_summary = dict()

    for jht, tests in jht_index.items():
        tests_cond_sizes = defaultdict(int)
        for test in tests:
            tests_cond_sizes[len(test.Z)] += 1

        index_summary[jht] = tests_cond_sizes

    return index_summary


def validate_and_flatten_summary(index_summary):
    flat_summary = dict()

    for jht, test_cond_sizes in index_summary.items():
        try:
            existing_cond_sizes = flat_summary[len(jht)]
            assert existing_cond_sizes == test_cond_sizes
        except KeyError:
            flat_summary[len(jht)] = test_cond_sizes

    return flat_summary


def expected_jht_count_by_cond_size(M, jht_size):
    expected_jht_counts = dict()

    c = jht_size
    expected_jht_counts[c] = math.comb(M - jht_size, 2)

    if jht_size >= 1:
        c = jht_size - 1
        expected_jht_counts[c] = jht_size * (M - jht_size)

    if jht_size >= 2:
        c = jht_size - 2
        expected_jht_counts[c] = math.comb(jht_size, 2)

    return expected_jht_counts


def expected_total_counts_for_jht_size(M, j):
    return M * (M - 1) / 2


def get_cli_arguments():
    parser = ArgumentParser()

    parser.add_argument('features_count', metavar='M', type=int)
    parser.add_argument('options', type=str, nargs='*')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_cli_arguments()
    M = args.features_count
    options = args.options


    enum_tests = 'enum' in options
    expand_jht_index = 'index' in options
    show_jht_index_summary = 'index-summary' in options
    show_flat_jht_index_summary = 'flat-index-summary' in options

    features = make_features(M)
    all_tests = list()
    all_tests_by_features = defaultdict(list)
    jht_counter = Counter()
    jht_index = defaultdict(list)

    for test in iter_tests(features):
        all_tests.append(test)
        all_tests_by_features[frozenset((test.X, test.Y))].append(test)

        jhts = list(test.joint_entropy_terms())
        jht_counter.update(jhts)

        for jht in jhts:
            jht_index[jht].append(test)
        if enum_tests:
            print(test)

    print()
    print(len(all_tests), 'tests for', M, 'features')

    expected_tests = (1 / 2) * M * (M - 1) * (2**(M - 2))
    print('expected tests', expected_tests)
    print()

    for cond_set_size in cond_set_sizes(features):
        count = count_tests_of_cond_set_size(all_tests, cond_set_size)
        print(count, 'tests of cond_set_size =', cond_set_size)

    print()
    if expand_jht_index:
        for jht, count in sorted(jht_counter.items()):
            tests = jht_index[jht]
            jht = '(' + ', '.join(sorted(jht)) + ')'
            count = '{:>4}'.format(count)
            print(count, '\t', jht)
            for test in tests:
                print('\t', test)
            print()

    if show_jht_index_summary:
        index_summary = make_jht_index_summary(jht_index)
        for jht, tests_cond_sizes in index_summary.items():
            jht = '(' + ', '.join(sorted(jht)) + ')'
            print(jht)
            for cond_size, count in tests_cond_sizes.items():
                print('\ttests of cond_size {}: {}'.format(cond_size, count))
            print()

    if show_flat_jht_index_summary:
        index_summary = make_jht_index_summary(jht_index)
        flat_summary = validate_and_flatten_summary(index_summary)
        for jht_size, tests_cond_sizes in sorted(flat_summary.items()):
            expected_jht_counts = expected_jht_count_by_cond_size(M, jht_size)
            expected_total = expected_total_counts_for_jht_size(M, jht_size)
            total = '({} vs {})'.format(sum(tests_cond_sizes.values()), expected_total)
            print(jht_size, ':', total)
            for cond_size, count in tests_cond_sizes.items():
                expected_count = expected_jht_counts[cond_size]
                print('\ttests of cond_size {}: {} vs {}'.format(cond_size, count, expected_count))
            print()
