import collections
import pickle
import operator
import math


def create_citr_analysis(algruns):
    analysis = dict()

    total_citr_count = 0
    for algrun in algruns:
        target = algrun['target']
        algrun_analysis = create_analysis(algrun)
        analysis.update(algrun_analysis)
        total_citr_count += algrun_analysis[f'T{target}_citr_count']

    analysis['Total CI count:'] = total_citr_count
    return analysis


def create_analysis(algrun):
    analysis = dict()
    target = algrun['target']
    results = load_citr_from_algrun_parameters(algrun)

    analysis[f'T{target}_citr_count'] = len(results)

    histogram = create_condset_size_histogram(results)
    analysis[f'T{target}_condset_size_histogram'] = histogram
    return analysis


def create_condset_size_histogram(results):
    condset = operator.attrgetter('Z')
    condsets = map(condset, results)
    cond_counter = collections.Counter(map(len, condsets))

    histogram = '\n'
    for size in sorted(cond_counter.keys()):
        count = cond_counter[size]
        histogram += f'\t\t{size:2}: {count:4} '
        histogram += '-' * int(math.log2(count))
        histogram += '\n'

    return histogram


def load_citr_from_algrun_parameters(parameters):
    path = parameters['ci_test_results_path__save']
    with path.open('rb') as f:
        results = pickle.load(f)
    return results
