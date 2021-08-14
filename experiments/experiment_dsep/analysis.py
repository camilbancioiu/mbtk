import collections
import pickle
import operator
import math


def create_citr_analysis(experimental_setup):
    algruns = experimental_setup.algorithm_run_parameters
    analysis = dict()

    total_citr_count = 0
    total_mb_errors = 0
    for algrun in algruns:
        target = algrun['target']

        citr_analysis = create_algrun_citr_analysis(algrun)
        analysis.update(citr_analysis)

        mb_analysis = create_algrun_mb_analysis(experimental_setup, algrun)
        analysis.update(mb_analysis)

        total_citr_count += citr_analysis[f'T{target}_citr_count']
        total_mb_errors += mb_analysis[f'T{target}_mb_error_count']

    analysis['Total CI count:'] = total_citr_count
    analysis['Total MB errors:'] = total_mb_errors
    return analysis



def create_algrun_citr_analysis(algrun):
    analysis = dict()
    target = algrun['target']
    results = load_citr_from_algrun(algrun)

    analysis[f'T{target}_citr_count'] = len(results)

    histogram = create_condset_size_histogram(results)
    analysis[f'T{target}_condset_size_histogram'] = histogram
    return analysis



def create_algrun_mb_analysis(experimental_setup, algrun):
    analysis = dict()
    target = algrun['target']

    source_type = experimental_setup.source_type
    datapoints_folder = experimental_setup.paths.Datapoints

    mb = load_mb_from_algrun(datapoints_folder, algrun)
    analysis[f'T{target}_mb_size'] = len(mb)

    if source_type == 'ds':
        bn_datapoints_folder = experimental_setup.make_bn_datapoints_path()
        bn_mb = load_mb_from_algrun(bn_datapoints_folder, algrun)
        mb_error = set(mb).symmetric_difference(set(bn_mb))
        analysis[f'T{target}_mb_error_count'] = len(mb_error)

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



def load_citr_from_algrun(algrun):
    path = algrun['ci_test_results_path__save']
    with path.open('rb') as f:
        results = pickle.load(f)
    return results



def load_mb_from_algrun(datapoints_folder, algrun):
    ID = algrun['ID']
    path = datapoints_folder / f'{ID}.pickle'
    with path.open('rb') as f:
        datapoint = pickle.load(f)
    return datapoint.mb
