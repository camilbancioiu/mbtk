import collections
import pickle
import operator
import math

from mbtk.math.CITestResult import CITestResult


def create_citr_analysis(experimental_setup):
    algruns = experimental_setup.algorithm_run_parameters
    analysis = dict()

    arguments = experimental_setup.arguments

    total_citr_count = 0
    total_accurate_citr_count = 0
    total_mb_errors = 0
    total_condset_size_counter = collections.Counter()
    for algrun in algruns:
        target = algrun['target']

        (citr_analysis, condset_size_counter) = create_algrun_citr_analysis(arguments, algrun)
        total_condset_size_counter.update(condset_size_counter)
        mb_analysis = create_algrun_mb_analysis(experimental_setup, algrun)

        if arguments.target_citr_analysis:
            analysis.update(citr_analysis)
        if arguments.target_mb_analysis:
            analysis.update(mb_analysis)

        total_citr_count += citr_analysis[f'T{target}_citr_count']
        total_accurate_citr_count += citr_analysis[f'T{target}_accurate_citr_count']
        try:
            total_mb_errors += mb_analysis[f'T{target}_mb_error_count']
        except KeyError:
            pass

    total_histogram = render_condset_size_histogram(total_condset_size_counter)
    total_accurate_citr_count_percentage = total_accurate_citr_count * 100 / total_citr_count
    if arguments.total_condset_histogram:
        analysis['Total condset size histogram'] = total_histogram
    analysis['Total CI count:'] = total_citr_count
    analysis['Total accurate CI count:'] = total_accurate_citr_count
    analysis['Total accurate CI count (%):'] = str(total_accurate_citr_count_percentage) + '%'
    analysis['Total MB errors:'] = total_mb_errors
    return analysis



def create_algrun_citr_analysis(arguments, algrun):
    analysis = dict()
    target = algrun['target']
    results = load_citr_from_algrun(algrun)

    accurate_count = sum(map(int, map(CITestResult.accurate, results)))
    analysis[f'T{target}_citr_count'] = len(results)
    analysis[f'T{target}_accurate_citr_count'] = accurate_count

    condset_size_counter = create_condset_size_counter(results)
    if arguments.target_condset_histogram:
        histogram = render_condset_size_histogram(condset_size_counter)
        analysis[f'T{target}_condset_size_histogram'] = histogram
    return (analysis, condset_size_counter)



def create_algrun_mb_analysis(experimental_setup, algrun):
    analysis = dict()
    target = algrun['target']

    source_type = experimental_setup.source_type
    datapoints_folder = experimental_setup.paths.Datapoints

    mb = load_mb_from_algrun(datapoints_folder, algrun['ID'])
    analysis[f'T{target}_mb_size'] = len(mb)

    if source_type == 'ds':
        bn_datapoints_folder = experimental_setup.make_bn_datapoints_path()
        algrunID = experimental_setup.create_algrun_ID(algrun, 'bn')
        algrunID = 'run_' + algrunID
        bn_mb = load_mb_from_algrun(bn_datapoints_folder, algrunID)
        mb_error = set(mb).symmetric_difference(set(bn_mb))
        analysis[f'T{target}_mb_error_count'] = len(mb_error)

    return analysis



def create_condset_size_counter(results):
    condset = operator.attrgetter('Z')
    condsets = map(condset, results)
    cond_counter = collections.Counter(map(len, condsets))

    return cond_counter



def render_condset_size_histogram(cond_counter):
    histogram = '\n'
    for size in sorted(cond_counter.keys()):
        count = cond_counter[size]
        histogram += f'\t\t{size:2}: {count:6} '
        histogram += '-' * int(math.log2(count))
        histogram += '\n'

    return histogram



def load_citr_from_algrun(algrun):
    path = algrun['ci_test_results_path__save']
    with path.open('rb') as f:
        results = pickle.load(f)
    return results



def load_mb_from_algrun(datapoints_folder, algrunID):
    path = datapoints_folder / f'{algrunID}.pickle'
    with path.open('rb') as f:
        datapoint = pickle.load(f)
    return datapoint.mb
