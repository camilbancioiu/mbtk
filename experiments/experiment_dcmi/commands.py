import time
import gc
import pickle
from pprint import pprint
from operator import attrgetter

from humanize import naturalsize
from pympler.asizeof import asizeof
from statistics import median

import mbff.math.G_test__with_AD_tree


def configure_objects_subparser__adtree(subparsers, expsetup):
    subparser = subparsers.add_parser('adtree')
    subparser.add_argument('verb',
                           choices=['show', 'build', 'analyze', 'print-analysis',
                                    'test-load', 'update'],
                           default='show', nargs='?')
    subparser.add_argument('--tree-type',
                           choices=expsetup.AllowedADTreeTypes,
                           type=str, action='store', default=None)
    subparser.add_argument('--llt',
                           choices=expsetup.AllowedLLT,
                           type=str, action='store', default=None)



def configure_objects_subparser__plot(subparsers):
    subparser = subparsers.add_parser('plot')
    subparser.add_argument('verb', choices=['create'],
                           default='create', nargs='?')
    subparser.add_argument('--metric', type=str, nargs='?',
                           choices=['duration', 'duration-cummulative'],
                           default='duration-cummulative')
    subparser.add_argument('--file', type=str, nargs='?', default=None)
    subparser.add_argument('tags', type=str)



def configure_objects_subparser__summary(subparsers):
    subparser = subparsers.add_parser('summary')
    subparser.add_argument('verb', choices=['create'], default='create',
                           nargs='?')
    subparser.add_argument('tags', type=str, default=None, nargs='?')
    subparser.add_argument('--refresh', action='store_true', default=False)



def handle_command(arguments, experimental_setup):
    command_handled = False
    command_object = arguments.object
    command_verb = arguments.verb

    if command_object == 'adtree':
        if command_verb == 'show':
            command_adtree_show(experimental_setup)
        elif command_verb == 'build':
            command_adtree_build(experimental_setup)
            command_handled = True
        elif command_verb == 'analyze':
            command_adtree_build_analysis(experimental_setup)
            command_handled = True
        elif command_verb == 'print-analysis':
            command_adtree_print_analysis(experimental_setup)
            command_handled = True
        elif command_verb == 'test-load':
            command_adtree_test_load(experimental_setup)
            command_handled = True
        elif command_verb == 'update':
            command_adtree_update(experimental_setup)
            command_handled = True

    if command_object == 'plot':
        if command_verb == 'create':
            command_plot_create(experimental_setup)
            command_handled = True

    if command_object == 'summary':
        if command_verb == 'create':
            command_summary_create(experimental_setup)
            command_handled = True

    return command_handled



def command_adtree_show(experimental_setup):
    adtree_repo = experimental_setup.Paths.ADTreeRepository
    for adtree_file in adtree_repo.iterdir():
        print(adtree_file.name)



def command_adtree_build(experimental_setup):
    exds = experimental_setup.ExDsDef.create_exds()
    if experimental_setup.ExDsDef.exds_ready():
        exds.load()
    else:
        exds.build()

    tree_type = experimental_setup.Arguments.tree_type
    if tree_type is None:
        raise ValueError('AD-tree type must be provided (one of {})'.format(
            experimental_setup.AllowedADTreeTypes))
    llt = experimental_setup.Arguments.llt
    if llt is None:
        raise ValueError('LLT must be provided (one of {})'.format(
            experimental_setup.AllowedLLT))
    absolute_llt = experimental_setup.calculate_absolute_LLT(llt)

    adtree_save_path = experimental_setup.get_ADTree_path(tree_type, llt)
    ADTreeClass = None
    if tree_type == 'static':
        ADTreeClass = mbff.structures.ADTree.ADTree
    elif tree_type == 'dynamic':
        ADTreeClass = mbff.structures.DynamicADTree.DynamicADTree
    else:
        raise ValueError('AD-tree type may be either static or dynamic.')

    print('Building {} AD-tree...'.format(tree_type))
    matrix = exds.matrix.X
    column_values = exds.matrix.get_values_per_column('X')
    start_time = time.time()

    adtree = ADTreeClass(matrix, column_values, absolute_llt)

    duration = time.time() - start_time
    print("AD-tree ({}) with LLT={} built in {:>10.4f}s".format(
        tree_type, llt, duration))

    if adtree_save_path is not None:
        with adtree_save_path.open('wb') as f:
            pickle.dump(adtree, f)
    print("AD-tree saved to", adtree_save_path)



def command_adtree_build_analysis(experimental_setup):
    from pympler.asizeof import asizeof

    analysis_path = experimental_setup.ExperimentDef.path / 'adtree_analysis'
    analysis_path.mkdir(parents=True, exist_ok=True)

    tree_type = experimental_setup.Arguments.tree_type
    llt = experimental_setup.Arguments.llt
    tree_path = experimental_setup.get_ADTree_path(tree_type, llt)

    tree_analysis_path = analysis_path / tree_path.name
    with tree_path.open('rb') as f:
        adtree = pickle.load(f)

    adtree.matrix = None
    adtree.column_values = None
    gc.collect()
    tree_size = asizeof(adtree)
    analysis = {
        'LLT': adtree.leaf_list_threshold,
        'nodes': adtree.ad_node_count + adtree.vary_node_count,
        'duration': adtree.duration,
        'size': tree_size,
        'type': tree_type,
    }
    print()
    print('Analysis')
    pprint(analysis)
    with tree_analysis_path.open('wb') as f:
        pickle.dump(analysis, f)
    print()



def command_adtree_print_analysis(experimental_setup):
    analysis_path = experimental_setup.ExperimentDef.path / 'adtree_analysis'

    tree_type = experimental_setup.Arguments.tree_type
    llt = experimental_setup.Arguments.llt
    tree_path = experimental_setup.get_ADTree_path(tree_type, llt)
    tree_analysis_path = analysis_path / tree_path.name
    with tree_analysis_path.open('rb') as f:
        analysis = pickle.load(f)
    print('Analysis')
    pprint(analysis)



def command_adtree_test_load(experimental_setup):
    print('Loading AD-tree from {}...'.format(experimental_setup.Paths.ADTree))
    start_time = time.time()

    adtree = load_adtree(experimental_setup.Paths.ADTree)

    adtree_node_count = adtree.ad_node_count + adtree.vary_node_count
    duration = time.time() - start_time
    print('AD-tree loaded in {:.2f}s. It contains {} nodes, ({} AD; {} Vary).'
          .format(duration, adtree_node_count, adtree.ad_node_count, adtree.vary_node_count))

    input('Press <ENTER> to run GC after loading the AD-tree...')
    gc.collect()

    input('Press <ENTER> to run GC after setting adtree = None...')
    adtree = None
    del adtree
    gc.collect()

    input('Press <ENTER> to exit...')



def command_adtree_update(experimental_setup):
    path = experimental_setup.Paths.ADTree
    print('Loading AD-tree from {}...'.format(path))
    adtree = load_adtree(path)
    print('AD-tree loaded, running GC...')
    gc.collect()
    print('Saving the AD-tree as updated to {}...'.format(path))
    with path.open('wb') as f:
        pickle.dump(adtree, f)
    print('Done')



def load_adtree(path):
    adtree = None

    with path.open('rb') as f:
        unpickler = pickle.Unpickler(f)
        adtree = unpickler.load()

    del unpickler

    return adtree



def command_plot_create(experimental_setup):
    metric = experimental_setup.Arguments.metric
    plot_save_filename = experimental_setup.Arguments.file

    if plot_save_filename is not None:
        plot_path = experimental_setup.ExperimentDef.path / 'plots'
        plot_path.mkdir(parents=True, exist_ok=True)
        plot_save_filename = plot_path / (plot_save_filename + '.png')

    citr = load_citr(experimental_setup)
    adtree_analysis = load_adtrees_analysis(experimental_setup)

    import plotting
    data = plotting.make_plot_data(metric, citr)
    plotting.plot(data, adtree_analysis, plot_save_filename)



def command_summary_create(experimental_setup):
    tags = experimental_setup.Arguments.tags
    if tags is None:
        tags = experimental_setup.DefaultTags
    else:
        tags = tags.split(',')

    summaries = experimental_setup.ExperimentDef.path / 'summaries'
    summaries.mkdir(parents=True, exist_ok=True)

    adtree_analysis = load_adtrees_analysis(experimental_setup)

    for tag in tags:
        summary_path = summaries / '{}.pickle'.format(tag)
        summary = None
        cached = ''
        try:
            if experimental_setup.Arguments.refresh:
                raise FileNotFoundError
            with summary_path.open('rb') as f:
                summary = pickle.load(f)
            cached = ' (cached)'
        except FileNotFoundError:
            summary = create_summary(experimental_setup, tag, adtree_analysis)
            cached = ''
            with summary_path.open('wb') as f:
                pickle.dump(summary, f)

        print()
        print('tag \'{}\'{}:'.format(tag, cached))

        for key, value in summary.items():
            print('\t' + key, value)



def create_summary(experimental_setup, tag, adtree_analysis):
    summary = dict()

    # Test early whether the AD-tree analysis is both required and available
    if tag.startswith('adtree'):
        analysis = adtree_analysis[tag]

    algruns = list(experimental_setup.get_algruns_by_tag(tag))
    summary['Runs:'] = len(algruns)

    citr = list()
    try:
        citr = list(load_citr_for_algrun_list(algruns))
    except FileNotFoundError:
        pass

    if len(citr) == 0:
        summary['Error'] = 'No CI tests found'
        return summary

    datapoints = load_datapoints(experimental_setup, algruns)
    ci_durations = list(map(attrgetter('duration'), citr))
    total_duration = sum(map(attrgetter('duration'), datapoints))
    total_ci_duration = sum(ci_durations)

    summary['Datapoints:'] = len(datapoints)
    summary['Total CI count:'] = len(citr)
    summary['Total duration (s):'] = total_duration
    summary['Total CI duration (s):'] = total_ci_duration
    summary['Avg. CI duration (s):'] = total_ci_duration / len(citr)
    summary['Avg. CI count per s:'] = len(citr) / total_ci_duration
    summary['Max. CI duration (s):'] = max(ci_durations)
    summary['Median CI duration (s):'] = median(ci_durations)

    if tag.startswith('adtree'):
        analysis = adtree_analysis[tag]
        summary['AD-tree size:'] = naturalsize(analysis['size'])
        summary['AD-tree nodes:'] = analysis['nodes']
        summary['AD-tree absolute LLT:'] = analysis['LLT']
        summary['AD-tree build time:'] = analysis['duration']

    if tag == 'dcmi':
        jht = load_jht(experimental_setup)
        size = asizeof(jht)
        reads = jht['reads']
        misses = jht['misses']
        del jht['reads']
        del jht['misses']
        entries = len(jht)
        hits = reads - misses
        hitrate = hits / reads
        max_key_size = max(map(len, jht.keys()))
        summary['JHT size:'] = naturalsize(size)
        summary['JHT entries:'] = entries
        summary['JHT reads:'] = reads
        summary['JHT hits:'] = hits
        summary['JHT misses:'] = misses
        summary['JHT hit rate:'] = hitrate
        summary['JHT max key size:'] = max_key_size

        dof_cache = load_dof_cache(experimental_setup)
        size = asizeof(dof_cache)
        entries = len(dof_cache)
        max_key_size = max(map(len, dof_cache.keys()))
        summary['DoF cache size:'] = naturalsize(size)
        summary['DoF cache entries:'] = entries
        summary['DoF cache max key size:'] = max_key_size

    return summary



def load_datapoints(experimental_setup, algruns):
    datapoints = []

    datapoints_folder = experimental_setup.ExperimentDef.subfolder('algorithm_run_datapoints')
    for algrun in algruns:
        datapoint_file = datapoints_folder / '{}.pickle'.format(algrun['ID'])
        try:
            with datapoint_file.open('rb') as f:
                datapoint = pickle.load(f)
            datapoints.append(datapoint)
        except FileNotFoundError:
            continue

    return datapoints



def load_citr(experimental_setup):
    citr = dict()
    tags = experimental_setup.Arguments.tags.split(',')
    for tag in tags:
        algruns = experimental_setup.get_algruns_by_tag(tag)
        citr[tag] = load_citr_for_algrun_list(algruns)
    return citr



def load_citr_for_algrun_list(algruns):
    for algrun in algruns:
        results = load_citr_from_algrun_parameters(algrun)
        for result in results:
            yield result



def load_adtrees_analysis(experimental_setup):
    analysis_path = experimental_setup.ExperimentDef.path / 'adtree_analysis'

    adtrees_analysis = dict()
    for parameters in experimental_setup.AlgorithmRunParameters:
        try:
            tree_analysis_path = analysis_path / (parameters['ci_test_ad_tree_path__load'].name)
        except KeyError:
            continue

        try:
            with tree_analysis_path.open('rb') as f:
                analysis = pickle.load(f)
            llt = parameters['ci_test_ad_tree_llt_argument']
            tree_type = parameters['ci_test_ad_tree_type']
            key = 'adtree-{}-llt{}'.format(tree_type, llt)
            adtrees_analysis[key] = analysis
        except FileNotFoundError:
            continue

    return adtrees_analysis



def load_jht(experimental_setup):
    # Only one JHT is expected for an experimental setup, so we return the
    # first we find, when iterating over all AlgorithmRunParameters.
    jht = None
    for parameters in experimental_setup.AlgorithmRunParameters:
        try:
            jht_path = parameters['ci_test_jht_path__save']
        except KeyError:
            continue

        with jht_path.open('rb') as f:
            jht = pickle.load(f)
        return jht

    return jht



def load_dof_cache(experimental_setup):
    dof_cache = None
    # Only one DoF calculator cache is expected for an experimental setup, so
    # we return the first we find, when iterating over all
    # AlgorithmRunParameters.
    for parameters in experimental_setup.AlgorithmRunParameters:
        try:
            dof_cache_path = parameters['ci_test_dof_calculator_cache_path__save']
        except KeyError:
            continue

        with dof_cache_path.open('rb') as f:
            dof_cache = pickle.load(f)
        return dof_cache

    return dof_cache



def validate_all_citrs_equal(citr):
    reference = citr['unoptimized']
    for key in citr.keys():
        if key != 'unoptimized':
            for lcitr, rcitr in zip(reference, citr[key]):
                lcitr.tolerance__p_value = 1e-9
                lcitr.tolerance__statistic_value = 1e-8
                if lcitr != rcitr:
                    print(lcitr)
                    print(lcitr.p_value)
                    print(rcitr)
                    print(rcitr.p_value)
                    print(lcitr.diff(rcitr))
                    raise ValueError("CI test results for {} deviate from the reference results".format(key))



def load_citr_from_algrun_parameters(parameters):
    path = parameters['ci_test_results_path__save']
    with path.open('rb') as f:
        results = pickle.load(f)
    return results



def filter_algrun_parameters_Gtest_unoptimized(AlgorithmRunParameters):
    ci_test_class = mbff.math.G_test__unoptimized.G_test
    return filter_algurn_parameters_by_ci_test_class(AlgorithmRunParameters, ci_test_class)



def filter_algrun_parameters_Gtest_ADtree(AlgorithmRunParameters):
    ci_test_class = mbff.math.G_test__with_AD_tree.G_test
    return filter_algurn_parameters_by_ci_test_class(AlgorithmRunParameters, ci_test_class)



def filter_algrun_parameters_Gtest_dcMI(AlgorithmRunParameters):
    ci_test_class = mbff.math.G_test__with_dcMI.G_test
    return filter_algurn_parameters_by_ci_test_class(AlgorithmRunParameters, ci_test_class)



def filter_algurn_parameters_by_ci_test_class(AlgorithmRunParameters, ci_test_class):
    for parameters in AlgorithmRunParameters:
        if parameters['ci_test_class'] == ci_test_class:
            yield parameters
