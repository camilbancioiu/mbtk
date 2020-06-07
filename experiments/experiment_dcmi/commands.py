import time
import gc
import pickle
from pprint import pprint
from operator import attrgetter

import mbff.math.G_test__with_AD_tree


def configure_objects_subparser__adtree(subparsers):
    subparser = subparsers.add_parser('adtree')
    subparser.add_argument('verb',
                           choices=['build', 'analyze', 'print-analysis',
                                    'test-load', 'update'],
                           default='print-analysis', nargs='?')
    subparser.add_argument('--tree-type',
                           choices=['static', 'dynamic'],
                           type=str, action='store')



def configure_objects_subparser__plot(subparsers):
    subparser = subparsers.add_parser('plot')
    subparser.add_argument('verb', choices=['create'],
                           default='create', nargs='?')
    subparser.add_argument('--metric', type=str, nargs='?',
                           default='duration-cummulative')
    subparser.add_argument('--file', type=str, nargs='?', default=None)
    subparser.add_argument('tags', type=str)



def configure_objects_subparser__summary(subparsers):
    subparser = subparsers.add_parser('summary')
    subparser.add_argument('verb', choices=['create'], default='create',
                           nargs='?')
    subparser.add_argument('tags', type=str, default=None, nargs='?')



def handle_command(arguments, experimental_setup):
    command_handled = False
    command_object = arguments.object
    command_verb = arguments.verb

    if command_object == 'adtree':
        if command_verb == 'build':
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



def command_adtree_build(experimental_setup):
    exds = experimental_setup.ExDsDef.create_exds()
    if experimental_setup.ExDsDef.exds_ready():
        exds.load()
    else:
        exds.build()

    tree_type = experimental_setup.Arguments.tree_type
    adtree_save_path = experimental_setup.get_ADTree_path()
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

    try:
        adtree = ADTreeClass(matrix, column_values, experimental_setup.LLT, debug=2)
    except TypeError:
        adtree = ADTreeClass(matrix, column_values, experimental_setup.LLT)

    duration = time.time() - start_time
    print("AD-tree ({}) with LLT={} built in {:>10.4f}s".format(
        tree_type, experimental_setup.LLTArgument, duration))

    if adtree_save_path is not None:
        with adtree_save_path.open('wb') as f:
            pickle.dump(adtree, f)
    print("AD-tree saved to", adtree_save_path)



def command_adtree_build_analysis(experimental_setup):
    from pympler.asizeof import asizeof

    analysis_path = experimental_setup.ExperimentDef.path / 'adtree_analysis'
    analysis_path.mkdir(parents=True, exist_ok=True)

    tree_type = experimental_setup.Arguments.tree_type
    tree_path = experimental_setup.get_ADTree_path()

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

    tree_path = experimental_setup.get_ADTree_path()
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
    # Sample count is already selected
    # group datapoints by tag, or by default tags
    # summary per group:
    # * number of datapoints
    # * total duration
    # * total CI duration
    # * AD-tree analysis, for AD-tree runs
    #       - size in MB
    #       - node count
    #       - LLT absolute count
    # * JHT analysis, for dcMI runs
    #       - size in MB
    #       - entry count
    #       - hit rate
    from humanize import naturalsize
    from pympler.asizeof import asizeof
    from statistics import median

    tags = experimental_setup.Arguments.tags
    if tags is None:
        tags = experimental_setup.DefaultTags
    else:
        tags = tags.split(',')

    adtree_analysis = load_adtrees_analysis(experimental_setup)

    for tag in tags:
        algruns = experimental_setup.get_algruns_by_tag(tag)
        datapoints = load_datapoints(experimental_setup, algruns)
        citr = load_citr_for_algrun_list(algruns)
        ci_durations = list(map(attrgetter('duration'), citr))
        total_duration = sum(map(attrgetter('duration'), datapoints))
        total_ci_duration = sum(ci_durations)
        print('tag \'{}\':'.format(tag))
        print('\tDatapoints:', len(datapoints))
        print('\tTotal CI count:', len(citr))
        print('\tTotal duration (s):', total_duration)
        print('\tTotal CI duration (s):', total_ci_duration)
        print('\tAvg. CI duration (s):', total_ci_duration / len(citr))
        print('\tMax. CI duration (s):', max(ci_durations))
        print('\tMedian CI duration (s):', median(ci_durations))

        if tag.startswith('adtree'):
            analysis = adtree_analysis[tag]
            print('\tAD-tree size:', naturalsize(analysis['size']))
            print('\tAD-tree nodes:', analysis['nodes'])
            print('\tAD-tree absolute LLT:', analysis['LLT'])
            print('\tAD-tree build time:', analysis['duration'])

        if tag == 'dcmi':
            jht = load_jht(experimental_setup)
            size = asizeof(jht)
            entries = len(jht) - 2
            reads = jht['reads']
            misses = jht['misses']
            hits = reads - misses
            hitrate = hits / reads
            print('\tJHT size:', naturalsize(size))
            print('\tJHT entries:', entries)
            print('\tJHT reads:', reads)
            print('\tJHT hits:', hits)
            print('\tJHT misses:', misses)
            print('\tJHT hit rate:', hitrate)



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
    citr = list()
    for algrun in algruns:
        results = load_citr_from_algrun_parameters(algrun)
        citr.extend(results)
    return citr



def load_adtrees_analysis(experimental_setup):
    adtrees_analysis = dict()

    analysis_path = experimental_setup.ExperimentDef.path / 'adtree_analysis'

    for parameters in experimental_setup.AlgorithmRunParameters:
        try:
            tree_analysis_path = analysis_path / (parameters['ci_test_ad_tree_path__load'].name)
        except KeyError:
            continue

        try:
            with tree_analysis_path.open('rb') as f:
                analysis = pickle.load(f)
            key = 'adtree-llt{}'.format(parameters['ci_test_ad_tree_llt_argument'])
            adtrees_analysis[key] = analysis
        except FileNotFoundError:
            continue

    return adtrees_analysis



def load_jht(experimental_setup):
    # Only one JHT is expected for an experimental setup, so we return the
    # first we find.
    jht = None
    for parameters in experimental_setup.AlgorithmRunParameters:
        try:
            jht_path = parameters['ci_test_jht_path__load']
        except KeyError:
            continue

        with jht_path.open('rb') as f:
            jht = pickle.load(f)
        return jht

    return jht



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
    with parameters['ci_test_results_path__save'].open('rb') as f:
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
