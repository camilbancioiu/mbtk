import time
import gc
import pickle
from pprint import pprint

import mbff.math.G_test__with_AD_tree


def configure_objects_subparser__adtree(subparsers):
    subparser = subparsers.add_parser('adtree')
    subparser.add_argument('verb', choices=['build', 'analyze', 'print-analysis', 'test-load', 'update'],
                           default='print-analysis', nargs='?')


def configure_objects_subparser__plot(subparsers):
    subparser = subparsers.add_parser('plot')
    subparser.add_argument('verb', choices=['create'],
                           default='create', nargs='?')
    subparser.add_argument('--metric', type=str, nargs='?', default='duration-cummulative')
    subparser.add_argument('--file', type=str, nargs='?', default=None)


def handle_command(experimental_setup):
    command_handled = False
    command_object = experimental_setup.Arguments.object
    command_verb = experimental_setup.Arguments.verb

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

    return command_handled


def command_adtree_build(experimental_setup):
    exds = experimental_setup.ExDsDef.create_exds()
    if experimental_setup.ExDsDef.exds_ready():
        exds.load()
    else:
        exds.build()

    matrix = exds.matrix.X
    column_values = exds.matrix.get_values_per_column('X')
    start_time = time.time()
    adtree = mbff.structures.ADTree.ADTree(matrix, column_values, experimental_setup.LLT, debug=2)
    duration = time.time() - start_time
    print("AD-tree with LLT={} built in {:>10.4f}s".format(experimental_setup.LLT, duration))

    adtree_save_path = experimental_setup.Paths.ADTree
    if adtree_save_path is not None:
        with adtree_save_path.open('wb') as f:
            pickle.dump(adtree, f)
    print("AD-tree saved to", adtree_save_path)



def command_adtree_build_analysis(experimental_setup):
    from pympler.asizeof import asizeof

    analysis_path = experimental_setup.ExperimentDef.path / 'adtree_analysis'
    analysis_path.mkdir(parents=True, exist_ok=True)

    tree_analysis_path = analysis_path / (experimental_setup.Paths.ADTree.name)
    with experimental_setup.Paths.ADTree.open('rb') as f:
        adtree = pickle.load(f)

    adtree.matrix = None
    adtree.column_values = None
    gc.collect()
    tree_size = asizeof(adtree)
    analysis = {
        'LLT': adtree.leaf_list_threshold,
        'nodes': adtree.ad_node_count + adtree.vary_node_count,
        'duration': adtree.duration,
        'size': tree_size
    }
    print()
    print('Analysis')
    pprint(analysis)
    with tree_analysis_path.open('wb') as f:
        pickle.dump(analysis, f)
    print()



def command_adtree_print_analysis(experimental_setup):
    analysis_path = experimental_setup.ExperimentDef.path / 'adtree_analysis'

    tree_analysis_path = analysis_path / (experimental_setup.Paths.ADTree.name)
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
    plot_what = experimental_setup.Arguments.metric
    plot_save_filename = experimental_setup.Arguments.file

    if plot_save_filename is not None:
        plot_path = experimental_setup.ExperimentDef.path / 'plots'
        plot_path.mkdir(parents=True, exist_ok=True)
        plot_save_filename = plot_path / (plot_save_filename + '.png')

    citr = load_citr(experimental_setup)

    algruns_Gtest_ADtree = experimental_setup.get_algruns_by_tag('adtree')
    adtree_analysis = load_adtrees_analysis(algruns_Gtest_ADtree, experimental_setup.ExperimentDef)

    import plotting
    data = plotting.make_plot_data(plot_what, citr)
    plotting.plot(data, adtree_analysis, plot_save_filename)



def load_citr(experimental_setup):
    citr = dict()

    # Concatenate the CI test results for all AlgorithmRuns with the unoptimized G-test
    algruns_Gtest_unoptimized = experimental_setup.get_algruns_by_tag('unoptimized')
    citr['unoptimized'] = list()
    for parameters in algruns_Gtest_unoptimized:
        results = load_citr_from_algrun_parameters(parameters)
        citr['unoptimized'].extend(results)


    # Concatenate the CI test results for all AlgorithmRuns with dcMI
    algruns_Gtest_dcMI = experimental_setup.get_algruns_by_tag('dcmi')
    citr['dcmi'] = list()
    for parameters in algruns_Gtest_dcMI:
        results = load_citr_from_algrun_parameters(parameters)
        citr['dcmi'].extend(results)

    # Concatenate the CI test results for all AlgorithmRuns with AD-trees, but
    # separated by the LLT value of the AD-trees
    algruns_Gtest_ADtree = experimental_setup.get_algruns_by_tag('adtree')
    for parameters in algruns_Gtest_ADtree:
        LLT = parameters['ci_test_ad_tree_leaf_list_threshold']
        key = 'adtree_{}'.format(LLT)
        loaded_citr_per_algrun = load_citr_from_algrun_parameters(parameters)
        try:
            citr[key].extend(loaded_citr_per_algrun)
        except KeyError:
            citr[key] = loaded_citr_per_algrun

    return citr



def load_adtrees_analysis(algrun_parameters, ExperimentDef):
    adtrees_analysis = dict()

    analysis_path = ExperimentDef.path / 'adtree_analysis'

    for parameters in algrun_parameters:
        key = 'adtree_{}'.format(parameters['ci_test_ad_tree_leaf_list_threshold'])
        tree_analysis_path = analysis_path / (parameters['ci_test_ad_tree_path__load'].name)
        with tree_analysis_path.open('rb') as f:
            analysis = pickle.load(f)

        adtrees_analysis[key] = analysis

    return adtrees_analysis



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
