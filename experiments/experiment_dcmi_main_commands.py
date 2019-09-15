import time
import pickle
import gc

import mbff.math.G_test__with_AD_tree


def command_build_adtree(arguments, ExdsDef, AlgorithmRunParameters):
    exds = ExdsDef.create_exds()
    if ExdsDef.exds_ready():
        exds.load()
    else:
        exds.build()
    try:
        specific_algrun_parameters_index = int(arguments[0])
        algrun_parameters_to_build_trees_for = [AlgorithmRunParameters[specific_algrun_parameters_index]]
    except IndexError:
        algrun_parameters_to_build_trees_for = filter_algrun_parameters_Gtest_ADtree(AlgorithmRunParameters)

    for parameters in algrun_parameters_to_build_trees_for:
        LLT = parameters['ci_test_ad_tree_leaf_list_threshold']
        matrix = exds.matrix.X
        column_values = exds.matrix.get_values_per_column('X')
        start_time = time.time()
        adtree = mbff.structures.ADTree.ADTree(matrix, column_values, LLT, debug=3)
        duration = time.time() - start_time
        print("AD-tree with LLT={} built in {:>10.4f}s".format(LLT, duration))

        adtree_save_path = parameters.get('ci_test_ad_tree_path__save', None)
        if adtree_save_path is not None:
            with adtree_save_path.open('wb') as f:
                pickle.dump(adtree, f)
        print("AD-tree saved to", adtree_save_path)



def command_build_adtree_analysis(arguments, ExperimentDef, AlgorithmRunParameters):
    try:
        specific_algrun_parameters_index = int(arguments[0])
        specific_algrun_parameters = AlgorithmRunParameters[specific_algrun_parameters_index]
        algrun_parameters_to_analyze_trees_for = [specific_algrun_parameters]
    except IndexError:
        algrun_parameters_to_analyze_trees_for = filter_algrun_parameters_Gtest_ADtree(AlgorithmRunParameters)

    from pympler.asizeof import asizeof
    from pprint import pprint

    analysis_path = ExperimentDef.path / 'adtree_analysis'
    analysis_path.mkdir(parents=True, exist_ok=True)

    for parameters in algrun_parameters_to_analyze_trees_for:
        tree_analysis_path = analysis_path / (parameters['ci_test_ad_tree_path__save'].name)
        with parameters['ci_test_ad_tree_path__save'].open('rb') as f:
            print('Loading tree')
            pprint(parameters)
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



def command_plot(arguments, ExperimentDef, AlgorithmRunParameters):

    try:
        plot_what = arguments[0]
    except IndexError:
        plot_what = 'duration'

    try:
        plot_save_filename = arguments[1]
        PlotPath = ExperimentDef.path / 'plots'
        PlotPath.mkdir(parents=True, exist_ok=True)
        plot_save_filename = PlotPath / (plot_save_filename + '.png')
    except IndexError:
        plot_save_filename = None

    citr = load_citr(AlgorithmRunParameters)

    algruns_Gtest_ADtree = filter_algrun_parameters_Gtest_ADtree(AlgorithmRunParameters)
    adtree_analysis = load_adtrees_analysis(algruns_Gtest_ADtree, ExperimentDef)

    import experiment_dcmi_main_plotting as plotting
    data = plotting.make_plot_data(plot_what, citr)
    plotting.plot(data, adtree_analysis, plot_save_filename)



def load_citr(AlgorithmRunParameters):
    citr = dict()

    # Concatenate the CI test results for all AlgorithmRuns with the unoptimized G-test
    algruns_Gtest_unoptimized = filter_algrun_parameters_Gtest_unoptimized(AlgorithmRunParameters)
    citr['unoptimized'] = list(map(load_citr_from_algrun_parameters, algruns_Gtest_unoptimized))

    # Concatenate the CI test results for all AlgorithmRuns with dcMI
    algruns_Gtest_dcMI = filter_algrun_parameters_Gtest_dcMI(AlgorithmRunParameters)
    algruns_Gtest_dcMI = list(algruns_Gtest_dcMI)
    citr['dcmi'] = list()
    for parameters in algruns_Gtest_dcMI:
        loaded_citr_per_algrun = load_citr_from_algrun_parameters(parameters)
        citr['dcmi'].extend(loaded_citr_per_algrun)

    # Concatenate the CI test results for all AlgorithmRuns with AD-trees, but
    # separated by the LLT value of the AD-trees
    algruns_Gtest_ADtree = filter_algrun_parameters_Gtest_ADtree(AlgorithmRunParameters)
    for parameters in algruns_Gtest_ADtree:
        LLT = parameters['ci_test_ad_tree_leaf_list_threshold']
        key = 'adtree_{}'.format(LLT)
        loaded_citr_per_algrun = load_citr_from_algrun_parameters(parameters)
        try:
            citr[key].extend(loaded_citr_per_algrun)
        except KeyError:
            citr[key] = list(loaded_citr_per_algrun)

    return citr



def load_adtrees_analysis(algrun_parameters, ExperimentDef):
    adtrees_analysis = dict()

    analysis_path = ExperimentDef.path / 'adtree_analysis'

    for parameters in algrun_parameters:
        key = 'adtree_{}'.format(parameters['ci_test_ad_tree_leaf_list_threshold'])
        tree_analysis_path = analysis_path / (parameters['ci_test_ad_tree_path__save'].name)
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
