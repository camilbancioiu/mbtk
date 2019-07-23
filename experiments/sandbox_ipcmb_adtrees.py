import sys
import time
import pickle
import os
from pathlib import Path
from string import Template
import gc

import itertools


# Assume that the 'experiments' folder, which contains this file, is directly
# near the 'mbff' package.
EXPERIMENTS_ROOT = Path(os.getcwd())
MBFF_PATH = EXPERIMENTS_ROOT.parents[0]
sys.path.insert(0, str(MBFF_PATH))

import mbff.utilities.functions as util


# Elements of the experiment

EXDS_REPO = None
EXPRUN_REPO = None

ExdsDef = None
ExperimentDef = None
AlgorithmRunConfiguration = None
AlgorithmRunParameters = None


# Create the Experimental Dataset Definition

from mbff.dataset.ExperimentalDatasetDefinition import ExperimentalDatasetDefinition
from mbff.dataset.ExperimentalDataset import ExperimentalDataset
from mbff.dataset.sources.SampledBayesianNetworkDatasetSource import SampledBayesianNetworkDatasetSource
import mbff.math.Variable

EXDS_REPO = EXPERIMENTS_ROOT / 'exds_repository'

ExDsDefinition_ALARM_med2 = ExperimentalDatasetDefinition(EXDS_REPO, 'synthetic_alarm_8e4')
ExDsDefinition_ALARM_med2.exds_class = ExperimentalDataset
ExDsDefinition_ALARM_med2.source = SampledBayesianNetworkDatasetSource
ExDsDefinition_ALARM_med2.source_configuration = {
    'sourcepath': EXPERIMENTS_ROOT / 'bif_repository' / 'alarm.bif',
    'sample_count': int(8e4),
    'random_seed': 128,
}


ExdsDef = ExDsDefinition_ALARM_med2
Omega = mbff.math.Variable.Omega(ExdsDef.source_configuration['sample_count'])


# Create the Experiment Definition

from mbff.experiment.ExperimentDefinition import ExperimentDefinition
from mbff.experiment.ExperimentRun import ExperimentRun
from mbff.experiment.AlgorithmRun import AlgorithmRun
from mbff.experiment.AlgorithmRunDatapoint import AlgorithmRunDatapoint
from mbff.algorithms.mb.ipcmb import AlgorithmIPCMB


def make_algorithm_run_label(parameters):
    if parameters['ci_test_class'] is mbff.math.DSeparationCITest.DSeparationCITest:
        return Template('run_${algorithm_run_index}_T${target}__dsep')
    if parameters['ci_test_class'] is mbff.math.G_test__unoptimized.G_test:
        return Template('run_${algorithm_run_index}_T${target}__unoptimized')
    if parameters['ci_test_class'] is mbff.math.G_test__with_AD_tree.G_test:
        return Template('run_${algorithm_run_index}_T${target}__@LLT=${ci_test_ad_tree_leaf_list_threshold}')
    if parameters['ci_test_class'] is mbff.math.G_test__with_dcMI.G_test:
        return Template('run_${algorithm_run_index}_T${target}__dcMI')


EXPRUN_REPO = EXPERIMENTS_ROOT / 'exprun_repository'

IPCMB_ADTree_LLT_Eval_Definition = ExperimentDefinition(EXPRUN_REPO, 'IPCMB_ADTree_LLT_Eval')
IPCMB_ADTree_LLT_Eval_Definition.experiment_run_class = ExperimentRun
IPCMB_ADTree_LLT_Eval_Definition.algorithm_run_class = AlgorithmRun
IPCMB_ADTree_LLT_Eval_Definition.algorithm_run_datapoint_class = AlgorithmRunDatapoint
IPCMB_ADTree_LLT_Eval_Definition.exds_definition = ExdsDef
IPCMB_ADTree_LLT_Eval_Definition.save_algorithm_run_datapoints = True
IPCMB_ADTree_LLT_Eval_Definition.algorithm_run_log__stdout = True
IPCMB_ADTree_LLT_Eval_Definition.algorithm_run_log__file = True

AlgorithmRunConfiguration = {
    'label': make_algorithm_run_label,
    'algorithm': AlgorithmIPCMB
}

ExperimentDef = IPCMB_ADTree_LLT_Eval_Definition
ExperimentDef.algorithm_run_configuration = AlgorithmRunConfiguration


# Create AlgorithmRun parameters

import mbff.math.DSeparationCITest
import mbff.math.G_test__with_AD_tree
import mbff.math.G_test__unoptimized
import mbff.math.G_test__with_dcMI

ADTree_repo = ExperimentDef.path / 'adtrees'
ADTree_repo.mkdir(parents=True, exist_ok=True)
CITestResult_repo = ExperimentDef.path / 'ci_test_results'
CITestResult_repo.mkdir(parents=True, exist_ok=True)
BayesianNetwork = util.read_bif_file(ExdsDef.source_configuration['sourcepath'])
BayesianNetwork.finalize()
CITest_Significance = 0.95

DefaultParameters = {
    'target': 3,
    'debug': False,
    'omega': Omega,
    'source_bayesian_network': BayesianNetwork,
    'algorithm_debug': 1,
    'ci_test_debug': 1,
    'ci_test_significance': CITest_Significance,
}

Parameters_direct_d_separation_ci_test = [
    {
        'ci_test_class': mbff.math.DSeparationCITest.DSeparationCITest,
        'ci_test_results_path__save': CITestResult_repo / 'ci_test_results_{}_T3_dsep.pickle'.format(ExdsDef.name)
    }
]

Parameters_unoptimized = [
    {
        'ci_test_class': mbff.math.G_test__unoptimized.G_test,
        'ci_test_results_path__save': CITestResult_repo / 'ci_test_results_{}_T3_unoptimized.pickle'.format(ExdsDef.name)
    }
]

Parameters_with_dcMI = [
    {
        'ci_test_class': mbff.math.G_test__with_dcMI.G_test,
        'ci_test_results_path__save': CITestResult_repo / 'ci_test_results_{}_T3_dcMI.pickle'.format(ExdsDef.name)
    }
]

Parameters_with_AD_tree = [
    {
        'ci_test_class': mbff.math.G_test__with_AD_tree.G_test,
        'ci_test_ad_tree_leaf_list_threshold': LLT,
        'ci_test_ad_tree_path__load': ADTree_repo / 'adtree_{}_llt{}.pickle'.format(ExdsDef.name, LLT),
        'ci_test_ad_tree_path__save': ADTree_repo / 'adtree_{}_llt{}.pickle'.format(ExdsDef.name, LLT),
        'ci_test_results_path__save': CITestResult_repo / 'ci_test_results_{}_T3_ADtree_LLT{}.pickle'.format(ExdsDef.name, LLT)
    }
    for LLT in [4096, 8192, 2048, 1024]
]

AlgorithmRunParameters = [] \
    + Parameters_direct_d_separation_ci_test \
    + Parameters_with_AD_tree \
    + Parameters_unoptimized \
    + Parameters_with_dcMI \
    + []

for parameters in AlgorithmRunParameters:
    parameters.update(DefaultParameters)

ExperimentDef.algorithm_run_parameters = AlgorithmRunParameters


# Command-line interface

def command_list_algrun(arguments):
    specific_key = None
    try:
        specific_key = arguments[0]
    except:
        pass

    from pprint import pprint
    for index, parameters in enumerate(AlgorithmRunParameters):
        print('AlgorithmRun parameters:', index)
        if specific_key is not None:
            try:
                print('{}: {}'.format(specific_key, parameters[specific_key]))
            except KeyError:
                print('{}: {}'.format(specific_key, 'not found'))
        else:
            pprint(parameters)
        print()
    print('Total of', len(AlgorithmRunParameters), 'AlgorithmRun parameters')



def command_list_algrun_datapoints(arguments):
    datapoints_folder = ExperimentDef.subfolder('algorithm_run_datapoints')
    datapoint_files = list(datapoints_folder.iterdir())

    try:
        datapoint_files_to_list = [datapoint_files[arguments[0]]]
    except:
        datapoint_files_to_list = datapoint_files

    from pprint import pprint

    for datapoint_file in datapoint_files_to_list:
        with datapoint_file.open('rb') as f:
            datapoint = pickle.load(f)
        pprint(datapoint)



def command_build_adtree(arguments):
    exds = ExdsDef.create_exds()
    if ExdsDef.exds_ready():
        exds.load()
    else:
        exds.build()
    try:
        specific_algrun_parameters_index = int(arguments[0])
        algrun_parameters_to_build_trees_for = [AlgorithmRunParameters[specific_algrun_parameters_index]]
    except:
        algrun_parameters_to_build_trees_for = Parameters_with_AD_tree

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



def command_build_adtree_analysis(arguments):
    try:
        specific_algrun_parameters_index = int(arguments[0])
        algrun_parameters_to_analyze_trees_for = [AlgorithmRunParameters[specific_algrun_parameters_index]]
    except:
        algrun_parameters_to_analyze_trees_for = Parameters_with_AD_tree

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



def command_plot(arguments):
    import matplotlib.pyplot as Plotter

    PlotPath = ExperimentDef.path / 'plots'
    PlotPath.mkdir(parents=True, exist_ok=True)

    citr = load_citr()
    data = dict()

    for key, results in citr.items():
        print(key, len(results))

    try:
        plot_what = arguments[0]
    except:
        plot_what = 'duration'

    if plot_what == 'duration':
        for key, results in citr.items():
            data[key] = [result.duration for result in results]
    if plot_what == 'duration-cummulative':
        for key, results in citr.items():
            durations = [result.duration for result in results]
            durations_cummulative = list(itertools.accumulate(durations))
            data[key] = durations_cummulative

    Xaxis = list(range(len(citr['unoptimized'])))

    Plotter.figure(figsize=(10, 6))
    # Plotter.clf()
    # Plotter.cla()
    Plotter.rcParams.update({'font.size': 20})
    # pos = Plotter.gca().get_position()
    Plotter.gca().tick_params(axis='both', which='major', pad=8)
    Plotter.gca().margins(0.01)
    Plotter.xlabel('CI Test number')
    Plotter.ylabel('Time (s)')

    runs = ['unoptimized', 'dcMI', 'adtree_8192', 'adtree_4096', 'adtree_2048', 'adtree_1024']
    # runs = ['unoptimized', 'dcMI', 'adtree_1024']
    for run in runs:
        Plotter.plot(Xaxis, data[run], lw=1.5)

    legend = list()
    legend.append('unoptimized')
    legend.append('dcMI')

    from humanize import naturalsize, naturaldelta
    from datetime import timedelta
    treekeys = ['adtree_8192', 'adtree_4096', 'adtree_2048', 'adtree_1024']
    treedata = load_adtrees_analysis(Parameters_with_AD_tree)
    for key in treekeys:
        data = treedata[key]
        entry = '{} ({size}, {nodes} nodes, {duration} to build)'.format(
            key,
            size=naturalsize(data['size']),
            nodes=data['nodes'],
            duration=naturaldelta(timedelta(seconds=data['duration'])))
        legend.append(entry)

    Plotter.legend(legend)
    # Plotter.yscale('log')
    Plotter.grid(True)
    Plotter.title('CI test times')
    Plotter.tight_layout()
    Plotter.show()
    # Plotter.savefig(plotpath + '/ks_iteration_durations__{}__T{}_K{}.png'.format(definition.name, Tj, K))



def load_citr():
    citr = dict()
    with Parameters_unoptimized[0]['ci_test_results_path__save'].open('rb') as f:
        citr['unoptimized'] = pickle.load(f)

    with Parameters_with_dcMI[0]['ci_test_results_path__save'].open('rb') as f:
        citr['dcMI'] = pickle.load(f)

    for parameters in Parameters_with_AD_tree:
        key = 'adtree_{}'.format(parameters['ci_test_ad_tree_leaf_list_threshold'])
        with parameters['ci_test_results_path__save'].open('rb') as f:
            citr[key] = pickle.load(f)

    return citr



def load_adtrees_analysis(algrun_parameters):
    adtrees_analysis = dict()

    analysis_path = ExperimentDef.path / 'adtree_analysis'

    for parameters in algrun_parameters:
        key = 'adtree_{}'.format(parameters['ci_test_ad_tree_leaf_list_threshold'])
        tree_analysis_path = analysis_path / (parameters['ci_test_ad_tree_path__save'].name)
        with tree_analysis_path.open('rb') as f:
            analysis = pickle.load(f)

        adtrees_analysis[key] = analysis

    return adtrees_analysis



def command_run_experiment(arguments):
    Experiment = ExperimentDef.create_experiment_run()
    Experiment.run()


if __name__ == '__main__':
    command = sys.argv[1]

    if command == 'list-algrun':
        command_list_algrun(sys.argv[2:])

    if command == 'list-algrun-datapoints':
        command_list_algrun_datapoints(sys.argv[2:])

    if command == 'build-adtree':
        command_build_adtree(sys.argv[2:])

    if command == 'build-adtree-analysis':
        command_build_adtree_analysis(sys.argv[2:])

    if command == 'run-experiment':
        command_run_experiment(sys.argv[2:])

    if command == 'plot':
        command_plot(sys.argv[2:])
