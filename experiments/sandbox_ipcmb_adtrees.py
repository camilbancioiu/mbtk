import sys
import os
from pathlib import Path
from string import Template


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


EXPRUN_REPO = EXPERIMENTS_ROOT / 'exprun_repository'

IPCMB_ADTree_LLT_Eval_Definition = ExperimentDefinition(EXPRUN_REPO, 'IPCMB_ADTree_LLT_Eval')
IPCMB_ADTree_LLT_Eval_Definition.experiment_run_class = ExperimentRun
IPCMB_ADTree_LLT_Eval_Definition.algorithm_run_class = AlgorithmRun
IPCMB_ADTree_LLT_Eval_Definition.algorithm_run_datapoint_class = AlgorithmRunDatapoint
IPCMB_ADTree_LLT_Eval_Definition.exds_definition = ExdsDef
IPCMB_ADTree_LLT_Eval_Definition.save_algorithm_run_datapoints = True
IPCMB_ADTree_LLT_Eval_Definition.algorithm_run_log__stdout = True
IPCMB_ADTree_LLT_Eval_Definition.algorithm_run_log__file = True

ExperimentDef = IPCMB_ADTree_LLT_Eval_Definition


# Create AlgorithmRun parameters

import mbff.math.DSeparationCITest
import mbff.math.G_test__with_AD_tree
import mbff.math.G_test__unoptimized
import mbff.math.G_test__with_dcMI


def make_algorithm_run_label(parameters):
    if parameters['ci_test_class'] is mbff.math.DSeparationCITest.DSeparationCITest:
        return Template('run_${algorithm_run_index}_T${target}__dsep')
    if parameters['ci_test_class'] is mbff.math.G_test__unoptimized.G_test:
        return Template('run_${algorithm_run_index}_T${target}__unoptimized')
    if parameters['ci_test_class'] is mbff.math.G_test__with_AD_tree.G_test:
        return Template('run_${algorithm_run_index}_T${target}__@LLT=${ci_test_ad_tree_leaf_list_threshold}')
    if parameters['ci_test_class'] is mbff.math.G_test__with_dcMI.G_test:
        return Template('run_${algorithm_run_index}_T${target}__dcMI')


AlgorithmRunConfiguration = {
    'label': make_algorithm_run_label,
    'algorithm': AlgorithmIPCMB
}
ExperimentDef.algorithm_run_configuration = AlgorithmRunConfiguration

ADTree_repo = ExperimentDef.path / 'adtrees'
ADTree_repo.mkdir(parents=True, exist_ok=True)
CITestResult_repo = ExperimentDef.path / 'ci_test_results'
CITestResult_repo.mkdir(parents=True, exist_ok=True)
BayesianNetwork = util.read_bif_file(ExdsDef.source_configuration['sourcepath'])
BayesianNetwork.finalize()
CITest_Significance = 0.95


DefaultParameters = {
    'target': 3,
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

# Quick and dirty experiment on multiple targets
Targets = [0, 1, 2, 3, 4]
del DefaultParameters['target']
Parameters_with_dcMI = [
    {
        'target': target,
        'ci_test_jht_path__save': ExperimentDef.path / 'JHT.pickle',
        'ci_test_jht_path__load': ExperimentDef.path / 'JHT.pickle',
        'ci_test_class': mbff.math.G_test__with_dcMI.G_test,
        'ci_test_results_path__save': CITestResult_repo / 'ci_test_results_{}_T{}_dcMI.pickle'.format(ExdsDef.name, target)
    }
    for target in Targets
]
LLT = 1024
Parameters_with_AD_tree = [
    {
        'target': target,
        'ci_test_class': mbff.math.G_test__with_AD_tree.G_test,
        'ci_test_ad_tree_leaf_list_threshold': LLT,
        'ci_test_ad_tree_path__load': ADTree_repo / 'adtree_{}_llt{}.pickle'.format(ExdsDef.name, LLT),
        'ci_test_ad_tree_path__save': ADTree_repo / 'adtree_{}_llt{}.pickle'.format(ExdsDef.name, LLT),
        'ci_test_results_path__save': CITestResult_repo / 'ci_test_results_{}_T{}_ADtree_LLT{}.pickle'.format(ExdsDef.name, target, LLT)
    }
    for target in Targets
]
AlgorithmRunParameters = [] \
    + Parameters_with_AD_tree \
    + Parameters_with_dcMI \
    + []


for parameters in AlgorithmRunParameters:
    parameters.update(DefaultParameters)



if __name__ == '__main__':
    command = sys.argv[1]
    arguments = sys.argv[2:]

    import mbff.utilities.experiment as utilcli
    from mbff.utilities.Exceptions import CLICommandNotHandled

    try:
        utilcli.handle_common_commands(command, arguments, ExperimentDef, ExdsDef, AlgorithmRunParameters)
    except CLICommandNotHandled:
        pass

    import experiment_dcmi_main_commands as custom_commands

    if command == 'build-adtree':
        custom_commands.command_build_adtree(arguments, ExdsDef, AlgorithmRunParameters)
    elif command == 'build-adtree-analysis':
        custom_commands.command_build_adtree_analysis(arguments, ExperimentDef, AlgorithmRunParameters)
    elif command == 'plot':
        custom_commands.command_plot(arguments, ExperimentDef, AlgorithmRunParameters)
    else:
        raise CLICommandNotHandled(command)
