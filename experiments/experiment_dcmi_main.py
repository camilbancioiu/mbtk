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



################################################################################
# Elements of the experiment

EXDS_REPO = None
EXPRUN_REPO = None

ExdsDef = None
ExperimentDef = None
AlgorithmRunConfiguration = None
AlgorithmRunParameters = None



################################################################################
# Create the Experimental Dataset Definition

from mbff.dataset.ExperimentalDatasetDefinition import ExperimentalDatasetDefinition
from mbff.dataset.ExperimentalDataset import ExperimentalDataset
from mbff.dataset.sources.SampledBayesianNetworkDatasetSource import SampledBayesianNetworkDatasetSource
import mbff.math.Variable

EXDS_REPO = EXPERIMENTS_ROOT / 'exds_repository'

ExdsDef = ExperimentalDatasetDefinition(EXDS_REPO, 'synthetic_alarm_4e4')
ExdsDef.exds_class = ExperimentalDataset
ExdsDef.source = SampledBayesianNetworkDatasetSource
ExdsDef.source_configuration = {
    'sourcepath': EXPERIMENTS_ROOT / 'bif_repository' / 'alarm.bif',
    'sample_count': int(4e4),
    'random_seed': 81628965211,
}

Omega = mbff.math.Variable.Omega(ExdsDef.source_configuration['sample_count'])



################################################################################
# Create the Experiment Definition

from mbff.experiment.ExperimentDefinition import ExperimentDefinition
from mbff.experiment.ExperimentRun import ExperimentRun
from mbff.experiment.AlgorithmRun import AlgorithmRun
from mbff.experiment.AlgorithmRunDatapoint import AlgorithmRunDatapoint
from mbff.algorithms.mb.ipcmb import AlgorithmIPCMB

EXPRUN_REPO = EXPERIMENTS_ROOT / 'exprun_repository'

ExperimentDef = ExperimentDefinition(EXPRUN_REPO, 'dcMI_vs_ADtree_in_IPCMB')
ExperimentDef.experiment_run_class = ExperimentRun
ExperimentDef.algorithm_run_class = AlgorithmRun
ExperimentDef.algorithm_run_datapoint_class = AlgorithmRunDatapoint
ExperimentDef.exds_definition = ExdsDef
ExperimentDef.save_algorithm_run_datapoints = True
ExperimentDef.algorithm_run_log__stdout = True
ExperimentDef.algorithm_run_log__file = True



################################################################################
# Create AlgorithmRun parameters

import mbff.math.DSeparationCITest
import mbff.math.G_test__with_AD_tree
import mbff.math.G_test__unoptimized
import mbff.math.G_test__with_dcMI


# This function will be called by the ExperimentRun object to give each
# AlgorithmRun a unique label, based on what CI test class was configured for
# IPC-MB during the run
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
JHT_repo = ExperimentDef.path / 'jht'
CITestResult_repo = ExperimentDef.path / 'ci_test_results'

ADTree_repo.mkdir(parents=True, exist_ok=True)
JHT_repo.mkdir(parents=True, exist_ok=True)
CITestResult_repo.mkdir(parents=True, exist_ok=True)

BayesianNetwork = util.read_bif_file(ExdsDef.source_configuration['sourcepath'])
BayesianNetwork.finalize()

CITest_Significance = 0.95
LLT = 0

DefaultParameters = {
    'omega': Omega,
    'source_bayesian_network': BayesianNetwork,
    'algorithm_debug': 1,
    'ci_test_debug': 1,
    'ci_test_significance': CITest_Significance,
}

# Create AlgorithmRun parameters using the D-separation CI test
Parameters_DSep = list()
for target in range(len(BayesianNetwork)):
    parameters = {
        'target': target,
        'ci_test_class': mbff.math.DSeparationCITest.DSeparationCITest,
        'ci_test_results_path__save': CITestResult_repo / 'ci_test_results_{}_T{}_dsep.pickle'.format(ExdsDef.name, target)
    }
    Parameters_DSep.append(parameters)

# Create AlgorithmRun parameters using the unoptimized G-test
Parameters_Gtest_unoptimized = list()
for target in range(len(BayesianNetwork)):
    parameters = {
        'target': target,
        'ci_test_class': mbff.math.G_test__unoptimized.G_test,
        'ci_test_results_path__save': CITestResult_repo / 'ci_test_results_{}_T{}_unoptimized.pickle'.format(ExdsDef.name, target)
    }
    Parameters_Gtest_unoptimized.append(parameters)

# Create AlgorithmRun parameters using the G-test optimized with an AD-tree @LLT=0
Parameters_Gtest_ADtree = list()
for target in range(len(BayesianNetwork)):
    parameters = {
        'target': target,
        'ci_test_class': mbff.math.G_test__with_AD_tree.G_test,
        'ci_test_ad_tree_leaf_list_threshold': LLT,
        'ci_test_ad_tree_path__load': ADTree_repo / 'adtree_{}_llt{}.pickle'.format(ExdsDef.name, LLT),
        'ci_test_ad_tree_path__save': ADTree_repo / 'adtree_{}_llt{}.pickle'.format(ExdsDef.name, LLT),
        'ci_test_results_path__save': CITestResult_repo / 'ci_test_results_{}_T{}_ADtree_LLT{}.pickle'.format(ExdsDef.name, target, LLT)
    }
    Parameters_Gtest_ADtree.append(parameters)

# Create AlgorithmRun parameters using the G-test optimized with dcMI
Parameters_Gtest_dcMI = list()
for target in range(len(BayesianNetwork)):
    parameters = {
        'target': target,
        'ci_test_class': mbff.math.G_test__with_dcMI.G_test,
        'ci_test_jht_path__load': JHT_repo / 'jht_{}.pickle'.format(ExdsDef.name),
        'ci_test_jht_path__save': JHT_repo / 'jht_{}.pickle'.format(ExdsDef.name),
        'ci_test_results_path__save': CITestResult_repo / 'ci_test_results_{}_T{}_dcMI.pickle'.format(ExdsDef.name, target)
    }
    Parameters_Gtest_dcMI.append(parameters)


# Concatenate the lists of all AlgorithmRun parameters defined above
AlgorithmRunParameters = [] \
    + Parameters_DSep \
    + Parameters_Gtest_unoptimized \
    + Parameters_Gtest_ADtree \
    + Parameters_Gtest_dcMI


# Apply defaults to all AlgorithmRun parameters
for parameters in AlgorithmRunParameters:
    parameters.update(DefaultParameters)



################################################################################
# Command-line interface
if __name__ == '__main__':
    command = sys.argv[1]
    arguments = sys.argv[2:]

    import experiment_dcmi_main_commands as custom_commands
    from mbff.utilities.Exceptions import CLICommandNotHandled

    try:
        if command == 'build-adtree':
            custom_commands.command_build_adtree(arguments, ExdsDef, AlgorithmRunParameters)
        elif command == 'build-adtree-analysis':
            custom_commands.command_build_adtree_analysis(arguments, ExperimentDef, AlgorithmRunParameters)
        elif command == 'plot':
            custom_commands.command_plot(arguments, ExperimentDef, AlgorithmRunParameters)
        else:
            raise CLICommandNotHandled(command)
    except CLICommandNotHandled:
        import mbff.utilities.experiment as utilcli
        utilcli.handle_common_commands(command, arguments, ExperimentDef, ExdsDef, AlgorithmRunParameters)
