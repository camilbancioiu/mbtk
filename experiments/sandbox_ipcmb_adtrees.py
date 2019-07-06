import sys
import os
from pathlib import Path
from string import Template

# Assume that the 'experiments' folder, which contains this file, is directly
# near the 'mbff' package.
EXPERIMENTS_ROOT = Path(os.getcwd())
MBFF_PATH = EXPERIMENTS_ROOT.parents[0]
sys.path.insert(0, str(MBFF_PATH))

from mbff.dataset.ExperimentalDatasetDefinition import ExperimentalDatasetDefinition
from mbff.dataset.ExperimentalDataset import ExperimentalDataset
from mbff.dataset.sources.SampledBayesianNetworkDatasetSource import SampledBayesianNetworkDatasetSource
from mbff.math.Variable import Omega

import mbff.math.G_test__with_AD_tree
import mbff.math.G_test__unoptimized
import mbff.math.G_test__with_dcMI

EXDS_REPO = EXPERIMENTS_ROOT / 'exds_repository'

ExDsDefinition_ALARM_med2 = ExperimentalDatasetDefinition(EXDS_REPO, 'synthetic_alarm_8e4')
ExDsDefinition_ALARM_med2.exds_class = ExperimentalDataset
ExDsDefinition_ALARM_med2.source = SampledBayesianNetworkDatasetSource
ExDsDefinition_ALARM_med2.source_configuration = {
    'sourcepath': EXPERIMENTS_ROOT / 'bif_repository' / 'alarm.bif',
    'sample_count': int(8e4),
    'random_seed': 128,
}


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
IPCMB_ADTree_LLT_Eval_Definition.exds_definition = ExDsDefinition_ALARM_med2
IPCMB_ADTree_LLT_Eval_Definition.save_algorithm_run_datapoints = True
IPCMB_ADTree_LLT_Eval_Definition.algorithm_run_log__stdout = True
IPCMB_ADTree_LLT_Eval_Definition.algorithm_run_log__file = True
IPCMB_ADTree_LLT_Eval_Definition.algorithm_run_configuration = {
    'label': Template('run_${algorithm_run_index}_T${target}__@LLT=${ci_test_ad_tree_leaf_list_threshold}'),
    'algorithm': AlgorithmIPCMB
}

if __name__ == '__main__':
    exdsDef = ExDsDefinition_ALARM_med2

    omega = Omega(exdsDef.source_configuration['sample_count'])

    adtree_folder = Path('adtrees')
    adtree_folder.mkdir(parents=True, exist_ok=True)

    ci_test_results_folder = Path('ci_test_results')
    ci_test_results_folder.mkdir(parents=True, exist_ok=True)

    parameters_unoptimized = [
        {
            'target': 3,
            'debug': False,
            'omega': omega,
            'ci_test_class': mbff.math.G_test__unoptimized.G_test,
            'ci_test_significance': 0.95,
            'ci_test_debug': True,
            'ci_test_results_path__save': ci_test_results_folder / 'ci_test_results_{}_T3_unoptimized.pickle'.format(exdsDef.name)
        }
    ]
    parameters_with_dcMI = [
        {
            'target': 3,
            'debug': False,
            'omega': omega,
            'ci_test_class': mbff.math.G_test__with_dcMI.G_test,
            'ci_test_significance': 0.95,
            'ci_test_debug': True,
            'ci_test_results_path__save': ci_test_results_folder / 'ci_test_results_{}_T3_dcMI.pickle'.format(exdsDef.name)
        }
    ]

    parameters_with_AD_tree = [
        {
            'target': 3,
            'debug': False,
            'omega': omega,
            'ci_test_class': mbff.math.G_test__with_AD_tree.G_test,
            'ci_test_significance': 0.95,
            'ci_test_debug': True,
            'ci_test_ad_tree_leaf_list_threshold': LLT,
            'ci_test_ad_tree_path__load': adtree_folder / 'adtree_{}_llt{}.pickle'.format(exdsDef.name, LLT),
            'ci_test_ad_tree_path__save': adtree_folder / 'adtree_{}_llt{}.pickle'.format(exdsDef.name, LLT),
            'ci_test_results_path__save': ci_test_results_folder / 'ci_test_results_{}_T3_ADtree_LLT{}.pickle'.format(exdsDef.name, LLT)
        }
        for LLT in [16384, 4096, 8192, 2048, 1024]
    ]

    IPCMB_ADTree_LLT_Eval_Definition.algorithm_run_parameters = [] \
        + parameters_with_dcMI \
        + parameters_unoptimized \
        + parameters_with_AD_tree

    IPCMB_ADTree_LLT_Eval = IPCMB_ADTree_LLT_Eval_Definition.create_experiment_run()
    IPCMB_ADTree_LLT_Eval.run()
