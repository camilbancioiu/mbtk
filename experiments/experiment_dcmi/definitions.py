import sys
import os
from pathlib import Path


# Assume that the 'experiments' folder, which contains this file, is directly
# near the 'mbff' package.
EXPERIMENTS_ROOT = Path(os.getcwd())
MBFF_PATH = EXPERIMENTS_ROOT.parents[0]
sys.path.insert(0, str(MBFF_PATH))


################################################################################
# Create the Experimental Dataset Definition

from mbff.dataset.ExperimentalDatasetDefinition import ExperimentalDatasetDefinition
from mbff.dataset.ExperimentalDataset import ExperimentalDataset
from mbff.dataset.sources.SampledBayesianNetworkDatasetSource import SampledBayesianNetworkDatasetSource
import mbff.math.Variable


def exds_definition(experimental_setup, sample_count_string):
    """Argument sample_count_string should be a positive integer in exponential
    notation, e.g. 4e4 or 2e5"""
    exds_name = 'synthetic_alarm_{}'.format(sample_count_string)
    sample_count = int(float(sample_count_string))

    exdsDef = ExperimentalDatasetDefinition(experimental_setup.Paths.ExDsRepository, exds_name)
    exdsDef.exds_class = ExperimentalDataset
    exdsDef.source = SampledBayesianNetworkDatasetSource
    exdsDef.source_configuration = {
        'sourcepath': experimental_setup.Paths.BIFRepository / 'alarm.bif',
        'sample_count': sample_count,
        'random_seed': 81628965211 + sample_count
    }

    return exdsDef



def omega(exdsDef):
    omega = mbff.math.Variable.Omega(exdsDef.source_configuration['sample_count'])
    return omega


################################################################################
# Create the Experiment Definition

from mbff.experiment.ExperimentDefinition import ExperimentDefinition
from mbff.experiment.ExperimentRun import ExperimentRun
from mbff.experiment.AlgorithmRun import AlgorithmRun
from mbff.experiment.AlgorithmRunDatapoint import AlgorithmRunDatapoint
from mbff.algorithms.mb.ipcmb import AlgorithmIPCMB


def experiment_definition(experimental_setup):
    experimentDef = ExperimentDefinition(experimental_setup.Paths.ExpRunRepository, 'dcMIEvExp')
    experimentDef.experiment_run_class = ExperimentRun
    experimentDef.algorithm_run_class = AlgorithmRun
    experimentDef.algorithm_run_configuration = {
        'label': make_algorithm_run_label,
        'algorithm': AlgorithmIPCMB
    }
    experimentDef.algorithm_run_datapoint_class = AlgorithmRunDatapoint
    experimentDef.exds_definition = experimental_setup.ExDsDef
    experimentDef.save_algorithm_run_datapoints = True
    experimentDef.algorithm_run_log__stdout = True
    experimentDef.algorithm_run_log__file = True

    return experimentDef


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
