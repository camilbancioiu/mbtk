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


def exds_definition(experimental_setup):
    """Argument sample_count_string should be a positive integer in exponential
    notation, e.g. 4e4 or 2e5"""
    sample_count_string = experimental_setup.Arguments.sample_count
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
    experiment_name = 'dcMIEvExp_{}'.format(experimental_setup.Arguments.sample_count)
    experimentDef = ExperimentDefinition(experimental_setup.Paths.ExpRunRepository, experiment_name)
    experimentDef.experiment_run_class = ExperimentRun
    experimentDef.algorithm_run_class = AlgorithmRun
    experimentDef.algorithm_run_configuration = {
        'algorithm': AlgorithmIPCMB
    }
    experimentDef.algorithm_run_datapoint_class = AlgorithmRunDatapoint
    experimentDef.exds_definition = experimental_setup.ExDsDef
    experimentDef.save_algorithm_run_datapoints = True
    experimentDef.algorithm_run_log__stdout = True
    experimentDef.algorithm_run_log__file = True

    return experimentDef
