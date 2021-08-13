#!/usr/bin/env python3

import sys
import os
from pathlib import Path
import argparse

# Assume that the 'experiments' folder, which contains this file, is directly
# near the 'mbtk' package.
EXPERIMENTS_ROOT = Path(os.getcwd()).parents[0]
MBTK_PATH = EXPERIMENTS_ROOT.parents[0]
sys.path.insert(0, str(MBTK_PATH))

from mbtk.dataset.sources.SampledBayesianNetworkDatasetSource import SampledBayesianNetworkDatasetSource
from mbtk.dataset.ExperimentalDataset import ExperimentalDataset
from mbtk.dataset.ExperimentalDatasetDefinition import ExperimentalDatasetDefinition
from mbtk.experiment.ExperimentDefinition import ExperimentDefinition
from mbtk.experiment.ExperimentRun import ExperimentRun
from mbtk.experiment.AlgorithmRun import AlgorithmRun
from mbtk.experiment.AlgorithmRunDatapoint import AlgorithmRunDatapoint
import mbtk.math.DSeparationCITest
from mbtk.math.BNCorrelationEstimator import BNCorrelationEstimator
import mbtk.utilities.experiment as util

# Experiment-specific modules
import expsetup
import commands


def main():
    experimental_setup = expsetup.DsepAlgsEvalExpSetup()

    argparser = create_argparser(experimental_setup)
    arguments = argparser.parse_args()

    experimental_setup.Paths = expsetup.DsepAlgsEvalExpPathSet(EXPERIMENTS_ROOT)

    experimental_setup.setup(arguments)
    experimental_setup.ExperimentDef = define_experiment(experimental_setup)
    experimental_setup.ExDsDef = define_source(experimental_setup)
    experimental_setup.update_paths()
    experimental_setup.AlgorithmRunParameters = create_algrun_parameters(experimental_setup)

    # Handle the (object, verb) command in the arguments
    command_handled = util.handle_command(arguments, experimental_setup)
    if command_handled is False:
        arguments.function(experimental_setup)



def create_argparser(experimental_setup):
    argparser = argparse.ArgumentParser()

    argparser.add_argument('source_type', type=str, default=None,
                           choices=['bn', 'ds'])
    argparser.add_argument('source_name', type=str, default=None,
                           choices=experimental_setup.BayesianNetworks)
    argparser.add_argument('--sample-count', type=str, default=None,
                           help='sample count of the datasetmatrix'
                                ' (required if source_type is "ds", otherwise ignored)')
    argparser.add_argument('algorithm', type=str, default=None,
                           choices=experimental_setup.Algorithms)
    object_subparsers = argparser.add_subparsers(dest='object')
    object_subparsers.required = True

    util.configure_objects_subparser__exp(object_subparsers)
    util.configure_objects_subparser__paths(object_subparsers)
    util.configure_objects_subparser__exds(object_subparsers)

    commands.configure_objects_subparser__summary(object_subparsers)

    return argparser



def define_experiment(experimental_setup):
    experiment_name = 'DsepAlgsEval'
    experimentDef = ExperimentDefinition(
        experimental_setup.Paths.ExpRunRepository,
        experiment_name,
        create_experiment_run_stem(experimental_setup)
    )

    experimentDef.experiment_run_class = ExperimentRun
    experimentDef.algorithm_run_class = AlgorithmRun
    experimentDef.exds_definition = experimental_setup.ExDsDef
    experimentDef.algorithm_run_configuration = {
        'algorithm': experimental_setup.AlgorithmClass
    }
    experimentDef.algorithm_run_datapoint_class = AlgorithmRunDatapoint
    experimentDef.save_algorithm_run_datapoints = True

    return experimentDef



def define_source(experimental_setup):
    if experimental_setup.Source != 'ds':
        return None

    bayesian_network = experimental_setup.BayesianNetworkName
    sample_count = experimental_setup.SampleCount
    sample_count_str = experimental_setup.SampleCountString
    exds_name = 'synthetic_{}_{}'.format(bayesian_network, sample_count_str)

    exdsDef = ExperimentalDatasetDefinition(experimental_setup.Paths.ExDsRepository, exds_name)
    exdsDef.exds_class = ExperimentalDataset
    exdsDef.source = SampledBayesianNetworkDatasetSource
    exdsDef.source_configuration = {
        'sourcepath': experimental_setup.Paths.BIFRepository / (bayesian_network + '.bif'),
        'sample_count': sample_count,
        'random_seed': 81628965211 + sample_count
    }

    return exdsDef



def create_algrun_parameters(experimental_setup):
    bayesian_network = experimental_setup.BayesianNetwork
    citrrepo = experimental_setup.Paths.CITestResultRepository
    hrrepo = experimental_setup.Paths.HResultRepository

    variables = list(range(len(bayesian_network)))
    parameters_list = list()
    for target in variables:
        parameters = {
            'target': target,
            'all_variables': variables,
            'source_bayesian_network': bayesian_network,
            'ci_test_class': mbtk.math.DSeparationCITest.DSeparationCITest,
            'correlation_heuristic_class': BNCorrelationEstimator,
            'tags': [
                experimental_setup.AlgorithmName,
                experimental_setup.BayesianNetworkName
            ],
        }

        filename_stem = create_algrun_filename_stem(experimental_setup, parameters)
        citr_filename = 'citr_' + filename_stem + '.pickle'
        parameters['ci_test_results_path__save'] = citrrepo / citr_filename
        hr_filename = 'hr_' + filename_stem + '.pickle'
        parameters['heuristic_results_path__save'] = hrrepo / hr_filename
        parameters_list.append(parameters)

    for index, parameters in enumerate(parameters_list):
        parameters['index'] = index
        parameters['ID'] = 'run_T{target}'.format(**parameters)

    return parameters_list



def create_experiment_run_stem(experimental_setup):
    source = experimental_setup.Source
    alg = experimental_setup.AlgorithmName
    sample_count = experimental_setup.SampleCountString
    bayesian_network = experimental_setup.BayesianNetworkName

    if source == 'bn':
        return f'{source}_{bayesian_network}_{alg}'

    if source == 'ds':
        return f'{source}_{bayesian_network}_{sample_count}_{alg}'



def create_algrun_filename_stem(experimental_setup, parameters):
    target = parameters['target']
    if experimental_setup.Source == 'bn':
        stem = 'bn_{}_{}_T{}'.format(
            experimental_setup.BayesianNetworkName,
            experimental_setup.AlgorithmName,
            target)
    if experimental_setup.Source == 'ds':
        stem = 'ds_{}_{}_{}_T{}'.format(
            experimental_setup.BayesianNetworkName,
            experimental_setup.SampleCountString,
            experimental_setup.AlgorithmName,
            target)

    return stem



if __name__ == '__main__':
    main()
