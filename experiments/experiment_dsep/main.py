import sys
import os
from pathlib import Path
import argparse

# Assume that the 'experiments' folder, which contains this file, is directly
# near the 'mbtk' package.
EXPERIMENTS_ROOT = Path(os.getcwd()).parents[0]
MBTK_PATH = EXPERIMENTS_ROOT.parents[0]
sys.path.insert(0, str(MBTK_PATH))

from mbtk.experiment.ExperimentDefinition import ExperimentDefinition
from mbtk.experiment.ExperimentRun import ExperimentRun
from mbtk.experiment.AlgorithmRun import AlgorithmRun
from mbtk.experiment.AlgorithmRunDatapoint import AlgorithmRunDatapoint
import mbtk.math.DSeparationCITest
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
    experimental_setup.update_paths()
    experimental_setup.AlgorithmRunParameters = create_algrun_parameters(experimental_setup)

    # Handle the (object, verb) command in the arguments
    command_handled = util.handle_command(arguments, experimental_setup)
    if command_handled is False:
        commands.handle_command(arguments, experimental_setup)


def create_argparser(experimental_setup):
    argparser = argparse.ArgumentParser()

    argparser.add_argument('algorithm', type=str, default=None,
                           choices=experimental_setup.AllowedAlgorithms)
    argparser.add_argument('bayesian_network', type=str, default=None,
                           choices=experimental_setup.AllowedBayesianNetworks)
    object_subparsers = argparser.add_subparsers(dest='object')
    object_subparsers.required = True

    util.configure_objects_subparser__exp(object_subparsers)
    util.configure_objects_subparser__paths(object_subparsers)

    commands.configure_objects_subparser__summary(object_subparsers)

    return argparser


def define_experiment(experimental_setup):
    experiment_name = 'DsepAlgsEval'
    alg = experimental_setup.AlgorithmName
    bayesian_network = experimental_setup.BayesianNetworkName
    experimentDef = ExperimentDefinition(
        experimental_setup.Paths.ExpRunRepository,
        experiment_name,
        f'{alg}_{bayesian_network}'
    )

    experimentDef.experiment_run_class = ExperimentRun
    experimentDef.algorithm_run_class = AlgorithmRun
    experimentDef.algorithm_run_configuration = {
        'algorithm': experimental_setup.AlgorithmClass
    }
    experimentDef.algorithm_run_datapoint_class = AlgorithmRunDatapoint

    return experimentDef


def create_algrun_parameters(experimental_setup):
    parameters_list = create_algrun_parameters__ipcmb(experimental_setup)
    for index, parameters in enumerate(parameters_list):
        parameters['index'] = index
        parameters['ID'] = 'run_T{target}'.format(**parameters)
    return parameters_list


def create_algrun_parameters__ipcmb(experimental_setup):
    bayesian_network = experimental_setup.BayesianNetwork
    citrrepo = experimental_setup.Paths.CITestResultRepository

    variables = list(range(len(bayesian_network)))
    parameters_list = list()
    for target in variables:
        citr_filename = 'citr_{}_T{}_dsep.pickle'.format(
            experimental_setup.BayesianNetworkName,
            target)

        parameters = {
            'target': target,
            'all_variables': variables,
            'source_bayesian_network': bayesian_network,
            'ci_test_class': mbtk.math.DSeparationCITest.DSeparationCITest,
            'ci_test_results_path__save': citrrepo / citr_filename,
            'tags': [
                experimental_setup.AlgorithmName,
                experimental_setup.BayesianNetworkName
            ],
        }
        parameters_list.append(parameters)



    return parameters_list


if __name__ == '__main__':
    main()
