import sys
import os
from pathlib import Path
import argparse

# Assume that the 'experiments' folder, which contains this file, is directly
# near the 'mbff' package.
EXPERIMENTS_ROOT = Path(os.getcwd()).parents[0]
MBFF_PATH = EXPERIMENTS_ROOT.parents[0]
sys.path.insert(0, str(MBFF_PATH))

import mbff.utilities.experiment as util

# Experiment-specific modules
import expsetup
import commands
import definitions
import algrun_parameters


def create_argparser():
    argparser = argparse.ArgumentParser()

    # Primary arguments, which must be provided before an 'object' argument
    argparser.add_argument('--dont-preload-adtree', action='store_true')
    argparser.add_argument('--algrun-tag', type=str, default=None, nargs='?')
    argparser.add_argument('--llt', type=int, default=0, nargs='?')

    argparser.add_argument('dataset_name', type=str, default=None)
    argparser.add_argument('sample_count', type=str, default=None)

    object_subparsers = argparser.add_subparsers(dest='object')
    object_subparsers.required = True

    # Commands provided by the MBFF library
    util.configure_objects_subparser__paths(object_subparsers)
    util.configure_objects_subparser__exp_def(object_subparsers)
    util.configure_objects_subparser__exds_def(object_subparsers)
    util.configure_objects_subparser__exds(object_subparsers)
    util.configure_objects_subparser__exp(object_subparsers)
    util.configure_objects_subparser__parameters(object_subparsers)
    util.configure_objects_subparser__datapoints(object_subparsers)

    # Custom commands, specific to the dcMI experiment
    commands.configure_objects_subparser__adtree(object_subparsers)
    commands.configure_objects_subparser__plot(object_subparsers)
    commands.configure_objects_subparser__summary(object_subparsers)

    return argparser


################################################################################
# Command-line interface
if __name__ == '__main__':

    # Parse CLI arguments
    argparser = create_argparser()
    arguments = argparser.parse_args()

    # Create the experimental setup, based on CLI arguments
    experimental_setup = expsetup.DCMIEvExpSetup()
    experimental_setup.set_arguments(arguments)
    experimental_setup.CITest_Significance = 0.95
    experimental_setup.Paths = expsetup.DCMIEvExpPathSet(EXPERIMENTS_ROOT)
    experimental_setup.ExDsDef = definitions.exds_definition(experimental_setup)
    experimental_setup.ExperimentDef = definitions.experiment_definition(experimental_setup)
    experimental_setup.update_paths()
    experimental_setup.AlgorithmRunParameters = algrun_parameters.create_algrun_parameters(experimental_setup)

    # The experiment defines many algorithm configurations, in the field
    # experimental_setup.AlgorithmRunParameters. If only a subset should be
    # acted upon at the moment, an `algrun_tag` can be specified on the CLI,
    # which causes the experiment to ignore all algorithm configurations that
    # have not been marked with the specified `algrun_tag`.
    if arguments.algrun_tag is not None:
        experimental_setup.filter_algruns_by_tag(arguments.algrun_tag)

    # By default, the experiment will preload the saved AD-tree, required by
    # algorithm configurations that require it (tagged with `adtree`),
    # although this only happens if there are such configurations selected for
    # running, after filtering by `algrun_tag`.
    should_preload_adtree = \
        arguments.object == 'exp' \
        and arguments.verb == 'run' \
        and not arguments.dont_preload_adtree

    if should_preload_adtree is True:
        if experimental_setup.is_tag_present_in_any_algrun('adtree-static'):
            experimental_setup.preload_static_ADTree()
        if experimental_setup.is_tag_present_in_any_algrun('adtree-dynamic'):
            experimental_setup.preload_dynamic_ADTree()

    # Handle the (object, verb) command in the arguments
    command_handled = util.handle_command(arguments, experimental_setup)
    if command_handled is False:
        commands.handle_command(arguments, experimental_setup)
