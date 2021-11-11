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

import mbtk.utilities.experiment as util

# Experiment-specific modules
import expsetup
import commands


def main():
    argparser = create_argparser()
    arguments = argparser.parse_args()

    experimental_setup = expsetup.DsepAlgsEvalExpSetup(EXPERIMENTS_ROOT, arguments)

    try:
        command_handled = util.handle_command(arguments, experimental_setup)
        if command_handled is False:
            arguments.function(experimental_setup)
    except Exception as e:
        print(experimental_setup.bayesian_network_name,
              experimental_setup.sample_count_string,
              experimental_setup.algorithm_name,
              e)
        raise



def create_argparser():
    argparser = argparse.ArgumentParser()

    argparser.add_argument('source_type', type=str, default=None,
                           choices=expsetup.SOURCES)
    argparser.add_argument('source_name', type=str, default=None,
                           choices=expsetup.BAYESIAN_NETWORKS)
    argparser.add_argument('-s', '--sample-count', type=str, default=None,
                           help='sample count of the datasetmatrix'
                                ' (required if source_type is "ds", otherwise ignored)')
    argparser.add_argument('algorithm', type=str, default=None,
                           choices=expsetup.ALGORITHMS)
    object_subparsers = argparser.add_subparsers(dest='object')
    object_subparsers.required = True

    util.configure_objects_subparser__exp(object_subparsers)
    util.configure_objects_subparser__paths(object_subparsers)
    util.configure_objects_subparser__exds(object_subparsers)

    commands.configure_objects_subparser__summary(object_subparsers)

    return argparser



if __name__ == '__main__':
    main()
