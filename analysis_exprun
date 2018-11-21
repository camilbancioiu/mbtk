#!/usr/bin/python3

from argparse import ArgumentParser
import utilities as util
import itertools
import definitions as Definitions

import analysis_exprun_operations

operations = ['print-ksic-stats', 'plot-ksic-stats', 'plot-iteration-time',
        'print-algrun-durations', 'print-algrun-results', 'plot-accuracy-igt-vs-ks',
        'generate-algrun-samples', 'print-accuracy-stats', 'custom-fix']

def create_argument_parser():
    parser = ArgumentParser()
    parser.add_argument(
            'operation', 
            help='The operation to be performed on the targets.',
            choices=operations)
    parser.add_argument(
            'targets', type=str, nargs='+',
            help='The experiments to perform the analysis on.')
    parser.add_argument('-cq', '--custom-q', type=bool)

    parser.set_defaults(
            operation=operations[0],
            targets=['all'],
            custom_q=False
            )

    return parser


if __name__ == '__main__':
    parser = create_argument_parser()
    arguments = parser.parse_args()

    analysis_exprun_operations.arguments = arguments
    op = arguments.operation
    experiment_names = arguments.targets

    if op == 'plot-ksic-stats':
        analysis_exprun_operations.op_plot_ksic_stats(experiment_names)
    elif op == 'print-ksic-stats':
        analysis_exprun_operations.op_print_ksic_stats(experiment_names)
    elif op == 'plot-iteration-time':
        analysis_exprun_operations.op_plot_iteration_time(experiment_names)
    elif op == 'print-algrun-durations':
        analysis_exprun_operations.op_print_algrun_durations(experiment_names)
    elif op == 'print-algrun-results':
        analysis_exprun_operations.op_print_algrun_results(experiment_names)
    elif op == 'plot-accuracy-igt-vs-ks':
        analysis_exprun_operations.op_plot_accuracy_igt_vs_ks(experiment_names)
    elif op == 'generate-algrun-samples':
        analysis_exprun_operations.op_generate_algrun_samples(experiment_names)
    elif op == 'print-accuracy-stats':
        analysis_exprun_operations.op_print_accuracy_stats(experiment_names)
    elif op == 'custom-fix':
        analysis_exprun_operations.op_custom_fix()
    else:
        raise Exception('Operation not recognised: {}'.format(op))

