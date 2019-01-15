#!/usr/bin/python3

from argparse import ArgumentParser
import utilities as util
import itertools
import definitions as Definitions
import analysis_exds_operations

operations = ['print-table-diff']

def create_argument_parser():
    parser = ArgumentParser()
    parser.add_argument(
            'operation', 
            help='The operation to be performed on the targets.',
            choices=operations)
    parser.add_argument(
            'targets', type=str, nargs='+',
            help='The datasets to perform the analysis on.')

    parser.add_argument('-dt', '--diff-table-tag', type=str)

    parser.set_defaults(
            operation=operations[0],
            diff_table_tag='',
            targets=['all']
            )

    return parser


if __name__ == '__main__':
    parser = create_argument_parser()
    arguments = parser.parse_args()

    op = arguments.operation
    exds_names = arguments.targets
    analysis_exds_operations.arguments = arguments

    if op == 'print-table-diff':
        analysis_exds_operations.op_print_table_diff(exds_names)
    else:
        raise Exception('Operation not recognised: {}'.format(op))

