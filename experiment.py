#!/usr/bin/python3

"""
Entry point for manipulating Experiments. Provides the following operations:

.. autosummary::

    experiment_operations.op_list
    experiment_operations.op_run
    experiment_operations.op_lock
    experiment_operations.op_unlock
    experiment_operations.op_desc
    experiment_operations.op_alg_runs_print_to_csv
    experiment_operations.op_alg_runs_delete

"""

from argparse import ArgumentParser
import experiment_operations

operations = ['list', 'desc', 'run', 'lock', 'unlock', 'alg-runs', 'build-ks-gamma']

def create_argument_parser():
    """
    Instantiates and configures an ArgumentParser.
    """
    parser = ArgumentParser()
    arg_operation = parser.add_argument( 'operation', choices=operations)
    arg_targets = parser.add_argument( 'targets', type=str, nargs='+')

    # TODO AlgRun operations should be moved to the entry-point analysis_exprun.
    alg_runs_action = parser.add_mutually_exclusive_group()
    arg_alg_runs_action__print_to_csv = alg_runs_action.add_argument(
            '-c', '--print-to-csv', action='store_const', dest='alg_runs_action', const='print_to_csv')
    arg_alg_runs_action__delete = alg_runs_action.add_argument(
            '-d', '--delete', action='store_const', dest='alg_runs_action', const='delete')

    parser.set_defaults(
            operation='list',
            targets=['all'],
            alg_runs_action='print_to_csv'
            )

    # Help strings for the arguments defined above:
    arg_operation.help ='The operation to be performed on the targets.'
    arg_targets.help = 'The experiment to perform the operation on.'
    arg_alg_runs_action__print_to_csv.help = 'Load saved AlgorithmRun instances and convert them to CSV.'
    arg_alg_runs_action__delete.help = 'Delete saved AlgorithmRun instances.'

    return parser


def entry_point_experiment():
    """
    Main function of the entry-point ``experiment``.

    Invokes the corresponding ``operation`` on the Experiments specified as
    ``targets``. See :ref:`defined_operations_experiment`.
    """
    parser = create_argument_parser()
    arguments = parser.parse_args()

    op = arguments.operation
    experiment_names = arguments.targets

    if op == 'list':
        experiment_operations.op_list(experiment_names)
    elif op == 'desc':
        experiment_operations.op_desc(experiment_names)
    elif op == 'run':
        experiment_operations.op_run(experiment_names)
    elif op == 'lock':
        experiment_operations.op_lock(experiment_names)
    elif op == 'unlock':
        experiment_operations.op_unlock(experiment_names)
    elif op == 'alg-runs':
        if arguments.alg_runs_action == 'print_to_csv':
            experiment_operations.op_alg_runs_print_to_csv(experiment_names)
        elif arguments.alg_runs_action == 'delete':
            experiment_operations.op_alg_runs_delete(experiment_names)
        else:
            raise Exception('Unkown action for alg-runs.')
    elif op == 'build-ks-gamma':
        experiment_operations.op_build_ks_gamma(experiment_names)
    else:
        raise Exception('Operation not recognised: {}'.format(op))


if __name__ == '__main__':
    entry_point_experiment()
