#!/usr/bin/python3

"""
Entry point for manipulating ExperimentalDatasets (ExDs). Provides the following operations:

.. autosummary::

  exds_operations.op_list
  exds_operations.op_build
  exds_operations.op_rebuild
  exds_operations.op_resave
  exds_operations.op_delete
  exds_operations.op_lock
  exds_operations.op_unlock
  exds_operations.op_stats_print
  exds_operations.op_stats_regenerate
  exds_operations.op_stats_print_to_csv
  exds_operations.op_build_ks_gamma
  exds_operations.op_custom

More detailed descriptions of these operations are found in :ref:`defined_operations_exds`.

Examples:

    * ``./exds.py list all`` prints a table with all known ExDs definitions.
    * ``./exds.py build -et TextProcessingDS GeneticDS`` will build all ExDs which have the tags ``TextProcessing`` or ``GeneticDS``.
    * ``./exds.py lock reuters_large_200`` will lock the folder of the ExDs named ``reuters_large_200``, preventing it from being rebuilt or deleted.
    * ``./exds.py build_ks_gamma --build-ks-gamma-table-for-target=2 --build-ks-gamma-optimization=off genetic_100_B`` will will build the KS
      gamma table for the objective variable ``2``, without optimization, for the dataset ``genetic_100_B``.
"""

from argparse import ArgumentParser
import exds_operations
import utilities as util
import itertools
import definitions as Definitions

operations = ['list', 'build', 'rebuild', 'resave', 'delete', 'lock', 'unlock', 'stats', 'build-ks-gamma', 'custom']

def create_argument_parser():
    """
    Instantiates and configures an ArgumentParser.
    """
    parser = ArgumentParser()

    arg_operation = parser.add_argument('operation', choices=operations)
    arg_targets = parser.add_argument('targets', type=str, nargs='+')
    arg_parallelism = parser.add_argument('-n', '--parallelism', type=int)
    arg_ks_gamma_optimization = parser.add_argument('--build-ks-gamma-optimization', type=str)
    arg_ks_gamma_target = parser.add_argument('--build-ks-gamma-table-for-target', type=int)

    arg_target_type = parser.add_mutually_exclusive_group()

    arg_target_type__names = arg_target_type.add_argument(
            '-e', '--as_names', action='store_const', dest='targets_type', const='names')

    arg_target_type__exds_lists = arg_target_type.add_argument(
            '-el', '--as_exds_lists', action='store_const', dest='targets_type', const='exds_lists')

    arg_target_type__exds_list_files = arg_target_type.add_argument(
            '-elf', '--as_exds_list_files', action='store_const', dest='targets_type', const='exds_list_files')

    arg_target_type__exds_tags = arg_target_type.add_argument(
            '-et', '--as_exds_tags', action='store_const', dest='targets_type', const='exds_tags')

    # TODO Add help strings to lock_type arguments.
    lock_type = parser.add_mutually_exclusive_group()
    lock_type.add_argument('--lock-exds', action='store_const', dest='lock_type', const='exds')
    lock_type.add_argument('--lock-ks-gamma', action='store_const', dest='lock_type', const='ks_gamma')

    # TODO Stats operations should be moved to the entry-point analysis_exds.
    stats_action = parser.add_mutually_exclusive_group()
    arg_stats_action__print = stats_action.add_argument('-p', '--print',
            action='store_const', dest='stats_action', const='print')
    arg_stats_action__regenerate = stats_action.add_argument('-r', '--regenerate',
            action='store_const', dest='stats_action', const='regenerate')
    arg_stats_action__print_to_csv = stats_action.add_argument('-c', '--print-to-csv',
            action='store_const', dest='stats_action', const='print_to_csv')

    parser.set_defaults(
            operation='list',
            targets=['all'],
            targets_type='names',
            lock_type='exds',
            stats_action='print',
            parallelism=1,
            build_ks_gamma_optimization='full_sharing',
            build_ks_gamma_table_for_target=-1
            )

    # Help strings for the arguments defined above:
    arg_operation.help = 'The operation to be performed on the targets.'
    arg_targets.help = 'The targets of the operation. What these targets are is controlled by the flags that affect the ``targets_type`` setting.'
    arg_target_type__names.help = 'Indicates that the targets are ExDs names. This sets ``targets_type=names``.'
    arg_target_type__exds_lists.help = 'Indicates that the targets are the ExDs from the specified ExDs lists. This sets ``targets_type=exds_lists``.'
    arg_target_type__exds_list_files.help = 'Indicates that the targets are the ExDs from the specified exdslist files. This sets ``targets_type=exds_list_files``.'
    arg_target_type__exds_tags.help = 'Indicates that the targets are the ExDs with the given tags. This sets ``targets_type=exds_tags``.'
    arg_stats_action__print.help = 'Print stats.'
    arg_stats_action__regenerate.help = 'Load the ExDs, recalculate stats and resave them to their own folders.'
    arg_stats_action__print_to_csv.help = 'Load ExDs stats and convert them to CSV.'

    return parser


def get_exds_names_from_arguments(arguments):
    """
    Resolve the ``targets`` argument to actual ExDs definition names,
    based on ``targets_type``.

    See :ref:`cli_usage_exds`.
    """
    exds_names = []
    if arguments.targets_type == 'names':
        exds_names = arguments.targets
    elif arguments.targets_type == 'exds_lists':
        raise NotImplementedError('Not yet supported: exds lists without files')
    elif arguments.targets_type == 'exds_list_files':
        exds_names = []
        for listid in arguments.targets:
            exds_names_in_file = util.load_list_from_file('exdslist', listid)
            exds_names = itertools.chain(exds_names, exds_names_in_file)
    elif arguments.targets_type == 'exds_tags':
        exds_names = []
        for tag in arguments.targets:
            exds_names_by_tag = Definitions.get_definition_names_by_tag(Definitions.ExperimentalDatasets, tag)
            exds_names = itertools.chain(exds_names, exds_names_by_tag)
    return sorted(exds_names)


def entry_point_exds():
    """
    Main function of the entry-point ``exds``.

    Invokes the corresponding ``operation`` on the ExDs specified as
    ``targets``. See :ref:`defined_operations_exds`.
    """
    parser = create_argument_parser()
    arguments = parser.parse_args()

    op = arguments.operation
    exds_operations.arguments = arguments
    exds_operations.parallelism = arguments.parallelism
    exds_names = get_exds_names_from_arguments(arguments)

    if op == 'list':
        exds_operations.op_list(exds_names)
    elif op == 'build':
        exds_operations.op_build(exds_names)
    elif op == 'rebuild':
        exds_operations.op_rebuild(exds_names)
    elif op == 'resave':
        exds_operations.op_resave(exds_names)
    elif op == 'delete':
        exds_operations.op_delete(exds_names)
    elif op == 'lock':
        exds_operations.op_lock(exds_names)
    elif op == 'unlock':
        exds_operations.op_unlock(exds_names)
    elif op == 'stats':
        if arguments.stats_action == 'print':
            exds_operations.op_stats_print(exds_names)
        elif arguments.stats_action == 'regenerate':
            exds_operations.op_stats_regenerate(exds_names)
        elif arguments.stats_action == 'print_to_csv':
            exds_operations.op_stats_print_to_csv(exds_names)
        else:
            raise Exception('Unkown action for stats.')
    elif op == 'build-ks-gamma':
        exds_operations.op_build_ks_gamma(exds_names)
    elif op == 'custom':
        exds_operations.op_custom(exds_names, arguments)
    else:
        raise Exception('Operation not recognised: {}'.format(op))


if __name__ == '__main__':
    entry_point_exds()
