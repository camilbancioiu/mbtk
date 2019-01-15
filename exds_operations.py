"""
Each ExDs operation is represented by a callback with the conventional name
``op_[OperationName]``. Each such callback receives a list of ExDs definition
names, representing the ExDs to act on.
"""

import multiprocessing
import itertools
import utilities as util
from pathlib import Path
import sys
import csv
from definitions import ExperimentalDatasets, get_from_definitions
import definitions as Definitions
from experimental_dataset import build_experimental_dataset, ExperimentalDataset
from experimental_dataset_stats import ExperimentalDatasetStats
import feature_selection as FS
from mpprint import mpprint
import experimental_dataset

from exds_operations_utilities import *
import exds_operations_custom

arguments = None
parallelism = 1


### ExDs operation "list"
def op_list(exds_names):
    """
    Print a table of the ExDs definitions known to MBFF.

    See :doc:`/infrastructure/datasets/index` on how to write ExDs definitions.
    """
    definitions = get_from_definitions(ExperimentalDatasets, exds_names)
    print(exds_definition_table_header())
    for definition in definitions:
        print(exds_definition_to_table_string(definition))


### ExDs operation "build"
def op_build(exds_names):
    """
    Build the specified ExDs, if they haven't been previously built.

    Building a dataset implies creating a corresponding folder in the
    ExperimentalDatasets folder, then processing the source information,
    aggregating it and saving it in the format specified by the ExDs
    definition.

    The difference between ``build`` and ``rebuild`` is that ``build`` will
    *not* overwrite an existing ExDs, whereas ``rebuild`` will never create an
    ExDs - it will only rebuild it *if it already exists*. These two separate
    commands exist mainly to prevent accidental overwriting of existing ExDs.

    See :py:func:`experimental_dataset.build_experimental_dataset`.
    """
    if parallelism > 1:
        map_over_exds_definitions_parallel(exds_names, 'Build', op_build_single, print_status=True)
    else:
        map_over_exds_definitions(exds_names, 'Build', op_build_single, print_status=True)

def op_build_single(definition):
    if not definition.folder_exists():
        build_experimental_dataset(definition)
    else:
        mpprint('ExDs {} already built, skipping.'.format(definition.name))
    return definition


### ExDs operation "build"
def op_rebuild(exds_names):
    """
    Rebuilds the specified ExDs, if the ExDs have been already built and are not locked.

    If the ExDs haven't yet been built, do nothing. Useful when building the
    ExDs is an operation involving randomness or when the ExDs must be saved in
    a new format, during development.

    The difference between ``build`` and ``rebuild`` is that ``build`` will
    *not* overwrite an existing ExDs, whereas ``rebuild`` will never create an
    ExDs - it will only rebuild it *if it already exists*. These two separate
    commands exist mainly to prevent accidental overwriting of existing ExDs.

    See :py:func:`experimental_dataset.build_experimental_dataset`.
    """
    if parallelism > 1:
        map_over_exds_definitions_parallel(exds_names, 'Rebuild', op_rebuild_single, print_status=True)
    else:
        map_over_exds_definitions(exds_names, 'Rebuild', op_rebuild_single, print_status=True)

def op_rebuild_single(definition):
    if definition.folder_exists():
        build_experimental_dataset(definition)
    else:
        mpprint('ExDs {} folder does not exist, it must be built first before rebuilding. Skipping.'.format(definition.name))
    return definition


### ExDs operation "resave"
def op_resave(exds_names):
    """
    Load already built ExDs from their folders, then immediately save them.

    This operation is useful during development, when an existing ExDs must be
    converted from an older format into a newer one, without having to rebuild
    it.
    """
    if parallelism > 1:
        map_over_exds_definitions_parallel(exds_names, 'Resave', op_resave_single)
    else:
        map_over_exds_definitions(exds_names, 'Resave', op_resave_single)

def op_resave_single(definition):
    if definition.folder_exists():
        exds = ExperimentalDataset(definition)
        exds.load()
        exds.save()
    return definition


### ExDs operation "delete"
def op_delete(exds_names):
    """
    Completely delete built ExDs by deleting their folder, if they are not locked.

    A deleted ExDs must be built again from its definition if it is to be used
    again.
    """

    map_over_exds_definitions(exds_names, 'Delete', op_delete_single, print_status=True)

def op_delete_single(definition):
    if definition.folder_exists():
        definition.delete_folder()
    return definition


### ExDs operation "lock"
def op_lock(exds_names):
    """
    Lock the specified ExDs, thus preventing any writing operation to them.

    A locked ExDs must first be unlocked before rebuilding, resaving or
    deleting it.

    An alternative locking operation is locking the already-built KS gamma
    tables. This is performed by adding the flag ``--lock-ks-gamma`` to the
    ``lock`` operation. Note that locking the KS gamma tables will **not** lock
    the entire ExDs. The two operations are separate.
    """

    map_over_exds_definitions(exds_names, 'Lock', op_lock_single, print_status=False)

def op_lock_single(definition):
    if definition.folder_exists():
        definition.lock_folder(arguments.lock_type)
    return definition


### ExDs operation "unlock"
def op_unlock(exds_names):
    """
    Unlock the specified ExDs, thus allowing any writing operation to them.

    A locked ExDs must first be unlocked before rebuilding, resaving or
    deleting it.

    An alternative unlocking operation is unlocking the already-built KS gamma
    tables, if they were previously locked. This is performed by adding the
    flag ``--lock-ks-gamma`` to the ``unlock`` operation. Note that unlocking
    the KS gamma tables will *not* unlock the entire ExDs. The two operations
    are separate.
    """
    map_over_exds_definitions(exds_names, 'Unlock', op_unlock_single, print_status=False)

def op_unlock_single(definition):
    if definition.folder_exists():
        definition.unlock_folder(arguments.lock_type)
    return definition


### ExDs operation "stats --print"
def op_stats_print(exds_names):
    """
    Deprecated, will be moved to the entry-point ``analysis_exds``.
    """
    map_over_exds_definitions(exds_names, 'Stats', op_stats_print_single)

def op_stats_print_single(definition):
    if definition.folder_exists():
        try:
            stats = ExperimentalDatasetStats(exds_definition=definition)
            stats_string = str(stats)
        except:
            stats_string = 'Stats not available'
            raise
        if len(stats_string) > 0:
            print(stats_string)
        print()
    return definition


### ExDs operation "stats --regenerate"
def op_stats_regenerate(exds_names):
    """
    Deprecated, will be moved to the entry-point ``analysis_exds``.
    """
    if parallelism > 1:
        map_over_exds_definitions_parallel(exds_names, 'Stats resave', op_stats_regenerate_single)
    else:
        map_over_exds_definitions(exds_names, 'Stats resave', op_stats_regenerate_single)

def op_stats_regenerate_single(definition):
    if definition.folder_exists():
        exds = ExperimentalDataset(definition)
        exds.load()
        stats = ExperimentalDatasetStats(exds=exds)
        try:
            stats.save()
        except Exception as e:
            print(e)
    return definition


### ExDs operation "stats --print-to-csv"
def op_stats_print_to_csv(exds_names):
    """
    Deprecated, will be moved to the entry-point ``analysis_exds``.
    """
    rows = map_over_exds_definitions(exds_names, '', op_stats_print_to_csv__stats_to_csv_dict)
    w = csv.DictWriter(sys.stdout, fieldnames=ExperimentalDatasetStats.csv_keys(), lineterminator='\n')
    w.writeheader()
    w.writerows(rows)

def op_stats_print_to_csv__stats_to_csv_dict(definition):
    return ExperimentalDatasetStats(exds_definition=definition).to_csv_dict()


### ExDs operation "build-ks-gamma"
def op_build_ks_gamma(exds_names):
    """
    Build the KS gamma tables for the specified ExDs.

    By default, this operation builds the gamma tables for all the objective
    variables in the ExDs. An explicit objective variable can be specified
    using the option ``--build-ks-gamma-table-for-target=T``, where T is a
    number, which means that only the gamma table for the objective variable T
    will be built.
    """
    if parallelism > 1:
        map_over_exds_definitions_parallel(exds_names, 'Build KS gamma tables', op_build_ks_gamma_single, print_status=True)
    else:
        map_over_exds_definitions(exds_names, 'Build KS gamma tables', op_build_ks_gamma_single, print_status=True)

def op_build_ks_gamma_single(definition):
    if definition.folder_exists():
        exds = ExperimentalDataset(definition)
        exds.load()
        if exds.definition.folder_is_locked('ks_gamma'):
            mpprint('{}: ExDs folder is locked, cannot build KS gamma tables.'.format(exds.definition.name))
        else:
            if arguments.build_ks_gamma_table_for_target > -1:
                targets = [arguments.build_ks_gamma_table_for_target]
            else:
                targets = range(0, exds.get_Y_count())
            FS.build_ks_gamma_tables(exds.definition.folder, exds, targets, arguments.build_ks_gamma_optimization)
    return definition


### ExDs operation "build-ks-gamma"
def op_custom(exds_names, arguments):
    """
    Invoke a custom operation.

    See :py:mod:`exds_operations_custom`.
    """
    exds_operations_custom.op_custom(exds_names, arguments)
