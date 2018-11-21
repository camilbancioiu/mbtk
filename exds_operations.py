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

import exds_operations_custom

arguments = None
parallelism = 1


### ExDs operation "list"
def op_list(exds_names):
    definitions = get_from_definitions(ExperimentalDatasets, exds_names)
    print(exds_definition_table_header())
    for definition in definitions:
        print(exds_definition_to_table_string(definition))


### ExDs operation "build"
def op_build(exds_names):
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
    map_over_exds_definitions(exds_names, 'Delete', op_delete_single, print_status=True)

def op_delete_single(definition):
    if definition.folder_exists():
        definition.delete_folder()
    return definition


### ExDs operation "lock"
def op_lock(exds_names):
    map_over_exds_definitions(exds_names, 'Lock', op_lock_single, print_status=False)

def op_lock_single(definition):
    if definition.folder_exists():
        definition.lock_folder(arguments.lock_type)
    return definition


### ExDs operation "unlock"
def op_unlock(exds_names):
    map_over_exds_definitions(exds_names, 'Unlock', op_unlock_single, print_status=False)

def op_unlock_single(definition):
    if definition.folder_exists():
        definition.unlock_folder(arguments.lock_type)
    return definition


### ExDs operation "stats --print"
def op_stats_print(exds_names):
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
    rows = map_over_exds_definitions(exds_names, '', op_stats_print_to_csv__stats_to_csv_dict)
    w = csv.DictWriter(sys.stdout, fieldnames=ExperimentalDatasetStats.csv_keys(), lineterminator='\n')
    w.writeheader()
    w.writerows(rows)

def op_stats_print_to_csv__stats_to_csv_dict(definition):
    return ExperimentalDatasetStats(exds_definition=definition).to_csv_dict()


### ExDs operation "build-ks-gamma"
def op_build_ks_gamma(exds_names):
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
    exds_operations_custom.op_custom(exds_names, arguments)
    

####################
### Helper functions

def map_over_exds_definitions(exds_names, opname, op, print_status=True):
    i = 1
    definitions = list(get_from_definitions(ExperimentalDatasets, exds_names))
    results = []
    for definition in definitions:
        if opname != '' and print_status == True:
            mpprint('{} ExDs {} ({} / {})'.format(opname, definition.name, i, len(definitions)))
        results.append(op(definition))
        i += 1
    return results

def map_over_exds_definitions_parallel(exds_names, opname, op, print_status=True):
    definitions = list(get_from_definitions(ExperimentalDatasets, exds_names))
    parallel_arguments = zip(
        definitions, 
        range(1, len(definitions) + 1), 
        itertools.repeat(op),
        itertools.repeat(opname),
        itertools.repeat(print_status),
        itertools.repeat(len(definitions))
        )
    results = []
    with multiprocessing.Pool(parallelism) as pool:
        results = pool.map(map_over_exds_definitions_single, parallel_arguments)
    return results

def map_over_exds_definitions_single(args):
    (definition, index, op, opname, print_status, len_definitions) = args
    if opname != '' and print_status == True:
        mpprint('{} ExDs {} ({} / {})'.format(opname, definition.name, index, len_definitions))
    return op(definition)

def exds_definition_table_format_string():
    return '{0:<30}\t{1:<12}\t{2:<12}\t{3!s:<10}\t{4:<12}\t{5:<18}\t{6}'

def exds_definition_table_header():
    return exds_definition_table_format_string().format('ExDs name', 'Industry', 'Train rows', 'Trim freqs', 'Folder', 'Tags', 'Locks')

def exds_definition_to_table_string(definition):
    locks = filter(definition.folder_is_locked, experimental_dataset.lock_types)
    return exds_definition_table_format_string().format(
            definition.name, definition.industry, definition.train_rows_proportion, 
            definition.trim_freqs, str(definition.folder_exists()), ', '.join(definition.tags), ', '.join(locks))

