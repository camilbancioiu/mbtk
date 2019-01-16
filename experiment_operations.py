"""
Each Experiment operation is represented by a callback with the conventional name
``op_[OperationName]``. Each such callback receives a list of Experiment definition
names, representing the Experiments to act on.
"""

import itertools
import functools as F
import utilities as util
from pathlib import Path
import sys
from pprint import pformat
import csv
from definitions import Experiments, ExperimentalDatasets, get_from_definitions
from experimental_dataset import ExperimentalDataset
from experimental_dataset_stats import ExperimentalDatasetStats
from experimental_pipeline import Experiment, AlgorithmRun
import feature_selection as FS

from experiment_operations_utilities import *


### Experiment operation "list"
def op_list(experiment_names):
    """
    Print a table of the ExDs definitions known to MBFF.

    See :doc:`/infrastructure/experiments/index` on how to write Experiment definitions.
    """
    definitions = get_from_definitions(Experiments, experiment_names)
    print(experiment_definition_table_header())
    for definition in definitions:
        print(experiment_definition_to_table_string(definition))


### Experiment operation "run"
def op_run(experiment_names):
    """
    Run the Experiments.
    """
    map_over_experiment_definitions(experiment_names, 'Run experiment', op_run_single)

def op_run_single(definition):
    experiment = Experiment(definition)
    experiment.run()
    return definition


### Experiment operation "lock"
def op_lock(experiment_names):
    """
    Lock the Experiments.
    """
    map_over_experiment_definitions(experiment_names, 'Lock', op_lock_single)

def op_lock_single(definition):
    definition.lock_folder()
    return definition


### Experiment operation "unlock"
def op_unlock(experiment_names):
    """
    Unlock the Experiments.
    """
    map_over_experiment_definitions(experiment_names, 'Unlock', op_unlock_single)

def op_unlock_single(definition):
    definition.unlock_folder()
    return definition


### Experiment operation "desc"
def op_desc(experiment_names):
    """
    Print a description of the Experiments.
    """
    map_over_experiment_definitions(experiment_names, 'Description', op_desc_single) 

def op_desc_single(definition):
    desc = '* Name: {}\n* Folder: {}\n* Parameters: \n{}\n* Config: \n{}\n\n* ExDs: {}'
    try:
        exds_stats = '\n' + str(ExperimentalDatasetStats(exds_definition=definition.exds_definition))
    except:
        exds_stats = 'n/a'
    output = desc.format(
        definition.name,
        definition.folder,
        pformat(definition.parameters, width=70, indent=4),
        pformat(definition.config, width=70, indent=4),
        exds_stats
        )
    print(output)
    return output


### Experiment operation "alg-runs --print-to-csv"
def op_alg_runs_print_to_csv(experiment_names):
    """
    Deprecated. To be moved to the entry point ``analysis_exprun``.
    """
    map_over_experiment_definitions(experiment_names, '', op_alg_runs_print_to_csv_single)

def op_alg_runs_print_to_csv_single(definition):
    experiment = Experiment(definition)
    algorithm_runs = experiment.load_saved_runs()
    convert_to_dict = F.partial(AlgorithmRun.todict, add_selected_features=False)
    rows = map(convert_to_dict, algorithm_runs)
    w = csv.DictWriter(sys.stdout,
            fieldnames=AlgorithmRun.csv_keys(definition.parameters.keys(), False), 
            lineterminator='\n')
    w.writeheader()
    w.writerows(rows)
    return definition


### Experiment operation
def op_alg_runs_delete(experiment_names):
    """
    Deprecated. To be moved to the entry point ``analysis_exprun``.
    """
    map_over_experiment_definitions(experiment_names, 'Delete alg-runs', op_alg_runs_delete)

def op_alg_runs_delete_single(definition):
    experiment = Experiment(definition)
    experiment.delete_saved_runs()
    return definition


### Experiment operation
def op_build_ks_gamma(experiment_names):
    map_over_experiment_definitions(experiment_names, 'Build KS gamma tables', op_build_ks_gamma_single)

def op_build_ks_gamma_single(definition):
    exds = ExperimentalDataset(definition.exds_definition)
    if exds.definition.folder_is_locked():
        print('{}: ExDs folder is locked, cannot build KS gamma tables.'.format(exds.definition.name))
    else:
        exds.load()
        FS.build_ks_gamma_tables(exds.definition.folder, exds, definition.parameters['target'])
    return definition

