import sys
import os
from pathlib import Path

# Assume that the 'experiments' folder, which contains this file, is directly
# near the 'mbff' package.
EXPERIMENTS_ROOT = Path(os.getcwd()).parents[0]
MBFF_PATH = EXPERIMENTS_ROOT.parents[0]
sys.path.insert(0, str(MBFF_PATH))

# Experiment-specific modules
import expsetup

EXPERIMENT_NAME = 'dcMIEvExpII'
DATA_FOLDERS = ['adtree_analysis', 'adtrees', 'algorithm_run_datapoints',
                'algorithm_run_logs', 'ci_test_results', 'dof_cache', 'jht',
                'summaries']


def restructure():
    paths = expsetup.DCMIEvExpPathSet(EXPERIMENTS_ROOT)
    new_experiment_folder = paths.ExpRunRepository / EXPERIMENT_NAME
    create_new_structure(new_experiment_folder)

    subexperiment_folders = get_subexperiment_folders(paths.ExpRunRepository)
    for subexp in subexperiment_folders:
        print(subexp)
        restructure_subexperiment_folder(new_experiment_folder, subexp)
        restructure_subexperiment_locks(new_experiment_folder, subexp)



def create_new_structure(root):
    root.mkdir(parents=True, exist_ok=True)
    for folder in DATA_FOLDERS:
        (root / folder).mkdir(parents=True, exist_ok=True)



def get_subexperiment_folders(repository):
    for path in repository.iterdir():
        if not path.is_dir():
            continue
        if not path.name.startswith(EXPERIMENT_NAME + '_'):
            continue
        yield path



def restructure_subexperiment_folder(root, subexp):
    for datafolder in DATA_FOLDERS:
        new_datafolder = root / datafolder
        new_subexp_name = get_new_subexperiment_name(subexp.name)
        try:
            (subexp / datafolder).rename(new_datafolder / new_subexp_name)
        except FileNotFoundError:
            pass



def restructure_subexperiment_locks(root, subexp):
    locks_folder = root / 'locks'
    locks_folder.mkdir(parents=True, exist_ok=True)

    for path in subexp.iterdir():
        if not path.is_file():
            continue
        if not path.name.startswith('locked'):
            continue
        new_subexp_name = get_new_subexperiment_name(subexp.name)
        (locks_folder / new_subexp_name).mkdir(parents=True, exist_ok=True)
        path.rename(locks_folder / new_subexp_name / path.name)



def get_new_subexperiment_name(old_name):
    substring_index = len(EXPERIMENT_NAME) + 1
    return old_name[substring_index:]



if __name__ == '__main__':
    restructure()
