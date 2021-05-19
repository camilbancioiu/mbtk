import sys
import math
import time
import statistics
import gc
import mbtk.math.DoFCalculators as DoFCalculators
import mbtk.structures.DynamicADTree
import tests.utilities as testutil
import tests.test_AlgorithmIPCMBWithGtests as ipcmb_tests


DEMO_HEADER = """
This is a short demonstration that highlights the efficiency of the dcMI
optimization applied on the computation of the conditional G-test. It consists
of a small experiment which involves discovering the Markov blankets of all the
variables in a Bayesian network. The discovery itself is performed by the
IPC-MB algorithm, repeated for each variable as its target.

For this demonstration, dcMI will be compared with another optimization, namely
the dynamic AD-tree. Both are added to IPC-MB, one at a time, to reduce the
discovery time. Therefore the demonstrative experiment will be run twice:
firstly with a dynamic AD-tree, secondly with dcMI.

IPC-MB durations are gathered while the experiment progresses and they will be
reported at the end.

The data set is synthetically generated at random from the ALARM network, which
contains 37 variables. A total of 8000 complete samples are generated in memory
for this demonstration, before the experiment starts.

The demonstration should require less than 3 minutes to complete (usually less
than 2 minutes, even on modest machines)."""


def main():
    print(DEMO_HEADER)
    print()
    print('Press [ENTER] to continue')
    input()
    sys.stdout.flush()

    folders = testfolders()
    ds = ds_alarm_8e3()

    durations = dict()

    print(experiment_header('dynamic AD-tree'))
    sys.stdout.flush()
    durations['dynadt'] = run_demo_ipcmb_test__dynamic_adtree(folders, ds)

    print()
    print(experiment_header('dcMI'))
    sys.stdout.flush()
    durations['dcmi'] = run_demo_ipcmb_test__dcmi(folders, ds)

    print()
    print(header('Results'))
    sys.stdout.flush()
    print()

    table = '{:<30}|{:>8.4}{:>8.4}{:>8.4}'
    print(table.format('Optimization', 'Mean', 'Min', 'Max'))
    print('_' * 60)
    print(table_row(table, 'IPC-MB with dynamic AD-tree', durations['dynadt']))
    print(table_row(table, 'IPC-MB with dcMI', durations['dcmi']))
    sys.stdout.flush()



def testfolders():
    folders = dict()
    root = testutil.ensure_empty_tmp_subfolder('test_demo_ipcmb_with_gtests')

    subfolders = ['dofCache', 'ci_test_results', 'jht', 'adtrees', 'dynamic_adtrees']
    for subfolder in subfolders:
        path = root / subfolder
        path.mkdir(parents=True, exist_ok=True)
        folders[subfolder] = path

    folders['root'] = root
    return folders



def ds_alarm_8e3():
    """ The 'alarm' dataset with 8e3 samples"""
    configuration = dict()
    configuration['label'] = 'ds_alarm_8e3'
    configuration['sourcepath'] = testutil.bif_folder / 'alarm.bif'
    configuration['sample_count'] = int(8e3)
    configuration['random_seed'] = 97
    configuration['values_as_indices'] = True
    configuration['objectives'] = []
    return testutil.MockDataset(configuration)



def experiment_header(optimization_name):
    text = "Starting IPC-MB runs optimized with {}".format(optimization_name)
    return header(text)



def header(text):
    border_len = len(text)
    return ('=' * border_len) + '\n' + text + '\n' + ('.' * border_len)



def duration_column(duration):
    dashes = int(duration*6)
    column = '>' + ('-' * dashes) + '|'
    return column


def table_row(table, label, durations):
    return table.format(
        label,
        str(statistics.mean(durations)) + 's',
        str(min(durations)) + 's',
        str(max(durations)) + 's')


def run_demo_ipcmb_test__dynamic_adtree(folders, ds):
    LLT = 0
    path = folders['dynamic_adtrees'] / (ds.label + '.pickle')
    parameters = ipcmb_tests.make_parameters__adtree(DoFCalculators.StructuralDoF, LLT, None, path, path)
    parameters['ci_test_ad_tree_class'] = mbtk.structures.DynamicADTree.DynamicADTree
    parameters['source_bayesian_network'] = None
    parameters['algorithm_debug'] = 1

    def post_run(ipcmb, parameters):
        adtree = ipcmb.CITest.AD_tree
        parameters['ci_test_ad_tree_preloaded'] = adtree

    return run_demo_ipcmb_test__optimized(folders, ds, parameters, 'dynamic AD-tree', post_run)



def run_demo_ipcmb_test__dcmi(folders, ds):
    jht_path = folders['jht'] / 'jht_{}.pickle'.format(ds.label)
    dof_path = folders['dofCache'] / 'dof_cache_{}.pickle'.format(ds.label)
    parameters = ipcmb_tests.make_parameters__dcmi(DoFCalculators.CachedStructuralDoF, jht_path, dof_path)
    parameters['source_bayesian_network'] = None
    parameters['algorithm_debug'] = 1

    def post_run(ipcmb, parameters):
        jht = ipcmb.CITest.JHT
        parameters['ci_test_jht_preloaded'] = jht

    return run_demo_ipcmb_test__optimized(folders, ds, parameters, 'dcMI', post_run)



def run_demo_ipcmb_test__optimized(folders, ds, parameters, name, post_run):
    targets = range(ds.datasetmatrix.get_column_count('X'))
    durations = list()

    for target in targets:
        print()
        print('Discovering Markov blanket using IPC-MB optimized with {}, target variable {} of {}'.format(name, target, len(targets) - 1))
        sys.stdout.flush()
        target_start = time.time()
        mb, _, ipcmb = ipcmb_tests.run_IPCMB(ds, target, parameters)
        target_end = time.time()
        target_duration = target_end - target_start
        durations.append(target_duration)

        column = duration_column(target_duration)
        print('Duration {:>6.3}s {}'.format(target_duration, column))

        post_run(ipcmb, parameters)
        gc.collect()
        sys.stdout.flush()

    return durations



if __name__ == '__main__':
    main()
