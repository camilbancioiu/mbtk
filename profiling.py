import sys
import tests.utilities as testutil
from mbff.algorithms.mb.ipcmb import AlgorithmIPCMB
import mbff.structures.DynamicADTree
import mbff.math.G_test__with_dcMI
import mbff.math.G_test__with_AD_tree
import mbff.math.DoFCalculators as DoFCalculators

import cProfile
import pstats
from pstats import SortKey


def testfolders():
    folders = dict()
    root = testutil.ensure_empty_tmp_subfolder('profile_dynadtree')

    subfolders = ['dofCache', 'jht', 'dynamic_adtrees']
    for subfolder in subfolders:
        path = root / subfolder
        path.mkdir(parents=True, exist_ok=True)
        folders[subfolder] = path

    folders['root'] = root
    return folders



def dataset(size):
    configuration = dict()
    configuration['label'] = 'ds_lc_repaired_{}'.format(int(size))
    configuration['sourcepath'] = testutil.bif_folder / 'lc_repaired.bif'
    configuration['sample_count'] = int(size)
    configuration['values_as_indices'] = True
    configuration['objectives'] = []
    configuration['method'] = 'exact'
    configuration['random_seed'] = 1984
    return testutil.MockDataset(configuration)



def make_parameters__unoptimized(dof_class):
    parameters = dict()
    parameters['ci_test_class'] = mbff.math.G_test__unoptimized.G_test
    parameters['ci_test_dof_calculator_class'] = dof_class
    parameters['ci_test_gc_collect_rate'] = 0
    return parameters



def make_parameters__dynadtree(dof_class, llt, adtree=None, path_load=None, path_save=None):
    parameters = dict()
    parameters['ci_test_class'] = mbff.math.G_test__with_AD_tree.G_test
    parameters['ci_test_dof_calculator_class'] = dof_class
    parameters['ci_test_ad_tree_class'] = mbff.structures.DynamicADTree.DynamicADTree
    parameters['ci_test_gc_collect_rate'] = 0
    parameters['ci_test_ad_tree_preloaded'] = adtree
    parameters['ci_test_ad_tree_class'] = mbff.structures.ADTree.ADTree
    parameters['ci_test_ad_tree_path__load'] = path_load
    parameters['ci_test_ad_tree_path__save'] = path_save
    parameters['ci_test_ad_tree_leaf_list_threshold'] = llt
    return parameters



def make_parameters__dcmi(dof_class, jht_path=None, dof_path=None):
    parameters = dict()
    parameters['ci_test_class'] = mbff.math.G_test__with_dcMI.G_test
    parameters['ci_test_dof_calculator_class'] = dof_class
    parameters['ci_test_gc_collect_rate'] = 0
    parameters['ci_test_jht_path__load'] = jht_path
    parameters['ci_test_jht_path__save'] = jht_path
    parameters['ci_test_dof_calculator_cache_path__load'] = dof_path
    parameters['ci_test_dof_calculator_cache_path__save'] = dof_path
    return parameters



def make_IPCMB(ds, target, extra_parameters=None):
    if extra_parameters is None:
        extra_parameters = dict()

    parameters = dict()
    parameters['target'] = target
    parameters['ci_test_debug'] = 0
    parameters['ci_test_significance'] = 0.8
    parameters['ci_test_results__print_accurate'] = False
    parameters['ci_test_results__print_inaccurate'] = True
    parameters['algorithm_debug'] = 0
    parameters['omega'] = ds.omega
    parameters['source_bayesian_network'] = ds.bayesiannetwork

    parameters.update(extra_parameters)
    ipcmb = AlgorithmIPCMB(ds.datasetmatrix, parameters)
    return ipcmb



def run_IPCMB(ds, target, parameters):
    ipcmb = make_IPCMB(ds, target, parameters)
    mb = ipcmb.select_features()
    ci_tests = ipcmb.CITest.ci_test_results
    return (mb, ci_tests)



def run_unoptimized():
    ds = dataset(4e3)
    parameters = make_parameters__unoptimized(DoFCalculators.StructuralDoF)

    print('start')
    targets = range(ds.datasetmatrix.get_column_count('X'))
    for target in targets:
        mb, _ = run_IPCMB(ds, target, parameters)
        print('MB({}) = {}'.format(target, mb))



def run_dynadtree():
    ds = dataset(4e4)
    LLT = 0
    folders = testfolders()
    path = folders['dynamic_adtrees'] / (ds.label + '.pickle')
    parameters = make_parameters__dynadtree(DoFCalculators.StructuralDoF, LLT, None, path, path)
    parameters['ci_test_ad_tree_class'] = mbff.structures.DynamicADTree.DynamicADTree

    print('start')
    targets = range(ds.datasetmatrix.get_column_count('X'))
    for target in targets:
        mb, _ = run_IPCMB(ds, target, parameters)
        print('MB({}) = {}'.format(target, mb))



def run_dcmi():
    ds = dataset(4e4)
    folders = testfolders()
    jht_path = folders['jht'] / 'jht_{}.pickle'.format(ds.label)
    dof_path = folders['dofCache'] / 'dof_cache_{}.pickle'.format(ds.label)
    parameters = make_parameters__dcmi(DoFCalculators.CachedStructuralDoF, jht_path, dof_path)

    print('start')
    targets = range(ds.datasetmatrix.get_column_count('X'))
    for target in targets:
        mb, _ = run_IPCMB(ds, target, parameters)
        print('MB({}) = {}'.format(target, mb))



def profile_unoptimized():
    cProfile.run('run_unoptimized()', 'unoptimized.pstats')
    p = pstats.Stats('unoptimized.pstats')
    # functions = '|'.join(['count_values'])
    # p.sort_stats(SortKey.CUMULATIVE).print_stats(50, functions)
    p.sort_stats(SortKey.CUMULATIVE).print_stats(50)



def profile_dynadtree():
    cProfile.run('run_dynadtree()', 'dynadtree.pstats')
    p = pstats.Stats('dynadtree.pstats')
    functions = '|'.join(['create_row_subselections_by_value'])
    p.sort_stats(SortKey.CUMULATIVE).print_stats(50, functions)



def profile_dcmi():
    cProfile.run('run_dcmi()', 'dcmi.pstats')
    p = pstats.Stats('dcmi.pstats')
    functions = '|'.join(['count_values'])
    p.sort_stats(SortKey.CUMULATIVE).print_stats(50, functions)


if __name__ == '__main__':
    try:
        profile = sys.argv[1]
    except IndexError:
        profile = 'dcmi'

    print('profile:', profile)
    if profile == 'unoptimized':
        profile_unoptimized()
    elif profile == 'dynadtree':
        profile_dynadtree()
    elif profile == 'dcmi':
        profile_dcmi()
    else:
        print('unknown profile')
