import pytest
import time
import gc
import mbtk.math.DoFCalculators as DoFCalculators

import mbtk.structures.DynamicADTree
import tests.test_AlgorithmIPCMBWithGtests as ipcmb_tests
import tests.utilities as testutil


@pytest.fixture(scope='session')
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



@pytest.mark.demo
def test_ipcmb_efficiency__unoptimized(ds_alarm_8e3):
    ds = ds_alarm_8e3
    parameters = ipcmb_tests.make_parameters__unoptimized(DoFCalculators.StructuralDoF)

    print()
    targets = range(ds.datasetmatrix.get_column_count('X'))
    for target in targets:
        print('target', target)
        mb, _, _ = ipcmb_tests.run_IPCMB(ds, target, parameters)
        print('MB({}) = {}'.format(target, mb))



@pytest.mark.demo
def test_ipcmb_efficiency__with_adtree(testfolders, ds_alarm_8e3):
    ds = ds_alarm_8e3
    LLT = 0
    path = testfolders['adtrees'] / (ds.label + '.pickle')
    parameters = ipcmb_tests.make_parameters__adtree(DoFCalculators.StructuralDoF, LLT, None, path, path)
    parameters['ci_test_ad_tree_class'] = mbtk.structures.ADTree.ADTree

    print()
    targets = range(ds.datasetmatrix.get_column_count('X'))
    for target in targets:
        print('target', target)
        mb, _, ipcmb = ipcmb_tests.run_IPCMB(ds, target, parameters)
        print('MB({}) = {}'.format(target, mb))

        adtree = ipcmb.CITest.AD_tree
        parameters['ci_test_ad_tree_preloaded'] = adtree



@pytest.mark.demo
@pytest.mark.demo_alarm
def test_ipcmb_efficiency__with_dynamic_adtree(testfolders, ds_alarm_8e3):
    run_demo_ipcmb_test__dynamic_adtree(testfolders, ds_alarm_8e3)



@pytest.mark.demo
@pytest.mark.demo_alarm
def test_ipcmb_efficiency__with_dcMI(testfolders, ds_alarm_8e3):
    run_demo_ipcmb_test__dcmi(testfolders, ds_alarm_8e3)



@pytest.mark.demo
@pytest.mark.demo_pathfinder
def test_ipcmb_efficiency__with_dynamic_adtree__pathfinder(testfolders, ds_pathfinder_4e3):
    print('Demo test with a dynamic AD-tree on Pathfinder')
    run_demo_ipcmb_test__dynamic_adtree(testfolders, ds_pathfinder_4e3)



@pytest.mark.demo
@pytest.mark.demo_pathfinder
def test_ipcmb_efficiency__with_dcMI__pathfinder(testfolders, ds_pathfinder_4e3):
    print('Demo test with dcMI on Pathfinder')
    run_demo_ipcmb_test__dcmi(testfolders, ds_pathfinder_4e3)



@pytest.mark.demo
@pytest.mark.demo_andes
def test_ipcmb_efficiency__with_dynamic_adtree__andes(testfolders, ds_andes_4e3):
    print('Demo test with a dynamic AD-tree on Andes')
    run_demo_ipcmb_test__dynamic_adtree(testfolders, ds_andes_4e3)



@pytest.mark.demo
@pytest.mark.demo_andes
def test_ipcmb_efficiency__with_dcMI__andes(testfolders, ds_andes_4e3):
    print('Demo test with a dcMI on Andes')
    run_demo_ipcmb_test__dcmi(testfolders, ds_andes_4e3)



def run_demo_ipcmb_test__dynamic_adtree(folders, ds):
    LLT = 0
    path = folders['dynamic_adtrees'] / (ds.label + '.pickle')
    parameters = ipcmb_tests.make_parameters__adtree(DoFCalculators.StructuralDoF, LLT, None, path, path)
    parameters['ci_test_ad_tree_class'] = mbtk.structures.DynamicADTree.DynamicADTree
    parameters['source_bayesian_network'] = None
    parameters['algorithm_debug'] = 1

    print()
    print()
    gc_total_duration = 0
    ipcmb_total_duration = 0
    targets = range(ds.datasetmatrix.get_column_count('X'))
    print('DS has {} targets.'.format(targets))
    start = time.time()
    print('Start time', time.ctime(start), flush=True)
    for target in targets:
        print()
        print('target', target)
        target_start = time.time()
        mb, _, ipcmb = ipcmb_tests.run_IPCMB(ds, target, parameters)
        target_duration = time.time() - target_start
        ipcmb_total_duration += target_duration
        print('[{:>}]\tMB({}) = {}'.format(target_duration, target, mb))

        adtree = ipcmb.CITest.AD_tree
        parameters['ci_test_ad_tree_preloaded'] = adtree

        print('gc.collect()... ', end='')
        gc_start = time.time()
        gc.collect()
        gc_duration = time.time() - gc_start
        print('ok, took {}s'.format(gc_duration))
        gc_total_duration += gc_duration

    end = time.time()
    print('End time', time.ctime(end))
    print('Duration', end - start)
    print('IPC-MB duration', ipcmb_total_duration)
    print('GC duration', gc_total_duration)



def run_demo_ipcmb_test__dcmi(folders, ds):
    jht_path = folders['jht'] / 'jht_{}.pickle'.format(ds.label)
    dof_path = folders['dofCache'] / 'dof_cache_{}.pickle'.format(ds.label)
    parameters = ipcmb_tests.make_parameters__dcmi(DoFCalculators.CachedStructuralDoF, jht_path, dof_path)
    parameters['source_bayesian_network'] = None
    parameters['algorithm_debug'] = 1

    print()
    print()
    targets = range(ds.datasetmatrix.get_column_count('X'))
    print('DS has {} targets.'.format(targets))
    gc_total_duration = 0
    ipcmb_total_duration = 0
    start = time.time()
    print('Start time', time.ctime(start))
    for target in targets:
        print()
        print('target', target)
        target_start = time.time()
        mb, _, ipcmb = ipcmb_tests.run_IPCMB(ds, target, parameters)
        target_end = time.time()
        target_duration = target_end - target_start
        ipcmb_total_duration += target_duration
        print('[{:>}]\tMB({}) = {}'.format(target_duration, target, mb))

        jht = ipcmb.CITest.JHT
        parameters['ci_test_jht_preloaded'] = jht

        print('gc.collect()... ', end='')
        gc_start = time.time()
        gc.collect()
        gc_duration = time.time() - gc_start
        print('ok, took {}s'.format(gc_duration))
        gc_total_duration += gc_duration

    end = time.time()
    print('End time', time.ctime(end))
    print('Duration', end - start)
    print('IPC-MB duration', ipcmb_total_duration)
    print('GC duration', gc_total_duration)
