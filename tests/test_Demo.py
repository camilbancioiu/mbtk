import pytest
import time
import gc
import mbtk.math.DoFCalculators as DoFCalculators

import mbtk.structures.DynamicADTree
import tests.test_AlgorithmIPCMBWithGtests as ipcmb_tests
import tests.utilities as testutil


@pytest.fixture(scope='session')
def demonstration_state():
    state = dict()
    state['durations'] = dict()
    return state



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



def test_ipcmb_efficiency__unoptimized(ds_alarm_8e3):
    ds = ds_alarm_8e3
    parameters = ipcmb_tests.make_parameters__unoptimized(DoFCalculators.StructuralDoF)

    print()
    targets = range(ds.datasetmatrix.get_column_count('X'))
    for target in targets:
        print('target', target)
        mb, _, _ = ipcmb_tests.run_IPCMB(ds, target, parameters)
        print('MB({}) = {}'.format(target, mb))



def test_ipcmb_efficiency__with_static_adtree(testfolders, ds_alarm_8e3):
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

        # Reuse the AD-tree when discovering the MB of the next target
        adtree = ipcmb.CITest.AD_tree
        parameters['ci_test_ad_tree_preloaded'] = adtree



DEMO_DYN_ADTREE_HEADER = """
=======================================================================
Starting demonstrative experiment run using IPC-MB optimized with a dynamic AD-tree."""

@pytest.mark.demo
def test_ipcmb_efficiency__with_dynamic_adtree(testfolders, ds_alarm_8e3, demonstration_state):
    print(DEMO_DYN_ADTREE_HEADER)
    run_demo_ipcmb_test__dynamic_adtree(testfolders, ds_alarm_8e3)



DEMO_DCMI_HEADER = """
=======================================================================
Starting demonstrative experiment run using IPC-MB optimized with dcMI."""

@pytest.mark.demo
def test_ipcmb_efficiency__with_dcMI(testfolders, ds_alarm_8e3, demonstration_state):
    print()
    print(DEMO_DCMI_HEADER)
    run_demo_ipcmb_test__dcmi(testfolders, ds_alarm_8e3)



def test_ipcmb_efficiency__with_dynamic_adtree__pathfinder(testfolders, ds_pathfinder_4e3):
    print('Demo test with a dynamic AD-tree on Pathfinder')
    run_demo_ipcmb_test__dynamic_adtree(testfolders, ds_pathfinder_4e3)



def test_ipcmb_efficiency__with_dcMI__pathfinder(testfolders, ds_pathfinder_4e3):
    print('Demo test with dcMI on Pathfinder')
    run_demo_ipcmb_test__dcmi(testfolders, ds_pathfinder_4e3)



def test_ipcmb_efficiency__with_dynamic_adtree__andes(testfolders, ds_andes_4e3):
    print('Demo test with a dynamic AD-tree on Andes')
    run_demo_ipcmb_test__dynamic_adtree(testfolders, ds_andes_4e3)



def test_ipcmb_efficiency__with_dcMI__andes(testfolders, ds_andes_4e3):
    print('Demo test with a dcMI on Andes')
    run_demo_ipcmb_test__dcmi(testfolders, ds_andes_4e3)
