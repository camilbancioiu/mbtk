import gc
import time

import mbff_test.utilities as testutil
import pytest

from mbff.algorithms.mb.ipcmb import AlgorithmIPCMB
import mbff.math.G_test__unoptimized
import mbff.math.G_test__with_AD_tree
import mbff.math.G_test__with_dcMI
import mbff.math.DSeparationCITest
import mbff.math.DoFCalculators as DoFCalculators

DebugLevel = 0
CITestDebugLevel = 0


@pytest.fixture(scope='session')
def testfolders():
    folders = dict()
    root = testutil.ensure_empty_tmp_subfolder('test_ipcmb_with_gtests')

    subfolders = ['dofCache', 'adtrees', 'ci_test_results', 'jht']
    for subfolder in subfolders:
        path = root / subfolder
        path.mkdir(parents=True, exist_ok=True)
        subfolders[subfolder] = path

    folders['root'] = root
    return folders


# @unittest.skipIf(TestBase.tag_excluded('ipcmb_run'), 'Tests running IPC-MB are excluded')
def test_dof_computation_methods__dcmi_vs_adtree(ds_alarm_3e3, adtree_alarm_3e3_llta100, testfolders):
    """
    This test ensures that IPC-MB produces the same results when run with:
        * an AD-tree with StructuralDoF
        * dcMI with CachedStructuralDoF
    """
    ds = ds_alarm_3e3
    adtree = adtree_alarm_3e3_llta100
    significance = 0.9
    LLT = 0

    dof_path = testfolders['dofCache'] / 'dof_cache_{}.pickle'.format(ds.label)
    parameters_dcmi = make_extra_parameters__dcmi(DoFCalculators.CachedStructuralDoF, None, dof_path)
    parameters_adtree = make_extra_parameters__adtree(DoFCalculators.StructuralDoF, LLT, adtree)

    targets = range(ds.datasetmatrix.get_column_count('X'))
    for target in targets:
        ipcmb = make_IPCMB_with_Gtest_dcMI(ds, testfolders, target, significance, parameters_dcmi)
        mb_dcmi = ipcmb.select_features()
        results_dcmi = ipcmb.CITest.ci_test_results

        ipcmb = make_IPCMB_with_Gtest_ADtree(ds, testfolders, target, significance, parameters_adtree)
        mb_adtree = ipcmb.select_features()
        results_adtree = ipcmb.CITest.ci_test_results

        assertEqualCITestResults(results_adtree, results_dcmi)
        assert mb_dcmi == mb_adtree



def test_CachedStructuralDoF_vs_StructuralDof_on_G_test__unoptimized(ds_alarm_3e3):
    """This test ensures that CachedStructuralDoF functions exactly like
    StructuralDoF, on the unoptimized G-test"""
    ds = ds_alarm_3e3
    significance = 0.9

    parameters_sdof = make_extra_parameters__unoptimized(DoFCalculators.StructuralDoF)
    parameters_csdof = make_extra_parameters__unoptimized(DoFCalculators.CachedStructuralDoF)

    targets = range(ds.datasetmatrix.get_column_count('X'))
    for target in targets:
        ipcmb_g_unoptimized = make_IPCMB_with_Gtest_unoptimized(ds, target, significance, extra_parameters=parameters_sdof)
        mb_unoptimized_StructuralDoF = ipcmb_g_unoptimized.select_features()
        ci_test_results__unoptimized__StructuralDoF = ipcmb_g_unoptimized.CITest.ci_test_results

        ipcmb_g_unoptimized = make_IPCMB_with_Gtest_unoptimized(ds, target, significance, extra_parameters=parameters_csdof)
        mb_unoptimized_CachedStructuralDoF = ipcmb_g_unoptimized.select_features()
        ci_test_results__unoptimized__CachedStructuralDoF = ipcmb_g_unoptimized.CITest.ci_test_results

        assertEqualCITestResults(ci_test_results__unoptimized__StructuralDoF, ci_test_results__unoptimized__CachedStructuralDoF)
        assert mb_unoptimized_StructuralDoF == mb_unoptimized_CachedStructuralDoF



def test_dof_across_Gtest_optimizations__UnadjustedDoF(ds_alarm_3e3, adtree_alarm_3e3_llta100):
    ds = ds_alarm_3e3
    adtree = adtree_alarm_3e3_llta100
    significance = 0.9
    LLT = 0

    dof = DoFCalculators.UnadjustedDoF

    if DebugLevel > 0: print()

    parameters = dict()

    parameters['G_test__unoptimized'] = make_extra_parameters__unoptimized(dof)
    parameters['G_test__with_AD_tree'] = make_extra_parameters__adtree(dof, LLT, adtree)
    parameters['G_test__with_dcMI'] = make_extra_parameters__dcmi(dof)

    for target in [3]:
        run_test_dof_across_Gtest_implementations(ds, target, significance, parameters)



def test_dof_across_Gtest_optimizations__StructuralDoF(ds_alarm_3e3, adtree_alarm_3e3_llta100):
    """This test ensures that StructuralDoF behaves consistently when used by
    the unoptimized G-test versus the G-test with AD-tree."""
    ds = ds_alarm_3e3
    adtree = adtree_alarm_3e3_llta100
    significance = 0.9
    LLT = 0

    dof = DoFCalculators.StructuralDoF

    parameters = dict()
    parameters['G_test__unoptimized'] = make_extra_parameters__unoptimized(dof)
    parameters['G_test__with_AD_tree'] = make_extra_parameters__adtree(dof, LLT, adtree)

    targets = range(ds.datasetmatrix.get_column_count('X'))
    for target in targets:
        run_test_dof_across_Gtest_implementations(ds, target, significance, parameters)



def test_dof_across_Gtest_optimizations__CachedStructuralDoF(ds_alarm_3e3, adtree_alarm_3e3_llta100):
    ds = ds_alarm_3e3
    adtree = adtree_alarm_3e3_llta100
    significance = 0.9
    LLT = 0

    dof = DoFCalculators.CachedStructuralDoF

    parameters = dict()
    parameters['G_test__unoptimized'] = make_extra_parameters__unoptimized(dof)
    parameters['G_test__with_AD_tree'] = make_extra_parameters__adtree(dof, LLT, adtree)
    parameters['G_test__with_dcMI'] = make_extra_parameters__dcmi(dof)

    for target in [19]:
        run_test_dof_across_Gtest_implementations(ds, target, significance, parameters)



def run_test_dof_across_Gtest_implementations(ds, target, significance, parameters):
    ci_test_results__unoptimized = None
    ci_test_results__adtree = None
    ci_test_results__dcmi = None

    if 'G_test__unoptimized' in parameters:
        if DebugLevel > 0: print('=== IPC-MB with G-test (unoptimized) ===')
        extra_parameters = parameters['G_test__unoptimized']
        ipcmb_g_unoptimized = make_IPCMB_with_Gtest_unoptimized(ds, target, significance, extra_parameters=extra_parameters)
        start_time = time.time()
        markov_blanket__unoptimized = ipcmb_g_unoptimized.select_features()
        duration__unoptimized = time.time() - start_time
        ci_test_results__unoptimized = ipcmb_g_unoptimized.CITest.ci_test_results
        if DebugLevel > 0: print(format_ipcmb_result('unoptimized', target, ds, markov_blanket__unoptimized))
        if DebugLevel > 0: print()

    if 'G_test__with_dcMI' in parameters:
        if DebugLevel > 0: print('=== IPC-MB with G-test (dcMI) ===')
        extra_parameters = parameters['G_test__with_dcMI']
        ipcmb_g_dcmi = make_IPCMB_with_Gtest_dcMI(ds, target, significance, extra_parameters=extra_parameters)
        start_time = time.time()
        markov_blanket__dcmi = ipcmb_g_dcmi.select_features()
        duration__dcmi = time.time() - start_time
        ci_test_results__dcmi = ipcmb_g_dcmi.CITest.ci_test_results
        if DebugLevel > 0: print(format_ipcmb_result('dcMI', target, ds, markov_blanket__dcmi))
        if ci_test_results__unoptimized is not None:
            assertEqualCITestResults(ci_test_results__unoptimized, ci_test_results__dcmi)
            if DebugLevel > 0: print()

    if 'G_test__with_AD_tree' in parameters:
        if DebugLevel > 0: print('=== IPC-MB with G-test (AD-tree) ===')
        extra_parameters = parameters['G_test__with_AD_tree']
        ipcmb_g_adtree = make_IPCMB_with_Gtest_ADtree(ds, target, significance, extra_parameters=extra_parameters)
        start_time = time.time()
        markov_blanket__adtree = ipcmb_g_adtree.select_features()
        duration__adtree = time.time() - start_time
        ci_test_results__adtree = ipcmb_g_adtree.CITest.ci_test_results
        if DebugLevel > 0: print(format_ipcmb_result('AD-tree', target, ds, markov_blanket__adtree))
        if ci_test_results__unoptimized is not None:
            assertEqualCITestResults(ci_test_results__unoptimized, ci_test_results__adtree)
            if DebugLevel > 0: print()

    if DebugLevel > 0: print('=== IPC-MB with d-sep ===')
    ipcmb_dsep = make_IPCMB_with_dsep(ds, target)
    markov_blanket__dsep = ipcmb_dsep.select_features()
    if DebugLevel > 0: print(format_ipcmb_result('dsep', target, ds, markov_blanket__dsep))

    gc.collect()

    if DebugLevel > 0: print()

    if 'G_test__with_dcMI' in parameters:
        mb_correctness = 'CORRECT'
        if markov_blanket__dsep != markov_blanket__dcmi:
            mb_correctness = 'WRONG'
        if DebugLevel > 0: print('MB, dcmi ({}): {}, duration {:<.2f}'.format(mb_correctness, markov_blanket__dcmi, duration__dcmi))

    if 'G_test__with_AD_tree' in parameters:
        mb_correctness = 'CORRECT'
        if markov_blanket__dsep != markov_blanket__adtree:
            mb_correctness = 'WRONG'
        if DebugLevel > 0: print('MB, adtree ({}): {}, duration {:<.2f}'.format(mb_correctness, markov_blanket__adtree, duration__adtree))

    if 'G_test__unoptimized' in parameters:
        mb_correctness = 'CORRECT'
        if markov_blanket__dsep != markov_blanket__unoptimized:
            mb_correctness = 'WRONG'
        if DebugLevel > 0: print('MB, unoptimized ({}): {}, duration {:<.2f}'.format(mb_correctness, markov_blanket__unoptimized, duration__unoptimized))

    if DebugLevel > 0: print('MB, d-sep : {}'.format(markov_blanket__dsep))
    if DebugLevel > 0: print()



def make_IPCMB_with_dsep(self, dm_label, target, extra_parameters=None):
    if extra_parameters is None:
        extra_parameters = dict()

    ci_test_class = mbff.math.DSeparationCITest.DSeparationCITest

    parameters = dict()
    parameters['ci_test_debug'] = 0
    parameters['algorithm_debug'] = 0
    parameters.update(extra_parameters)

    ipcmb = self.make_IPCMB(dm_label, target, ci_test_class, None, parameters)
    return ipcmb



def make_IPCMB_with_Gtest_dcMI(ds, testfolders, target, significance, extra_parameters=None):
    if extra_parameters is None:
        extra_parameters = dict()

    ci_test_class = mbff.math.G_test__with_dcMI.G_test

    JHT_path = testfolders['jht'] / (ds.label + '.pickle')
    parameters = dict()
    parameters['ci_test_jht_path__load'] = JHT_path
    parameters['ci_test_jht_path__save'] = JHT_path
    parameters.update(extra_parameters)

    ipcmb = make_IPCMB(ds, target, ci_test_class, significance, parameters)
    return ipcmb



def make_IPCMB_with_Gtest_ADtree(ds, testfolders, dm_label, target, significance, extra_parameters=None):
    if extra_parameters is None:
        extra_parameters = dict()

    ci_test_class = mbff.math.G_test__with_AD_tree.G_test

    LLT = extra_parameters['ci_test_ad_tree_leaf_list_threshold']
    ADTree_path = testfolders['adtree'] / ('{}_LLT={}.pickle'.format(ds.label, LLT))
    parameters = dict()
    parameters['ci_test_ad_tree_path__load'] = ADTree_path
    parameters['ci_test_ad_tree_path__save'] = ADTree_path
    parameters.update(extra_parameters)

    ipcmb = make_IPCMB(ds, target, ci_test_class, significance, parameters)
    return ipcmb



def make_IPCMB_with_Gtest_unoptimized(ds, target, significance, extra_parameters=None):
    if extra_parameters is None:
        extra_parameters = dict()

    ci_test_class = mbff.math.G_test__unoptimized.G_test
    ipcmb = make_IPCMB(ds, target, ci_test_class, significance, extra_parameters)

    return ipcmb



def make_IPCMB(ds, target, ci_test_class, significance, extra_parameters=None):
    if extra_parameters is None:
        extra_parameters = dict()

    parameters = dict()
    parameters['target'] = target
    parameters['ci_test_class'] = ci_test_class
    parameters['ci_test_debug'] = CITestDebugLevel
    parameters['ci_test_significance'] = significance
    parameters['ci_test_results__print_accurate'] = False
    parameters['ci_test_results__print_inaccurate'] = True
    parameters['algorithm_debug'] = DebugLevel
    parameters['omega'] = ds.omega
    parameters['source_bayesian_network'] = ds.bayesiannetwork

    parameters.update(extra_parameters)
    ipcmb = AlgorithmIPCMB(ds.datasetmatrix, parameters)
    return ipcmb


def assertEqualCITestResults(expected_results, obtained_results):
    for (expected_result, obtained_result) in zip(expected_results, obtained_results):
        expected_result.tolerance__statistic_value = 1e-8
        expected_result.tolerance__p_value = 1e-9
        # failMessage = (
        #     'Differing CI test results:\n'
        #     'REFERENCE: {}\n'
        #     'COMPUTED:  {}\n'
        #     '{}\n'
        # ).format(expected_result, obtained_result, expected_result.diff(obtained_result))
        assert expected_result == obtained_result



def make_extra_parameters__unoptimized(dof_class):
    parameters = dict()
    parameters['ci_test_dof_calculator_class'] = dof_class
    parameters['ci_test_gc_collect_rate'] = 0
    return parameters



def make_extra_parameters__adtree(dof_class, llt, adtree=None, path_load=None, path_save=None):
    parameters = dict()
    parameters['ci_test_dof_calculator_class'] = dof_class
    parameters['ci_test_gc_collect_rate'] = 0
    parameters['ci_test_ad_tree_preloaded'] = adtree
    parameters['ci_test_ad_tree_path__load'] = path_load
    parameters['ci_test_ad_tree_path__save'] = path_save
    parameters['ci_test_ad_tree_leaf_list_threshold'] = llt
    return parameters



def make_extra_parameters__dcmi(dof_class, jht_path=None, dof_path=None):
    parameters = dict()
    parameters['ci_test_dof_calculator_class'] = dof_class
    parameters['ci_test_gc_collect_rate'] = 0
    parameters['ci_test_jht_path__load'] = jht_path
    parameters['ci_test_jht_path__save'] = jht_path
    parameters['ci_test_dof_calculator_cache_path__load'] = dof_path
    parameters['ci_test_dof_calculator_cache_path__save'] = dof_path
    return parameters


def format_ipcmb_result(label, target, ds, markov_blanket):
    named_markov_blanket = [ds.datasetmatrix.column_labels_X[n] for n in markov_blanket]
    named_target = ds.datasetmatrix.column_labels_X[target]
    output = '{}: {} ({}) → {} ({})'.format(label, target, named_target, markov_blanket, named_markov_blanket)
    return output



def print_ci_test_results(ci_test_results):
    print()
    print('==========')
    print('CI test results:')
    for result in ci_test_results:
        print(result)
    print()
    print('Total: {} CI tests'.format(len(ci_test_results)))
