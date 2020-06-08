import gc

import tests.utilities as testutil
import pytest

from mbff.algorithms.mb.ipcmb import AlgorithmIPCMB
import mbff.math.G_test__unoptimized
import mbff.math.G_test__with_AD_tree
import mbff.math.debug.G_test__with_AD_tree__debug
import mbff.structures.ADTree
import mbff.structures.DynamicADTree
import mbff.math.G_test__with_dcMI
import mbff.math.DSeparationCITest
import mbff.math.DoFCalculators as DoFCalculators
import time

DebugLevel = 0
CITestDebugLevel = 0
CITestSignificance = 0.8


@pytest.fixture(scope='session')
def testfolders():
    folders = dict()
    root = testutil.ensure_empty_tmp_subfolder('test_ipcmb_with_gtests')

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
    parameters = make_parameters__unoptimized(DoFCalculators.StructuralDoF)

    print()
    targets = range(ds.datasetmatrix.get_column_count('X'))
    for target in targets:
        print('target', target)
        mb, _, _ = run_IPCMB(ds, target, parameters)
        print('MB({}) = {}'.format(target, mb))



@pytest.mark.demo
def test_ipcmb_efficiency__with_dcMI(testfolders, ds_alarm_8e3):
    ds = ds_alarm_8e3
    jht_path = testfolders['jht'] / 'jht_{}.pickle'.format(ds.label)
    dof_path = testfolders['dofCache'] / 'dof_cache_{}.pickle'.format(ds.label)
    parameters = make_parameters__dcmi(DoFCalculators.CachedStructuralDoF, jht_path, dof_path)

    print()
    targets = range(ds.datasetmatrix.get_column_count('X'))
    for target in targets:
        print('target', target)
        mb, _, ipcmb = run_IPCMB(ds, target, parameters)
        print('MB({}) = {}'.format(target, mb))

        jht = ipcmb.CITest.JHT
        parameters['ci_test_jht_preloaded'] = jht



@pytest.mark.demo_andes
def test_ipcmb_efficiency__with_dcMI__andes(testfolders, ds_andes_4e3):
    ds = ds_andes_4e3
    jht_path = testfolders['jht'] / 'jht_{}.pickle'.format(ds.label)
    dof_path = testfolders['dofCache'] / 'dof_cache_{}.pickle'.format(ds.label)
    parameters = make_parameters__dcmi(DoFCalculators.CachedStructuralDoF, jht_path, dof_path)
    parameters['source_bayesian_network'] = None
    parameters['algorithm_debug'] = 1

    print()
    targets = range(ds.datasetmatrix.get_column_count('X'))
    start = time.time()
    print('Start time', start)
    for target in targets:
        print('target', target)
        mb, _, ipcmb = run_IPCMB(ds, target, parameters)
        print('MB({}) = {}'.format(target, mb))

        jht = ipcmb.CITest.JHT
        parameters['ci_test_jht_preloaded'] = jht
    end = time.time()
    print('End time', end)
    print('Duration', end - time)



@pytest.mark.demo
def test_ipcmb_efficiency__with_dynamic_adtree(testfolders, ds_alarm_8e3):
    ds = ds_alarm_8e3
    LLT = 0
    path = testfolders['dynamic_adtrees'] / (ds.label + '.pickle')
    parameters = make_parameters__adtree(DoFCalculators.StructuralDoF, LLT, None, path, path)
    parameters['ci_test_ad_tree_class'] = mbff.structures.DynamicADTree.DynamicADTree

    print()
    targets = range(ds.datasetmatrix.get_column_count('X'))
    for target in targets:
        print('target', target)
        mb, _, ipcmb = run_IPCMB(ds, target, parameters)
        print('MB({}) = {}'.format(target, mb))

        adtree = ipcmb.CITest.AD_tree
        parameters['ci_test_ad_tree_preloaded'] = adtree



@pytest.mark.demo_andes
def test_ipcmb_efficiency__with_dynamic_adtree__andes(testfolders, ds_andes_4e3):
    ds = ds_andes_4e3
    LLT = 0
    path = testfolders['dynamic_adtrees'] / (ds.label + '.pickle')
    parameters = make_parameters__adtree(DoFCalculators.StructuralDoF, LLT, None, path, path)
    parameters['ci_test_ad_tree_class'] = mbff.structures.DynamicADTree.DynamicADTree

    print()
    targets = range(ds.datasetmatrix.get_column_count('X'))
    start = time.time()
    print('Start time', start)
    for target in targets:
        print('target', target)
        mb, _, ipcmb = run_IPCMB(ds, target, parameters)
        print('MB({}) = {}'.format(target, mb))

        adtree = ipcmb.CITest.AD_tree
        parameters['ci_test_ad_tree_preloaded'] = adtree
    end = time.time()
    print('End time', end)
    print('Duration', end - time)



@pytest.mark.demo
def test_ipcmb_efficiency__with_adtree(testfolders, ds_alarm_8e3):
    ds = ds_alarm_8e3
    LLT = 0
    path = testfolders['adtrees'] / (ds.label + '.pickle')
    parameters = make_parameters__adtree(DoFCalculators.StructuralDoF, LLT, None, path, path)
    parameters['ci_test_ad_tree_class'] = mbff.structures.ADTree.ADTree

    print()
    targets = range(ds.datasetmatrix.get_column_count('X'))
    for target in targets:
        print('target', target)
        mb, _, ipcmb = run_IPCMB(ds, target, parameters)
        print('MB({}) = {}'.format(target, mb))

        adtree = ipcmb.CITest.AD_tree
        parameters['ci_test_ad_tree_preloaded'] = adtree



@pytest.mark.slow
def test_ipcmb_correctness__unoptimized(ds_lc_repaired_8e3):
    """
    This test ensures that IPC-MB correctly finds all the MBs
    in the repaired LUNGCANCER dataset when using the unoptimized G-test.
    """
    ds = ds_lc_repaired_8e3
    parameters = make_parameters__unoptimized(DoFCalculators.StructuralDoF)
    parameters_dsep = make_parameters__dsep()

    targets = range(ds.datasetmatrix.get_column_count('X'))
    for target in targets:
        mb, _, _ = run_IPCMB(ds, target, parameters)
        mb_dsep, _, _ = run_IPCMB(ds, target, parameters_dsep)
        assert mb == mb_dsep



@pytest.mark.slow
def test_ipcmb_correctness__adtree(ds_lc_repaired_8e3, adtree_lc_repaired_8e3_llta200):
    """
    This test ensures that IPC-MB correctly finds all the MBs
    in the repaired LUNGCANCER dataset when using the G-test with AD-tree.
    """
    ds = ds_lc_repaired_8e3
    adtree = adtree_lc_repaired_8e3_llta200
    LLT = 200
    parameters = make_parameters__adtree(DoFCalculators.StructuralDoF, LLT, adtree)
    parameters['ci_test_ad_tree_class'] = mbff.structures.ADTree.ADTree
    parameters_dsep = make_parameters__dsep()

    targets = range(ds.datasetmatrix.get_column_count('X'))
    for target in targets:
        mb, _, ipcmb = run_IPCMB(ds, target, parameters)
        mb_dsep, _, _ = run_IPCMB(ds, target, parameters_dsep)
        assert mb == mb_dsep

        adtree = ipcmb.CITest.AD_tree
        parameters['ci_test_ad_tree_preloaded'] = adtree



@pytest.mark.slow
def test_ipcmb_correctness__dynamic_adtree(ds_lc_repaired_8e3, testfolders):
    """
    This test ensures that IPC-MB correctly finds all the MBs
    in the repaired LUNGCANCER dataset when using the G-test with AD-tree.
    """
    ds = ds_lc_repaired_8e3
    LLT = 0
    path = testfolders['dynamic_adtrees'] / (ds.label + '.pickle')
    parameters = make_parameters__adtree(DoFCalculators.StructuralDoF, LLT, None, path, path)
    parameters['ci_test_ad_tree_class'] = mbff.structures.DynamicADTree.DynamicADTree
    parameters_dsep = make_parameters__dsep()

    targets = range(ds.datasetmatrix.get_column_count('X'))
    for target in targets:
        mb, _, ipcmb = run_IPCMB(ds, target, parameters)
        mb_dsep, _, _ = run_IPCMB(ds, target, parameters_dsep)
        assert mb == mb_dsep

        adtree = ipcmb.CITest.AD_tree
        parameters['ci_test_ad_tree_preloaded'] = adtree



def test_ipcmb_correctness__dcmi(ds_lc_repaired_8e3, testfolders):
    """
    This test ensures that IPC-MB correctly finds all the MBs
    in the repaired LUNGCANCER dataset when using the G-test with dcMI.
    """
    ds = ds_lc_repaired_8e3
    jht_path = testfolders['jht'] / 'jht_{}.pickle'.format(ds.label)
    dof_path = testfolders['dofCache'] / 'dof_cache_{}.pickle'.format(ds.label)
    parameters = make_parameters__dcmi(DoFCalculators.CachedStructuralDoF, jht_path, dof_path)
    parameters_dsep = make_parameters__dsep()

    targets = range(ds.datasetmatrix.get_column_count('X'))
    for target in targets:
        mb, _, _ = run_IPCMB(ds, target, parameters)
        mb_dsep, _, _ = run_IPCMB(ds, target, parameters_dsep)
        assert mb == mb_dsep



@pytest.mark.slow
def test_dof_computation_methods__dcmi_vs_adtree(ds_survey_2e3, adtree_survey_2e3_llta0, testfolders):
    """
    This test ensures that IPC-MB produces the same results when run with:
        * an AD-tree with StructuralDoF
        * dcMI with CachedStructuralDoF
    """
    ds = ds_survey_2e3
    adtree = adtree_survey_2e3_llta0
    LLT = 0

    jht_path = testfolders['jht'] / 'jht_{}.pickle'.format(ds.label)
    dof_path = testfolders['dofCache'] / 'dof_cache_{}.pickle'.format(ds.label)
    parameters_dcmi = make_parameters__dcmi(DoFCalculators.CachedStructuralDoF, jht_path, dof_path)
    parameters_adtree = make_parameters__adtree(DoFCalculators.StructuralDoF, LLT, adtree)

    targets = range(ds.datasetmatrix.get_column_count('X'))
    for target in targets:
        mb_dcmi, ci_tests_dcmi, _ = run_IPCMB(ds, target, parameters_dcmi)
        mb_adtree, ci_tests_adtree, _ = run_IPCMB(ds, target, parameters_adtree)

        assertEqualCITestResults(ci_tests_adtree, ci_tests_dcmi)
        assert mb_dcmi == mb_adtree



@pytest.mark.slow
def test_CachedStructuralDoF_vs_StructuralDof_on_G_test__unoptimized(ds_survey_2e3):
    """
    This test ensures that CachedStructuralDoF functions exactly like
    StructuralDoF, on the unoptimized G-test.
    """
    ds = ds_survey_2e3

    parameters_sdof = make_parameters__unoptimized(DoFCalculators.StructuralDoF)
    parameters_csdof = make_parameters__unoptimized(DoFCalculators.CachedStructuralDoF)

    targets = range(ds.datasetmatrix.get_column_count('X'))
    for target in targets:
        mb_sdof, results_sdof, _ = run_IPCMB(ds, target, parameters_sdof)
        mb_csdof, results_csdof, _ = run_IPCMB(ds, target, parameters_csdof)

        assertEqualCITestResults(results_sdof, results_csdof)
        assert mb_sdof == mb_csdof



@pytest.mark.slow
def test_dof_across_Gtest_optimizations__UnadjustedDoF(ds_survey_2e3, adtree_survey_2e3_llta0):
    """
    This test ensures that all 3 implementations of the G-test produce the same
    results, regardless of the correctness of the MB.
    """
    ds = ds_survey_2e3
    adtree = adtree_survey_2e3_llta0
    LLT = 0

    dof = DoFCalculators.UnadjustedDoF

    parameters = dict()
    parameters['G_test__unoptimized'] = make_parameters__unoptimized(dof)
    parameters['G_test__with_AD_tree'] = make_parameters__adtree(dof, LLT, adtree)
    parameters['G_test__with_dcMI'] = make_parameters__dcmi(dof)

    targets = range(ds.datasetmatrix.get_column_count('X'))
    for target in targets:
        validate_IPCMB_across_Gtest_implementations(ds, target, parameters, validate_mb=False, validate_ci_tests=True)



@pytest.mark.slow
def test_dof_across_Gtest_optimizations__StructuralDoF(ds_survey_2e3, adtree_survey_2e3_llta0):
    """This test ensures that StructuralDoF behaves consistently when used by
    the unoptimized G-test versus the G-test with AD-tree."""
    ds = ds_survey_2e3
    adtree = adtree_survey_2e3_llta0
    LLT = 0

    dof = DoFCalculators.StructuralDoF

    parameters = dict()
    parameters['G_test__unoptimized'] = make_parameters__unoptimized(dof)
    parameters['G_test__with_AD_tree'] = make_parameters__adtree(dof, LLT, adtree)

    targets = range(ds.datasetmatrix.get_column_count('X'))
    for target in targets:
        validate_IPCMB_across_Gtest_implementations(ds, target, parameters, validate_mb=False, validate_ci_tests=True)



@pytest.mark.slow
def test_dof_across_Gtest_optimizations__CachedStructuralDoF(ds_survey_2e3, adtree_survey_2e3_llta0):
    ds = ds_survey_2e3
    adtree = adtree_survey_2e3_llta0
    LLT = 0

    dof = DoFCalculators.CachedStructuralDoF

    parameters = dict()
    parameters['G_test__unoptimized'] = make_parameters__unoptimized(dof)
    parameters['G_test__with_AD_tree'] = make_parameters__adtree(dof, LLT, adtree)
    parameters['G_test__with_dcMI'] = make_parameters__dcmi(dof)

    targets = range(ds.datasetmatrix.get_column_count('X'))
    for target in targets:
        validate_IPCMB_across_Gtest_implementations(ds, target, parameters, validate_mb=False, validate_ci_tests=True)



def validate_IPCMB_across_Gtest_implementations(ds, target, parameters_for_implementations, validate_mb=True, validate_ci_tests=True):
    results = dict()
    for implementation, parameters in parameters_for_implementations.items():
        results[implementation] = run_IPCMB(ds, target, parameters)

    if validate_mb:
        ipcmb_dsep = make_IPCMB(ds, target, make_parameters__dsep())
        mb_dsep = ipcmb_dsep.select_features()
        for implementation, result in results.items():
            mb = result[0]
            assert mb == mb_dsep

    if validate_ci_tests:
        expected_ci_tests = None
        for implementation, result in results.items():
            ci_tests = result[1]
            if expected_ci_tests is None:
                expected_ci_tests = ci_tests
            else:
                assertEqualCITestResults(expected_ci_tests, ci_tests)

    gc.collect()



def run_IPCMB(ds, target, parameters):
    ipcmb = make_IPCMB(ds, target, parameters)
    mb = ipcmb.select_features()
    ci_tests = ipcmb.CITest.ci_test_results
    return (mb, ci_tests, ipcmb)



def make_IPCMB(ds, target, extra_parameters=None):
    if extra_parameters is None:
        extra_parameters = dict()

    parameters = dict()
    parameters['target'] = target
    parameters['ci_test_debug'] = CITestDebugLevel
    parameters['ci_test_significance'] = CITestSignificance
    parameters['ci_test_results__print_accurate'] = False
    parameters['ci_test_results__print_inaccurate'] = True
    parameters['algorithm_debug'] = DebugLevel
    parameters['omega'] = ds.omega
    parameters['source_bayesian_network'] = ds.bayesiannetwork

    parameters.update(extra_parameters)
    ipcmb = AlgorithmIPCMB(ds.datasetmatrix, parameters)
    return ipcmb



def make_parameters__unoptimized(dof_class):
    parameters = dict()
    parameters['ci_test_class'] = mbff.math.G_test__unoptimized.G_test
    parameters['ci_test_dof_calculator_class'] = dof_class
    parameters['ci_test_gc_collect_rate'] = 0
    return parameters



def make_parameters__adtree(dof_class, llt, adtree=None, path_load=None, path_save=None):
    parameters = dict()
    parameters['ci_test_class'] = mbff.math.G_test__with_AD_tree.G_test
    parameters['ci_test_dof_calculator_class'] = dof_class
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



def make_parameters__dsep():
    parameters = dict()
    parameters['ci_test_class'] = mbff.math.DSeparationCITest.DSeparationCITest
    parameters['ci_test_debug'] = 0
    parameters['algorithm_debug'] = 0
    return parameters



def assertEqualCITestResults(expected_results, obtained_results):
    for (expected_result, obtained_result) in zip(expected_results, obtained_results):
        expected_result.tolerance__statistic_value = 1e-8
        expected_result.tolerance__p_value = 1e-7
        # failMessage = (
        #     'Differing CI test results:\n'
        #     'REFERENCE: {}\n'
        #     'COMPUTED:  {}\n'
        #     '{}\n'
        # ).format(expected_result, obtained_result, expected_result.diff(obtained_result))
        try:
            assert expected_result == obtained_result
        except AssertionError:
            print('expected:')
            print(expected_result)
            print('obtained:')
            print(obtained_result)
            print('diff:')
            print(obtained_result.diff(expected_result))
            raise




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
