import pytest
import tests.utilities as testutil

from mbtk.algorithms.mb.iamb import AlgorithmIAMB
import mbtk.math.G_test__unoptimized
import mbtk.math.DoFCalculators as DoFCalculators
import mbtk.math.DSeparationCITest
from mbtk.math.CMICalculator import CMICalculator
from mbtk.math.BNCorrelationEstimator import BNCorrelationEstimator


DebugLevel = 0
CITestDebugLevel = 0
CITestSignificance = 0.8


@pytest.fixture(scope='session')
def testfolders():
    folders = dict()
    root = testutil.ensure_empty_tmp_subfolder('test_iamb_with_gtests')

    subfolders = ['ci_test_results', 'heuristic_results']
    for subfolder in subfolders:
        path = root / subfolder
        path.mkdir(parents=True, exist_ok=True)
        folders[subfolder] = path

    folders['root'] = root
    return folders



@pytest.mark.slow
def test_iamb_timing__unoptimized(ds_alarm_8e3):
    ds = ds_alarm_8e3
    parameters = make_parameters__unoptimized(DoFCalculators.StructuralDoF)
    parameters_dsep = make_parameters__dsep()

    print()

    total_mb_errors = 0

    targets = range(ds.datasetmatrix.get_column_count('X'))
    for target in targets:
        mb, _, _ = run_IAMB(ds, target, parameters)
        mb_dsep, _, _ = run_IAMB(ds, target, parameters_dsep)
        mb_error = set(mb).symmetric_difference(mb_dsep)
        print('mb error', f'({len(mb_error)})', mb_error)
        total_mb_errors += len(mb_error)

    print('Total MB errors', total_mb_errors)


@pytest.mark.slow
def test_iamb_correctness__unoptimized(ds_lc_repaired_8e3):
    """
    This test ensures that IAMB correctly finds all the MBs
    in the repaired LUNGCANCER dataset when using the unoptimized G-test.
    """
    ds = ds_lc_repaired_8e3
    parameters = make_parameters__unoptimized(DoFCalculators.StructuralDoF)
    parameters_dsep = make_parameters__dsep()

    targets = range(ds.datasetmatrix.get_column_count('X'))
    for target in targets:
        mb, _, _ = run_IAMB(ds, target, parameters)
        mb_dsep, _, _ = run_IAMB(ds, target, parameters_dsep)
        assert mb == mb_dsep



def run_IAMB(ds, target, parameters):
    iamb = make_IAMB(ds, target, parameters)
    mb = iamb.discover_mb()
    ci_tests = iamb.CITest.ci_test_results
    return (mb, ci_tests, iamb)



def make_IAMB(ds, target, extra_parameters=None):
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
    iamb = AlgorithmIAMB(ds.datasetmatrix, parameters)
    return iamb



def make_parameters__unoptimized(dof_class):
    parameters = dict()
    parameters['ci_test_class'] = mbtk.math.G_test__unoptimized.G_test
    parameters['ci_test_dof_calculator_class'] = dof_class
    parameters['ci_test_gc_collect_rate'] = 0
    parameters['correlation_heuristic_class'] = CMICalculator
    parameters['heuristic_pmf_source'] = 'datasetmatrix'
    return parameters



def make_parameters__dsep():
    parameters = dict()
    parameters['ci_test_class'] = mbtk.math.DSeparationCITest.DSeparationCITest
    parameters['ci_test_debug'] = 0
    parameters['algorithm_debug'] = 0
    parameters['correlation_heuristic_class'] = BNCorrelationEstimator
    return parameters
