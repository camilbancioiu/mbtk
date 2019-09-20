import unittest
import gc
import pickle
import time
from pathlib import Path

from mbff_tests.TestBase import TestBase
from mbff_tests.ADTreeClient import ADTreeClient
from mbff.algorithms.mb.ipcmb import AlgorithmIPCMB
import mbff.math.G_test__unoptimized
import mbff.math.G_test__with_AD_tree
import mbff.math.G_test__with_dcMI
import mbff.math.DSeparationCITest
from mbff.math.DoFCalculators import UnadjustedDoF, StructuralDoF, CachedStructuralDoF

from mbff.math.PMF import PMF, CPMF


@unittest.skipIf(TestBase.tag_excluded('ipcmb_run'), 'Tests running IPC-MB are excluded')
class TestAlgorithmIPCMBWithGtests(TestBase):

    @classmethod
    def initTestResources(testClass):
        super().initTestResources()
        testClass.DatasetsInUse = ['lc_repaired', 'alarm', 'alarm_small']
        testClass.RootFolder = Path('testfiles', 'tmp', 'test_ipcmb_with_gtests')

        testClass.DatasetMatrixFolder = testClass.RootFolder / 'dm'
        testClass.JHTFolder = testClass.RootFolder / 'jht'
        testClass.ADTreesFolder = testClass.RootFolder / 'adtrees'
        testClass.CITestResultsFolder = testClass.RootFolder / 'ci_test_results'
        testClass.DebugLevel = 0
        testClass.CITestDebugLevel = 1
        testClass.DoFCacheFolder = testClass.RootFolder / 'dofcache'


    @classmethod
    def prepareTestResources(testClass):
        super(TestAlgorithmIPCMBWithGtests, testClass).prepareTestResources()
        testClass.ADTreesFolder.mkdir(parents=True, exist_ok=True)
        testClass.JHTFolder.mkdir(parents=True, exist_ok=True)
        testClass.CITestResultsFolder.mkdir(parents=True, exist_ok=True)
        testClass.DoFCacheFolder.mkdir(parents=True, exist_ok=True)


    @classmethod
    def configure_dataset(testClass, dm_label):
        configuration = dict()

        if dm_label == 'lc_repaired':
            configuration['sourcepath'] = Path('testfiles', 'bif_files', 'lc_repaired.bif')
            configuration['sample_count'] = int(4e4)
            configuration['values_as_indices'] = True
            configuration['objectives'] = []
            configuration['method'] = 'exact'

        if dm_label == 'alarm':
            configuration['sourcepath'] = Path('testfiles', 'bif_files', 'alarm.bif')
            configuration['sample_count'] = int(3e3)
            configuration['values_as_indices'] = True
            configuration['objectives'] = []
            configuration['random_seed'] = 1984
            configuration['method'] = 'random'

        if dm_label == 'alarm_small':
            configuration['sourcepath'] = Path('testfiles', 'bif_files', 'alarm.bif')
            configuration['sample_count'] = int(3e2)
            configuration['values_as_indices'] = True
            configuration['objectives'] = []
            configuration['random_seed'] = 1983
            configuration['method'] = 'random'

        return configuration


    def test_dof_computation_methods__misc(self):
        dm_label = 'alarm'
        significance = 0.9
        LLT = 0

        parameters_dcmi = dict()
        parameters_dcmi['ci_test_dof_calculator_class'] = CachedStructuralDoF
        parameters_dcmi['ci_test_dof_calculator_cache_path__load'] = self.DoFCacheFolder / 'dof_cache_{}.pickle'.format(dm_label)
        parameters_dcmi['ci_test_dof_calculator_cache_path__save'] = self.DoFCacheFolder / 'dof_cache_{}.pickle'.format(dm_label)
        parameters_dcmi['ci_test_gc_collect_rate'] = 0

        parameters_adtree = dict()
        parameters_adtree['ci_test_dof_calculator_class'] = StructuralDoF
        parameters_adtree['ci_test_ad_tree_preloaded'] = ADTreeClient('tcp://127.0.0.1:8888')
        parameters_adtree['ci_test_ad_tree_path__load'] = None
        parameters_adtree['ci_test_ad_tree_path__save'] = None
        parameters_adtree['ci_test_ad_tree_leaf_list_threshold'] = LLT
        parameters_adtree['ci_test_gc_collect_rate'] = 0
        parameters_adtree['ci_test_sufficient_samples_criterion'] = None

        for target in range(10):
            ipcmb = self.make_IPCMB_with_Gtest_dcMI(dm_label, target, significance, parameters_dcmi)
            mb = ipcmb.select_features()
            results_dcmi = ipcmb.CITest.ci_test_results
            print(mb)

            ipcmb = self.make_IPCMB_with_Gtest_ADtree(dm_label, target, significance, parameters_adtree)
            mb = ipcmb.select_features()
            results_adtree = ipcmb.CITest.ci_test_results
            print(mb)

            self.assertEqualCITestResults(results_adtree, results_dcmi)



    def test_CachedStructuralDoF_vs_StructuralDof_on_G_test__unoptimized(self):
        dm_label = 'alarm'
        significance = 0.9

        target = 19

        print()
        print('=== IPC-MB with G-test (unoptimized - with StructuralDoF) ===')
        parameters = dict()
        parameters['ci_test_dof_calculator_class'] = StructuralDoF
        parameters['ci_test_gc_collect_rate'] = 0
        ipcmb_g_unoptimized = self.make_IPCMB_with_Gtest_unoptimized(dm_label, target, significance, extra_parameters=parameters)
        mb_unoptimized_StructuralDoF = ipcmb_g_unoptimized.select_features()
        ci_test_results__unoptimized__StructuralDoF = ipcmb_g_unoptimized.CITest.ci_test_results
        print()

        print('=== IPC-MB with G-test (unoptimized - with CachedStructuralDoF) ===')
        parameters = dict()
        parameters['ci_test_dof_calculator_class'] = CachedStructuralDoF
        parameters['ci_test_gc_collect_rate'] = 0
        ipcmb_g_unoptimized = self.make_IPCMB_with_Gtest_unoptimized(dm_label, target, significance, extra_parameters=parameters)
        mb_unoptimized_CachedStructuralDoF = ipcmb_g_unoptimized.select_features()
        ci_test_results__unoptimized__CachedStructuralDoF = ipcmb_g_unoptimized.CITest.ci_test_results
        print()

        self.assertEqualCITestResults(ci_test_results__unoptimized__StructuralDoF, ci_test_results__unoptimized__CachedStructuralDoF)
        self.assertEqual(mb_unoptimized_StructuralDoF, mb_unoptimized_CachedStructuralDoF)


    def test_dof_across_Gtest_optimizations__UnadjustedDoF(self):
        dm_label = 'alarm'
        significance = 0.9
        LLT = 0

        DoF_calculator_class = UnadjustedDoF

        print()

        parameters = dict()

        parameters['G_test__unoptimized'] = dict()
        parameters['G_test__unoptimized']['ci_test_dof_calculator_class'] = DoF_calculator_class
        parameters['G_test__unoptimized']['ci_test_gc_collect_rate'] = 0
        parameters['G_test__unoptimized']['ci_test_sufficient_samples_criterion'] = None

        # ADTree_path = self.ADTreesFolder / ('{}_LLT={}.pickle'.format(dm_label, LLT))
        parameters['G_test__with_AD_tree'] = dict()
        parameters['G_test__with_AD_tree']['ci_test_dof_calculator_class'] = DoF_calculator_class
        parameters['G_test__with_AD_tree']['ci_test_ad_tree_preloaded'] = ADTreeClient('tcp://127.0.0.1:8888')
        # parameters['G_test__with_AD_tree']['ci_test_ad_tree_path__load'] = ADTree_path
        parameters['G_test__with_AD_tree']['ci_test_ad_tree_path__load'] = None
        parameters['G_test__with_AD_tree']['ci_test_ad_tree_path__save'] = None
        parameters['G_test__with_AD_tree']['ci_test_ad_tree_leaf_list_threshold'] = LLT
        parameters['G_test__with_AD_tree']['ci_test_gc_collect_rate'] = 0
        parameters['G_test__with_AD_tree']['ci_test_sufficient_samples_criterion'] = None

        parameters['G_test__with_dcMI'] = dict()
        parameters['G_test__with_dcMI']['ci_test_dof_calculator_class'] = DoF_calculator_class
        parameters['G_test__with_dcMI']['ci_test_gc_collect_rate'] = 0
        parameters['G_test__with_dcMI']['ci_test_jht_path__load'] = None
        parameters['G_test__with_dcMI']['ci_test_jht_path__save'] = None
        parameters['G_test__with_dcMI']['ci_test_sufficient_samples_criterion'] = None

        for target in [3]:
            self.run_test_dof_across_Gtest_optimizations(dm_label, target, significance, parameters)


    def test_dof_across_Gtest_optimizations__StructuralDoF(self):
        dm_label = 'alarm'
        significance = 0.9
        LLT = 0

        DoF_calculator_class = StructuralDoF

        print()

        parameters = dict()

        parameters['G_test__unoptimized'] = dict()
        parameters['G_test__unoptimized']['ci_test_dof_calculator_class'] = DoF_calculator_class
        parameters['G_test__unoptimized']['ci_test_gc_collect_rate'] = 0

        ADTree_path = self.ADTreesFolder / ('{}_LLT={}.pickle'.format(dm_label, LLT))
        parameters['G_test__with_AD_tree'] = dict()
        parameters['G_test__with_AD_tree']['ci_test_dof_calculator_class'] = DoF_calculator_class
        # parameters['G_test__with_AD_tree']['ci_test_ad_tree_preloaded'] = ADTreeClient('tcp://127.0.0.1:8888')
        parameters['G_test__with_AD_tree']['ci_test_ad_tree_path__load'] = ADTree_path
        # parameters['G_test__with_AD_tree']['ci_test_ad_tree_path__load'] = None
        parameters['G_test__with_AD_tree']['ci_test_ad_tree_path__save'] = ADTree_path
        parameters['G_test__with_AD_tree']['ci_test_ad_tree_leaf_list_threshold'] = LLT
        parameters['G_test__with_AD_tree']['ci_test_gc_collect_rate'] = 0

        for target in [3]:
            self.run_test_dof_across_Gtest_optimizations(dm_label, target, significance, parameters)


    def test_dof_across_Gtest_optimizations__CachedStructuralDoF(self):
        dm_label = 'alarm'
        significance = 0.9
        LLT = 0

        DoF_calculator_class = CachedStructuralDoF

        print()

        parameters = dict()

        parameters['G_test__unoptimized'] = dict()
        parameters['G_test__unoptimized']['ci_test_dof_calculator_class'] = DoF_calculator_class
        parameters['G_test__unoptimized']['ci_test_gc_collect_rate'] = 0

        ADTree_path = self.ADTreesFolder / ('{}_LLT={}.pickle'.format(dm_label, LLT))
        parameters['G_test__with_AD_tree'] = dict()
        parameters['G_test__with_AD_tree']['ci_test_dof_calculator_class'] = DoF_calculator_class
        # parameters['G_test__with_AD_tree']['ci_test_ad_tree_preloaded'] = ADTreeClient('tcp://127.0.0.1:8888')
        parameters['G_test__with_AD_tree']['ci_test_ad_tree_path__load'] = ADTree_path
        # parameters['G_test__with_AD_tree']['ci_test_ad_tree_path__load'] = None
        parameters['G_test__with_AD_tree']['ci_test_ad_tree_path__save'] = None
        parameters['G_test__with_AD_tree']['ci_test_ad_tree_leaf_list_threshold'] = LLT
        parameters['G_test__with_AD_tree']['ci_test_gc_collect_rate'] = 0

        parameters['G_test__with_dcMI'] = dict()
        parameters['G_test__with_dcMI']['ci_test_dof_calculator_class'] = DoF_calculator_class
        parameters['G_test__with_dcMI']['ci_test_gc_collect_rate'] = 0
        parameters['G_test__with_dcMI']['ci_test_jht_path__load'] = None
        parameters['G_test__with_dcMI']['ci_test_jht_path__save'] = None

        for target in [19]:
            self.run_test_dof_across_Gtest_optimizations(dm_label, target, significance, parameters)


    def run_test_dof_across_Gtest_optimizations(self, dm_label, target, significance, parameters):
        ci_test_results__unoptimized = None
        ci_test_results__adtree = None
        ci_test_results__dcmi = None

        datasetmatrix = self.DatasetMatrices[dm_label]

        if 'G_test__unoptimized' in parameters:
            print('=== IPC-MB with G-test (unoptimized) ===')
            extra_parameters = parameters['G_test__unoptimized']
            ipcmb_g_unoptimized = self.make_IPCMB_with_Gtest_unoptimized(dm_label, target, significance, extra_parameters=extra_parameters)
            start_time = time.time()
            markov_blanket__unoptimized = ipcmb_g_unoptimized.select_features()
            duration__unoptimized = time.time() - start_time
            ci_test_results__unoptimized = ipcmb_g_unoptimized.CITest.ci_test_results
            print(self.format_ipcmb_result('unoptimized', target, datasetmatrix, markov_blanket__unoptimized))
            print()

        if 'G_test__with_dcMI' in parameters:
            print('=== IPC-MB with G-test (dcMI) ===')
            extra_parameters = parameters['G_test__with_dcMI']
            ipcmb_g_dcmi = self.make_IPCMB_with_Gtest_dcMI(dm_label, target, significance, extra_parameters=extra_parameters)
            start_time = time.time()
            markov_blanket__dcmi = ipcmb_g_dcmi.select_features()
            duration__dcmi = time.time() - start_time
            ci_test_results__dcmi = ipcmb_g_dcmi.CITest.ci_test_results
            print(self.format_ipcmb_result('dcMI', target, datasetmatrix, markov_blanket__dcmi))
            if ci_test_results__unoptimized is not None:
                self.assertEqualCITestResults(ci_test_results__unoptimized, ci_test_results__dcmi)
            print()

        if 'G_test__with_AD_tree' in parameters:
            print('=== IPC-MB with G-test (AD-tree) ===')
            extra_parameters = parameters['G_test__with_AD_tree']
            ipcmb_g_adtree = self.make_IPCMB_with_Gtest_ADtree(dm_label, target, significance, extra_parameters=extra_parameters)
            start_time = time.time()
            markov_blanket__adtree = ipcmb_g_adtree.select_features()
            duration__adtree = time.time() - start_time
            ci_test_results__adtree = ipcmb_g_adtree.CITest.ci_test_results
            print(self.format_ipcmb_result('AD-tree', target, datasetmatrix, markov_blanket__adtree))
            if ci_test_results__unoptimized is not None:
                self.assertEqualCITestResults(ci_test_results__unoptimized, ci_test_results__adtree)
            print()

        print('=== IPC-MB with d-sep ===')
        ipcmb_dsep = self.make_IPCMB_with_dsep(dm_label, target)
        markov_blanket__dsep = ipcmb_dsep.select_features()
        print(self.format_ipcmb_result('dsep', target, datasetmatrix, markov_blanket__dsep))

        gc.collect()

        print()

        if 'G_test__with_dcMI' in parameters:
            mb_correctness = 'CORRECT'
            if markov_blanket__dsep != markov_blanket__dcmi:
                mb_correctness = 'WRONG'
            print('MB, dcmi ({}): {}, duration {:<.2f}'.format(mb_correctness, markov_blanket__dcmi, duration__dcmi))

        if 'G_test__with_AD_tree' in parameters:
            mb_correctness = 'CORRECT'
            if markov_blanket__dsep != markov_blanket__adtree:
                mb_correctness = 'WRONG'
            print('MB, adtree ({}): {}, duration {:<.2f}'.format(mb_correctness, markov_blanket__adtree, duration__adtree))

        if 'G_test__unoptimized' in parameters:
            mb_correctness = 'CORRECT'
            if markov_blanket__dsep != markov_blanket__unoptimized:
                mb_correctness = 'WRONG'
            print('MB, unoptimized ({}): {}, duration {:<.2f}'.format(mb_correctness, markov_blanket__unoptimized, duration__unoptimized))

        print('MB, d-sep : {}'.format(markov_blanket__dsep))
        print()


    def format_ipcmb_result(self, label, target, datasetmatrix, markov_blanket):
        named_markov_blanket = [datasetmatrix.column_labels_X[n] for n in markov_blanket]
        named_target = datasetmatrix.column_labels_X[target]
        output = '{}: {} ({}) → {} ({})'.format(label, target, named_target, markov_blanket, named_markov_blanket)
        return output


    def make_IPCMB_with_dsep(self, dm_label, target, extra_parameters=dict()):
        ci_test_class = mbff.math.DSeparationCITest.DSeparationCITest

        parameters = dict()
        parameters['ci_test_debug'] = 0
        parameters['algorithm_debug'] = 0
        parameters.update(extra_parameters)

        ipcmb = self.make_IPCMB(dm_label, target, ci_test_class, None, parameters)
        return ipcmb


    def make_IPCMB_with_Gtest_dcMI(self, dm_label, target, significance, extra_parameters=dict()):
        ci_test_class = mbff.math.G_test__with_dcMI.G_test

        JHT_path = self.JHTFolder / (dm_label + '.pickle')
        parameters = dict()
        parameters['ci_test_jht_path__load'] = JHT_path
        parameters['ci_test_jht_path__save'] = JHT_path
        parameters.update(extra_parameters)

        ipcmb = self.make_IPCMB(dm_label, target, ci_test_class, significance, parameters)
        return ipcmb


    def make_IPCMB_with_Gtest_ADtree(self, dm_label, target, significance, extra_parameters=dict()):
        ci_test_class = mbff.math.G_test__with_AD_tree.G_test

        LLT = extra_parameters['ci_test_ad_tree_leaf_list_threshold']
        ADTree_path = self.ADTreesFolder / ('{}_LLT={}.pickle'.format(dm_label, LLT))
        parameters = dict()
        parameters['ci_test_ad_tree_path__load'] = ADTree_path
        parameters['ci_test_ad_tree_path__save'] = ADTree_path
        parameters.update(extra_parameters)

        ipcmb = self.make_IPCMB(dm_label, target, ci_test_class, significance, parameters)
        return ipcmb


    def make_IPCMB_with_Gtest_unoptimized(self, dm_label, target, significance, extra_parameters=dict()):
        ci_test_class = mbff.math.G_test__unoptimized.G_test
        ipcmb = self.make_IPCMB(dm_label, target, ci_test_class, significance, extra_parameters)

        return ipcmb


    def make_IPCMB(self, dm_label, target, ci_test_class, significance, extra_parameters=dict()):
        omega = self.OmegaVariables[dm_label]
        datasetmatrix = self.DatasetMatrices[dm_label]
        bn = self.BayesianNetworks[dm_label]

        parameters = dict()
        parameters['target'] = target
        parameters['ci_test_class'] = ci_test_class
        parameters['ci_test_debug'] = self.CITestDebugLevel
        parameters['ci_test_significance'] = significance
        parameters['ci_test_results__print_accurate'] = False
        parameters['ci_test_results__print_inaccurate'] = True
        parameters['algorithm_debug'] = self.DebugLevel
        parameters['omega'] = omega
        parameters['source_bayesian_network'] = bn

        parameters.update(extra_parameters)
        ipcmb = AlgorithmIPCMB(datasetmatrix, parameters)
        return ipcmb


    def assertEqualCITestResults(self, expected_results, obtained_results):
        for (expected_result, obtained_result) in zip(expected_results, obtained_results):
            expected_result.tolerance__statistic_value = 1e-8
            expected_result.tolerance__p_value = 1e-9
            failMessage = (
                'Differing CI test results:\n'
                'REFERENCE: {}\n'
                'COMPUTED:  {}\n'
                '{}\n'
            ).format(expected_result, obtained_result, expected_result.diff(obtained_result))
            self.assertTrue(expected_result == obtained_result, failMessage)


    def print_ci_test_results(self, ci_test_results):
        print()
        print('==========')
        print('CI test results:')
        for result in ci_test_results:
            print(result)
        print()
        print('Total: {} CI tests'.format(len(ci_test_results)))


    def load_AD_tree(self, path):
        if path is None:
            return None

        print('Preloading AD-tree from', path)
        start = time.time()
        AD_tree = None
        try:
            with path.open('rb') as f:
                AD_tree = pickle.load(f)
        except FileNotFoundError:
            raise

        print('Duration: {:<.2f}s'.format(time.time() - start))
        return AD_tree


    def mock_pmf_for_cjpi_testing__xyz(self):
        p = dict()
        cp = dict()

        z = 0
        cp[z] = dict()
        for x in [6, 7]:
            for y in range(5):
                p[(x, y, z)] = 1
                cp[z][(x, y)] = 1

        z = 1
        cp[z] = dict()
        for x in [0, 1, 2, 5, 6, 7]:
            for y in [0, 2, 3, 4]:
                p[(x, y, z)] = 1
                cp[z][(x, y)] = 1

        p[(5, 3, z)] = 0
        cp[z][(5, 3)] = 0
        p[(5, 4, z)] = 0
        cp[z][(5, 4)] = 0

        z = 2
        cp[z] = dict()
        for x in [0, 1, 5, 6, 7]:
            for y in [0, 1, 4]:
                p[(x, y, z)] = 1
                cp[z][(x, y)] = 1

        z = 3
        cp[z] = dict()
        for x in [1, 2, 3, 4, 5, 6, 7]:
            for y in range(5):
                p[(x, y, z)] = 1
                cp[z][(x, y)] = 1
        p[(3, 3, z)] = 0
        cp[z][(3, 3)] = 0

        z = 4
        cp[z] = dict()
        for x in range(8):
            for y in range(5):
                if x == 2 or y == 4:
                    continue
                p[(x, y, z)] = 1
                cp[z][(x, y)] = 1
        p[(3, 3, z)] = 0
        cp[z][(3, 3)] = 0

        PrXYZ = PMF(None)
        PrXYZ.probabilities = p
        PrXYcZ = CPMF(None, None)
        PrXYcZ.conditional_probabilities = cp
        return (PrXYZ, PrXYcZ)


    def mock_pmf_for_cjpi_testing__xz(self):
        p = dict()
        cp = dict()

        z = 0
        cp[z] = dict()
        for x in [6, 7]:
            p[(x, z)] = 1
            cp[z][x] = 1

        z = 1
        cp[z] = dict()
        for x in [0, 1, 2, 5, 6, 7]:
            p[(x, z)] = 1
            cp[z][x] = 1

        z = 2
        cp[z] = dict()
        for x in [0, 1, 5, 6, 7]:
            p[(x, z)] = 1
            cp[z][x] = 1

        z = 3
        cp[z] = dict()
        for x in [1, 2, 3, 4, 5, 6, 7]:
            p[(x, z)] = 1
            cp[z][x] = 1

        z = 4
        cp[z] = dict()
        for x in [0, 1, 3, 4, 5, 6, 7]:
            p[(x, z)] = 1
            cp[z][x] = 1

        PrXZ = PMF(None)
        PrXZ.probabilities = p
        PrXcZ = CPMF(None, None)
        PrXcZ.conditional_probabilities = cp
        return (PrXZ, PrXcZ)


    def mock_pmf_for_cjpi_testing__yz(self):
        p = dict()
        cp = dict()

        z = 0
        cp[z] = dict()
        for y in [0, 1, 2, 3, 4]:
            p[(y, z)] = 1
            cp[z][y] = 1

        z = 1
        cp[z] = dict()
        for y in [0, 2, 3, 4]:
            p[(y, z)] = 1
            cp[z][y] = 1

        z = 2
        cp[z] = dict()
        for y in [0, 1, 4]:
            p[(y, z)] = 1
            cp[z][y] = 1

        z = 3
        cp[z] = dict()
        for y in [0, 1, 2, 3, 4]:
            p[(y, z)] = 1
            cp[z][y] = 1

        z = 4
        cp[z] = dict()
        for y in [0, 1, 2, 3]:
            p[(y, z)] = 1
            cp[z][y] = 1

        PrYZ = PMF(None)
        PrYZ.probabilities = p
        PrYcZ = CPMF(None, None)
        PrYcZ.conditional_probabilities = cp
        return (PrYZ, PrYcZ)


    def mock_pmf_for_cjpi_testing__z(self):
        p = dict()

        for z in [0, 1, 2, 3, 4]:
            p[z] = 1

        PrZ = PMF(None)
        PrZ.probabilities = p
        return PrZ
