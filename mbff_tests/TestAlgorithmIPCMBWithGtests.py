import unittest
import gc
from pathlib import Path

from mbff_tests.TestBase import TestBase
from mbff.algorithms.mb.ipcmb import AlgorithmIPCMB
import mbff.math.G_test__unoptimized
import mbff.math.G_test__with_AD_tree
import mbff.math.G_test__with_dcMI
import mbff.math.DSeparationCITest


@unittest.skipIf(TestBase.tag_excluded('ipcmb_run'), 'Tests running IPC-MB are excluded')
class TestAlgorithmIPCMBWithGtests(TestBase):

    @classmethod
    def initTestResources(testClass):
        super().initTestResources()
        testClass.DatasetsInUse = ['lc_repaired', 'alarm']
        testClass.RootFolder = Path('testfiles', 'tmp', 'test_ipcmb_with_gtests')
        testClass.DatasetMatrixFolder = testClass.RootFolder / 'dm'
        testClass.JHTFolder = testClass.RootFolder / 'jht'
        testClass.ADTreesFolder = testClass.RootFolder / 'adtrees'
        testClass.CITestResultsFolder = testClass.RootFolder / 'ci_test_results'
        testClass.DebugLevel = 0
        testClass.CITestDebugLevel = 1


    @classmethod
    def prepareTestResources(testClass):
        super(TestAlgorithmIPCMBWithGtests, testClass).prepareTestResources()
        testClass.ADTreesFolder.mkdir(parents=True, exist_ok=True)
        testClass.JHTFolder.mkdir(parents=True, exist_ok=True)
        testClass.CITestResultsFolder.mkdir(parents=True, exist_ok=True)

    @classmethod
    def configure_dataset(testClass, dm_label):
        configuration = dict()

        if dm_label == 'lc_repaired':
            configuration['sourcepath'] = Path('testfiles', 'bif_files', 'lc_repaired.bif')
            configuration['sample_count'] = int(1e4)
            configuration['random_seed'] = 19**8
            configuration['values_as_indices'] = True
            configuration['objectives'] = []

        if dm_label == 'alarm':
            configuration['sourcepath'] = Path('testfiles', 'bif_files', 'alarm.bif')
            configuration['sample_count'] = int(8e3)
            configuration['random_seed'] = 1984
            configuration['values_as_indices'] = True
            configuration['objectives'] = []

        return configuration


    def test_accurate_Markov_blankets(self):
        dm_label = 'lc_repaired'
        datasetmatrix = self.DatasetMatrices[dm_label]
        targets = range(datasetmatrix.get_column_count('X'))
        significance = 0.98

        print()
        for target in targets:
            print('=== IPC-MB with d-sep ===')
            ipcmb_dsep = self.make_IPCMB_with_dsep(dm_label, target)
            markov_blanket__dsep = ipcmb_dsep.select_features()
            print(self.format_ipcmb_result('dsep', target, datasetmatrix, markov_blanket__dsep))

            print('=== IPC-MB with G-test (dcMI) ===')
            extra_parameters = {'ci_test_dof_computation_method': 'structural'}
            ipcmb_g_dcmi = self.make_IPCMB_with_Gtest_dcMI(dm_label, target, significance, extra_parameters)
            markov_blanket__dcmi = ipcmb_g_dcmi.select_features()
            print(self.format_ipcmb_result('dcMI', target, datasetmatrix, markov_blanket__dcmi))
            self.assertEqual(markov_blanket__dsep, markov_blanket__dcmi)

            print('=== IPC-MB with G-test (AD-tree) ===')
            extra_parameters = {'ci_test_dof_computation_method': 'rowcol'}
            ipcmb_g_adtree = self.make_IPCMB_with_Gtest_ADtree(dm_label, target, significance, LLT=200, extra_parameters=extra_parameters)
            markov_blanket__adtree = ipcmb_g_adtree.select_features()
            print(self.format_ipcmb_result('AD-tree', target, datasetmatrix, markov_blanket__adtree))
            self.assertEqual(markov_blanket__dsep, markov_blanket__adtree)

            print('=== IPC-MB with G-test (unoptimized) ===')
            extra_parameters = {'ci_test_dof_computation_method': 'rowcol_minus_zerocells'}
            ipcmb_g_unoptimized = self.make_IPCMB_with_Gtest_unoptimized(dm_label, target, significance, extra_parameters)
            markov_blanket__unoptimized = ipcmb_g_unoptimized.select_features()
            print(self.format_ipcmb_result('unoptimized', target, datasetmatrix, markov_blanket__unoptimized))
            self.assertEqual(markov_blanket__dsep, markov_blanket__unoptimized)

            print()

            gc.collect()


    def test_dof_computation_methods(self):
        dm_label = 'alarm'
        datasetmatrix = self.DatasetMatrices[dm_label]
        targets = range(datasetmatrix.get_column_count('X'))
        significance = 0.95

        print()
        for target in targets:
            print('=== IPC-MB with d-sep ===')
            ipcmb_dsep = self.make_IPCMB_with_dsep(dm_label, target)
            markov_blanket__dsep = ipcmb_dsep.select_features()
            print(self.format_ipcmb_result('dsep', target, datasetmatrix, markov_blanket__dsep))

            print('=== IPC-MB with G-test (AD-tree - dof_computation_method = structural) ===')
            extra_parameters = {'ci_test_dof_computation_method': 'structural'}
            ipcmb_g_adtree = self.make_IPCMB_with_Gtest_ADtree(dm_label, target, significance, LLT=400, extra_parameters=extra_parameters)
            markov_blanket__adtree = ipcmb_g_adtree.select_features()
            print(self.format_ipcmb_result('AD-tree', target, datasetmatrix, markov_blanket__adtree))
            self.assertEqual(markov_blanket__dsep, markov_blanket__adtree)

            print('=== IPC-MB with G-test (AD-tree - dof_computation_method = rowcol) ===')
            extra_parameters = {'ci_test_dof_computation_method': 'rowcol'}
            ipcmb_g_adtree = self.make_IPCMB_with_Gtest_ADtree(dm_label, target, significance, LLT=400, extra_parameters=extra_parameters)
            markov_blanket__adtree = ipcmb_g_adtree.select_features()
            print(self.format_ipcmb_result('AD-tree', target, datasetmatrix, markov_blanket__adtree))
            self.assertEqual(markov_blanket__dsep, markov_blanket__adtree)

            print('=== IPC-MB with G-test (AD-tree - dof_computation_method = rowcol_minus_zerocells) ===')
            extra_parameters = {'ci_test_dof_computation_method': 'rowcol_minus_zerocells'}
            ipcmb_g_adtree = self.make_IPCMB_with_Gtest_ADtree(dm_label, target, significance, LLT=400, extra_parameters=extra_parameters)
            markov_blanket__adtree = ipcmb_g_adtree.select_features()
            print(self.format_ipcmb_result('AD-tree', target, datasetmatrix, markov_blanket__adtree))
            self.assertEqual(markov_blanket__dsep, markov_blanket__adtree)

            print()

            gc.collect()




    def format_ipcmb_result(self, label, target, datasetmatrix, markov_blanket):
        named_markov_blanket = [datasetmatrix.column_labels_X[n] for n in markov_blanket]
        named_target = datasetmatrix.column_labels_X[target]
        output = '{}: {} ({}) → {} ({})'.format(label, target, named_target, markov_blanket, named_markov_blanket)
        return output


    def make_IPCMB_with_dsep(self, dm_label, target, extra_parameters=dict()):
        ci_test_class = mbff.math.DSeparationCITest.DSeparationCITest

        extra_parameters['ci_test_debug'] = 0
        extra_parameters['algorithm_debug'] = 0

        ipcmb = self.make_IPCMB(dm_label, target, ci_test_class, None, extra_parameters)
        return ipcmb


    def make_IPCMB_with_Gtest_dcMI(self, dm_label, target, significance, extra_parameters=dict()):
        ci_test_class = mbff.math.G_test__with_dcMI.G_test

        JHT_path = self.JHTFolder / (dm_label + '.pickle')
        extra_parameters['ci_test_jht_path__load'] = JHT_path
        extra_parameters['ci_test_jht_path__save'] = JHT_path

        ipcmb = self.make_IPCMB(dm_label, target, ci_test_class, significance, extra_parameters)
        return ipcmb


    def make_IPCMB_with_Gtest_ADtree(self, dm_label, target, significance, LLT, extra_parameters=dict()):
        ci_test_class = mbff.math.G_test__with_AD_tree.G_test

        extra_parameters['ci_test_ad_tree_leaf_list_threshold'] = LLT
        ADTree_path = self.ADTreesFolder / ('{}_LLT={}.pickle'.format(dm_label, LLT))
        extra_parameters['ci_test_ad_tree_path__load'] = ADTree_path
        extra_parameters['ci_test_ad_tree_path__save'] = ADTree_path

        ipcmb = self.make_IPCMB(dm_label, target, ci_test_class, significance, extra_parameters)
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


    def print_ci_test_results(self, ci_test_results):
        print()
        print('==========')
        print('CI test results:')
        for result in ci_test_results:
            print(result)
        print()
        print('Total: {} CI tests'.format(len(ci_test_results)))


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
