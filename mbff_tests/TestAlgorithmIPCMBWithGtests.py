import unittest
import gc
import pickle
from pathlib import Path

from mbff_tests.TestBase import TestBase
from mbff.algorithms.mb.ipcmb import AlgorithmIPCMB
import mbff.math.G_test__unoptimized
import mbff.math.G_test__with_AD_tree
import mbff.math.G_test__with_dcMI
import mbff.math.DSeparationCITest

from mbff.math.PMF import PMF, CPMF


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

        return configuration


    def test_accurate_Markov_blankets(self):
        dm_label = 'alarm'
        datasetmatrix = self.DatasetMatrices[dm_label]
        # targets = range(datasetmatrix.get_column_count('X'))
        significance = 0.9

        print()
        for target in [19]:
            print('=== IPC-MB with d-sep ===')
            ipcmb_dsep = self.make_IPCMB_with_dsep(dm_label, target)
            markov_blanket__dsep = ipcmb_dsep.select_features()
            print(self.format_ipcmb_result('dsep', target, datasetmatrix, markov_blanket__dsep))

            print('=== IPC-MB with G-test (dcMI) ===')
            extra_parameters = {'ci_test_dof_computation_method': 'structural'}
            ipcmb_g_dcmi = self.make_IPCMB_with_Gtest_dcMI(dm_label, target, significance, extra_parameters)
            markov_blanket__dcmi = ipcmb_g_dcmi.select_features()
            print(self.format_ipcmb_result('dcMI', target, datasetmatrix, markov_blanket__dcmi))
            # self.assertEqual(markov_blanket__dsep, markov_blanket__dcmi)

            print('=== IPC-MB with G-test (unoptimized) ===')
            extra_parameters = {'ci_test_dof_computation_method': 'rowcol'}
            ipcmb_g_unoptimized = self.make_IPCMB_with_Gtest_unoptimized(dm_label, target, significance, extra_parameters)
            markov_blanket__unoptimized = ipcmb_g_unoptimized.select_features()
            print(self.format_ipcmb_result('unoptimized', target, datasetmatrix, markov_blanket__unoptimized))
            # self.assertEqual(markov_blanket__dsep, markov_blanket__unoptimized)

            print('=== IPC-MB with G-test (unoptimized) ===')
            extra_parameters = {'ci_test_dof_computation_method': 'structural_minus_zerocells'}
            ipcmb_g_unoptimized = self.make_IPCMB_with_Gtest_unoptimized(dm_label, target, significance, extra_parameters)
            markov_blanket__unoptimized = ipcmb_g_unoptimized.select_features()
            print(self.format_ipcmb_result('unoptimized', target, datasetmatrix, markov_blanket__unoptimized))
            # self.assertEqual(markov_blanket__dsep, markov_blanket__unoptimized)

            print()

            gc.collect()


    def test_dof_computation_methods__cjpi(self):
        dm_label = 'alarm'
        # targets = range(datasetmatrix.get_column_count('X'))
        significance = 0.9

        cjpi_folder = self.RootFolder / 'cjpi'
        cjpi_folder.mkdir(parents=True, exist_ok=True)

        print()
        for target in [19]:
            print('=== IPC-MB with G-test (unoptimized - dof_computation_method = cjpi) ===')
            extra_parameters = dict()
            extra_parameters['ci_test_dof_computation_method'] = 'cached_joint_pmf_info'
            extra_parameters['ci_test_results_path__save'] = self.CITestResultsFolder / '{}_{}_dcmi_dof_cjpi.pickle'.format(dm_label, target)
            ipcmb = self.make_IPCMB_with_Gtest_unoptimized(dm_label, target, significance, extra_parameters=extra_parameters)
            markov_blanket = ipcmb.select_features()
            print(markov_blanket)


    def test_dof_computation_methods__cjpi__detailed(self):
        parameters = dict()
        parameters['ci_test_dof_computation_method'] = 'cached_joint_pmf_info'
        parameters['ci_test_debug'] = self.CITestDebugLevel
        parameters['ci_test_significance'] = 0.9
        parameters['ci_test_results__print_accurate'] = True
        parameters['ci_test_results__print_inaccurate'] = True

        G_test = mbff.math.G_test__unoptimized.G_test(None, parameters)
        G_test.column_values = [[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5]]
        (PrXYZ, PrXYcZ) = self.mock_pmf_for_cjpi_testing__xyz()
        (PrXZ, PrXcZ) = self.mock_pmf_for_cjpi_testing__xz()
        (PrYZ, PrYcZ) = self.mock_pmf_for_cjpi_testing__yz()
        PrZ = self.mock_pmf_for_cjpi_testing__z()
        X = 0
        Y = 1
        Z = 2

        expected_dof = G_test.calculate_degrees_of_freedom__rowcol(PrXYcZ, PrXcZ, PrYcZ, PrZ)

        G_test.cache_pmf_infos__XYZ(PrXYZ, PrXZ, PrYZ, PrZ, X, Y, [Z])
        computed_dof = G_test.calculate_degrees_of_freedom__cached_joint_pmf_info(X, Y, [Z])
        self.assertEqual(expected_dof, computed_dof)


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


    def test_dof_cjpi_across_Gtest_optimizations(self):
        dm_label = 'alarm'
        datasetmatrix = self.DatasetMatrices[dm_label]
        # targets = range(datasetmatrix.get_column_count('X'))
        significance = 0.9
        LLT = 0

        ADTree_path = self.ADTreesFolder / ('{}_LLT={}.pickle'.format(dm_label, LLT))
        preloaded_AD_tree = None
        try:
            with ADTree_path.open('rb') as f:
                preloaded_AD_tree = pickle.load(f)
        except FileNotFoundError:
            pass

        cjpi_folder = self.RootFolder / 'cjpi'
        cjpi_folder.mkdir(parents=True, exist_ok=True)

        print()
        for target in [19]:
            print('=== IPC-MB with d-sep ===')
            ipcmb_dsep = self.make_IPCMB_with_dsep(dm_label, target)
            markov_blanket__dsep = ipcmb_dsep.select_features()
            print(self.format_ipcmb_result('dsep', target, datasetmatrix, markov_blanket__dsep))

            print('=== IPC-MB with G-test (unoptimized - dof_computation_method = cached_joint_pmf_info) ===')
            extra_parameters = dict()
            extra_parameters['ci_test_dof_computation_method'] = 'cached_joint_pmf_info'
            extra_parameters['ci_test_results_path__save'] = self.CITestResultsFolder / '{}_{}_unoptimized_dof_cjpi.pickle'.format(dm_label, target)
            extra_parameters['ci_test_pmf_info_path__load'] = cjpi_folder / '{}_{}_unoptimized_dof_cjpi.pickle'.format(dm_label, target)
            extra_parameters['ci_test_pmf_info_path__save'] = cjpi_folder / '{}_{}_unoptimized_dof_cjpi.pickle'.format(dm_label, target)
            extra_parameters['ci_test_gc_collect_rate'] = 0
            ipcmb_g_unoptimized = self.make_IPCMB_with_Gtest_unoptimized(dm_label, target, significance, extra_parameters=extra_parameters)
            markov_blanket__unoptimized = ipcmb_g_unoptimized.select_features()
            print(self.format_ipcmb_result('unoptimized', target, datasetmatrix, markov_blanket__unoptimized))
            print()

            print('=== IPC-MB with G-test (AD-tree - dof_computation_method = cached_joint_pmf_info) ===')
            extra_parameters = dict()
            extra_parameters['ci_test_dof_computation_method'] = 'cached_joint_pmf_info'
            extra_parameters['ci_test_results_path__save'] = self.CITestResultsFolder / '{}_{}_adtree_dof_cjpi.pickle'.format(dm_label, target)
            extra_parameters['ci_test_pmf_info_path__load'] = cjpi_folder / '{}_{}_adtree_dof_cjpi.pickle'.format(dm_label, target)
            extra_parameters['ci_test_pmf_info_path__save'] = cjpi_folder / '{}_{}_adtree_dof_cjpi.pickle'.format(dm_label, target)
            extra_parameters['ci_test_ad_tree_preloaded'] = preloaded_AD_tree
            extra_parameters['ci_test_ad_tree_path__save'] = None
            ipcmb_g_adtree = self.make_IPCMB_with_Gtest_ADtree(dm_label, target, significance, LLT=LLT, extra_parameters=extra_parameters)
            markov_blanket__adtree = ipcmb_g_adtree.select_features()
            print(self.format_ipcmb_result('AD-tree', target, datasetmatrix, markov_blanket__adtree))
            print()

            print('=== IPC-MB with G-test (dcMI - dof_computation_method = cached_joint_pmf_info) ===')
            extra_parameters = dict()
            extra_parameters['ci_test_dof_computation_method'] = 'cached_joint_pmf_info'
            extra_parameters['ci_test_pmf_info_path__load'] = cjpi_folder / '{}_{}_dcmi_dof_cjpi.pickle'.format(dm_label, target)
            extra_parameters['ci_test_pmf_info_path__save'] = cjpi_folder / '{}_{}_dcmi_dof_cjpi.pickle'.format(dm_label, target)
            extra_parameters['ci_test_results_path__save'] = self.CITestResultsFolder / '{}_{}_dcmi_dof_cjpi.pickle'.format(dm_label, target)
            ipcmb_g_dcmi = self.make_IPCMB_with_Gtest_dcMI(dm_label, target, significance, extra_parameters=extra_parameters)
            markov_blanket__dcmi = ipcmb_g_dcmi.select_features()
            print(self.format_ipcmb_result('dcMI', target, datasetmatrix, markov_blanket__dcmi))
            print()

            gc.collect()

            print()

            mb_correctness = 'CORRECT'
            if markov_blanket__dsep != markov_blanket__unoptimized:
                mb_correctness = 'WRONG'
            print('MB, unoptimized ({}): {} vs {}'.format(mb_correctness, markov_blanket__dsep, markov_blanket__unoptimized))

            mb_correctness = 'CORRECT'
            if markov_blanket__dsep != markov_blanket__adtree:
                mb_correctness = 'WRONG'
            print('MB, adtree ({}): {} vs {}'.format(mb_correctness, markov_blanket__dsep, markov_blanket__adtree))

            mb_correctness = 'CORRECT'
            if markov_blanket__dsep != markov_blanket__dcmi:
                mb_correctness = 'WRONG'
            print('MB, dcmi ({}): {} vs {}'.format(mb_correctness, markov_blanket__dsep, markov_blanket__dcmi))

            print()
            print()


    def test_individual_gtest(self):
        dm_label = 'alarm'

        target = 16
        significance = 0.9

        self.CITestDebugLevel = 0

        extra_parameters = dict()
        extra_parameters['ci_test_dof_computation_method'] = 'cached_joint_pmf_info'
        extra_parameters['ci_test_cjpi_adjustment_type'] = 'max'
        extra_parameters['ci_test_gc_collect_rate'] = 0
        extra_parameters['ci_test_results__print_accurate'] = False
        extra_parameters['ci_test_results__print_inaccurate'] = False
        ipcmb_g_unoptimized = self.make_IPCMB_with_Gtest_unoptimized(dm_label, target, significance, extra_parameters=extra_parameters)
        markov_blanket__unoptimized = ipcmb_g_unoptimized.select_features()
        citr = ipcmb_g_unoptimized.CITest.ci_test_results
        accuracy = len([r for r in citr if r.accurate()]) / len(citr)
        citr_cjpi_A = citr
        accuracy_cjpi_A = accuracy

        extra_parameters = dict()
        extra_parameters['ci_test_dof_computation_method'] = 'rowcol'
        extra_parameters['ci_test_gc_collect_rate'] = 0
        extra_parameters['ci_test_results__print_accurate'] = False
        extra_parameters['ci_test_results__print_inaccurate'] = False
        ipcmb_g_unoptimized = self.make_IPCMB_with_Gtest_unoptimized(dm_label, target, significance, extra_parameters=extra_parameters)
        markov_blanket__unoptimized = ipcmb_g_unoptimized.select_features()
        citr = ipcmb_g_unoptimized.CITest.ci_test_results
        accuracy = len([r for r in citr if r.accurate()]) / len(citr)
        citr_rowcol = citr
        accuracy_rowcol = accuracy

        extra_parameters = dict()
        extra_parameters['ci_test_dof_computation_method'] = 'structural_minus_zerocells'
        extra_parameters['ci_test_gc_collect_rate'] = 0
        extra_parameters['ci_test_results__print_accurate'] = False
        extra_parameters['ci_test_results__print_inaccurate'] = False
        ipcmb_g_unoptimized = self.make_IPCMB_with_Gtest_unoptimized(dm_label, target, significance, extra_parameters=extra_parameters)
        markov_blanket__unoptimized = ipcmb_g_unoptimized.select_features()
        citr = ipcmb_g_unoptimized.CITest.ci_test_results
        accuracy = len([r for r in citr if r.accurate()]) / len(citr)
        citr_structural_minus_zerocells = citr
        accuracy_structural_minus_zerocells = accuracy

        ipcmb_dsep = self.make_IPCMB_with_dsep(dm_label, target)
        markov_blanket__dsep = ipcmb_dsep.select_features()

        print()
        print('markov_blanket__unoptimized:', markov_blanket__unoptimized)
        print('markov_blanket__dsep:', markov_blanket__dsep, 'test count:', len(ipcmb_dsep.CITest.ci_test_results))
        print('accuracy_cjpi_A:', accuracy_cjpi_A, 'test count:', len(citr_cjpi_A))
        print('accuracy_rowcol:', accuracy_rowcol, 'test count:', len(citr_rowcol))
        print('accuracy_structural_minus_zerocells:', accuracy_structural_minus_zerocells, 'test count:', len(citr_structural_minus_zerocells))


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
        PMF_info_path = self.JHTFolder / (dm_label + '_pmf_info.pickle')
        extra_parameters['ci_test_jht_path__load'] = JHT_path
        extra_parameters['ci_test_jht_path__save'] = JHT_path
        extra_parameters['ci_test_pmf_info_path__load'] = PMF_info_path
        extra_parameters['ci_test_pmf_info_path__save'] = PMF_info_path

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
