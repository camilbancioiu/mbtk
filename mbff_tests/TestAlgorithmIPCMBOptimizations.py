import pickle
import unittest
from pathlib import Path

from mbff_tests.TestBase import TestBase
from mbff.algorithms.mb.ipcmb import AlgorithmIPCMB
import mbff.math.G_test__unoptimized
import mbff.math.G_test__with_AD_tree
import mbff.math.G_test__with_dcMI

import pympler.asizeof
import humanize
import time


@unittest.skipIf(TestBase.tag_excluded('ipcmb_run'), 'Tests running IPC-MB are excluded')
class TestAlgorithmIPCMBOptimizations(TestBase):

    def initTestResources(self):
        super().initTestResources()
        self.DatasetsInUse = ['survey']
        self.DatasetMatrixFolder = Path('testfiles', 'tmp', 'test_ipcmb_optimizations_dm')
        self.ReferenceCITestResultsFolder = Path('testfiles', 'tmp', 'test_ipcmb_optimizations_ci_tests')
        self.ReferenceCITestResults = None


    def prepareTestResources(self):
        super().prepareTestResources()
        self.prepare_reference_ci_test_results()


    def test_IPCMB_w_AD_tree(self):
        parameters = {'ci_test_debug': 2}
        ipcmb = self.run_IPCMB('survey', 3, mbff.math.G_test__with_AD_tree.G_test, extra_parameters=parameters)
        AD_tree = ipcmb.CITest.AD_tree
        size = pympler.asizeof.asizeof(AD_tree)
        print("AD-tree size: {}".format(humanize.naturalsize(size)))


    def test_IPCMB_w_dcMI(self):
        parameters = {'ci_test_debug': 1}
        ipcmb = self.run_IPCMB('survey', 3, mbff.math.G_test__with_dcMI.G_test, extra_parameters=parameters)
        jmi_cache = ipcmb.CITest.JMI_cache
        size = pympler.asizeof.asizeof(jmi_cache)
        print("JMI_cache size: {}".format(humanize.naturalsize(size)))


    def run_IPCMB(self, dm_label, target, ci_test_class, bif_file=None, extra_parameters=dict()):
        omega = self.Omega[dm_label]
        datasetmatrix = self.DatasetMatrices[dm_label]
        bn = self.BayesianNetworks[dm_label]

        ad_tree_path = Path('testfiles', 'tmp', 'test_ipcmb_optimizations_ad_trees')
        ad_tree_path.mkdir(parents=True, exist_ok=True)

        parameters = dict()
        parameters['target'] = target
        parameters['ci_test_class'] = ci_test_class
        parameters['ci_test_debug'] = 1
        parameters['ci_test_significance'] = 0.95
        parameters['ci_test_ad_tree_leaf_list_threshold'] = 5000
        parameters['ci_test_ad_tree_path__save'] = ad_tree_path / (dm_label + '.pickle')
        parameters['ci_test_ad_tree_path__load'] = ad_tree_path / (dm_label + '.pickle')
        parameters['algorithm_debug'] = 1
        parameters['omega'] = omega
        parameters['source_bayesian_network'] = bn

        parameters.update(extra_parameters)

        print()
        start_time = time.time()
        ipcmb = AlgorithmIPCMB(datasetmatrix, parameters)
        ipcmb.select_features()
        end_time = time.time()
        print("Algorithm run duration: {:.2f}s".format(end_time - start_time))
        computed_citrs = ipcmb.CITest.ci_test_results
        self.assert_ci_test_results_equal_to_reference(dm_label, computed_citrs)

        return ipcmb


    def prepare_reference_ci_test_results(self):
        self.ReferenceCITestResults = dict()

        self.ReferenceCITestResultsFolder.mkdir(parents=True, exist_ok=True)
        for dm_label in self.DatasetsInUse:
            try:
                source_path = self.ReferenceCITestResultsFolder / 'ci_test_results__{}.pickle'.format(dm_label)
                with source_path.open('rb') as f:
                    self.ReferenceCITestResults[dm_label] = pickle.load(f)
            except FileNotFoundError:
                self.ReferenceCITestResults[dm_label] = self.build_reference_ci_test_results(dm_label)
                with source_path.open('wb') as f:
                    pickle.dump(self.ReferenceCITestResults[dm_label], f)


    def build_reference_ci_test_results(self, dm_label):
        omega = self.Omega[dm_label]
        datasetmatrix = self.DatasetMatrices[dm_label]
        bn = self.BayesianNetworks[dm_label]

        parameters = dict()
        parameters['target'] = 3
        parameters['ci_test_class'] = mbff.math.G_test__unoptimized.G_test
        parameters['ci_test_significance'] = 0.95
        parameters['ci_test_debug'] = 1
        parameters['algorithm_debug'] = 1
        parameters['omega'] = omega
        parameters['source_bayesian_network'] = bn
        parameters['ci_test_results'] = list()

        print()
        print('Building reference CI test results...')
        start_time = time.time()
        ipcmb = AlgorithmIPCMB(datasetmatrix, parameters)
        ipcmb.select_features()
        end_time = time.time()
        print("Reference run duration: {:.2f}s".format(end_time - start_time))
        return ipcmb.CITest.ci_test_results


    def print_ci_test_results(self, ci_test_results):
        print()
        print('==========')
        print('CI test results:')
        for result in ci_test_results:
            print(result)
        print()
        print('Total: {} CI tests'.format(len(ci_test_results)))


    def assert_ci_test_results_equal_to_reference(self, dm_label, computed_ci_test_results):
        reference_ci_test_results = self.ReferenceCITestResults[dm_label]
        for (ref_citr, comp_citr) in zip(reference_ci_test_results, computed_ci_test_results):
            ref_citr.tolerance__statistic_value = 1e-8
            ref_citr.tolerance__p_value = 1e-9
            failMessage = (
                'Differing CI test results:\n'
                'REFERENCE: {}\n'
                'COMPUTED:  {}\n'
                '{}\n'
            ).format(ref_citr, comp_citr, ref_citr.diff(comp_citr))
            self.assertTrue(ref_citr == comp_citr, failMessage)


    def configure_dataset(self, dm_label):
        configuration = dict()

        if dm_label == 'survey':
            configuration['sourcepath'] = Path('testfiles', 'bif_files', 'survey.bif')
            configuration['sample_count'] = int(5e5)
            configuration['random_seed'] = 42 * 42
            configuration['values_as_indices'] = True
            configuration['objectives'] = []
        return configuration
