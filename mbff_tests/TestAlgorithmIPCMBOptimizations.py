import pickle
import unittest
from pprint import pprint
from pathlib import Path

from mbff_tests.TestBase import TestBase
from mbff.dataset.DatasetMatrix import DatasetMatrix
from mbff.math.BayesianNetwork import BayesianNetwork
from mbff.math.Variable import Variable, JointVariables, Omega
from mbff.dataset.sources.SampledBayesianNetworkDatasetSource import SampledBayesianNetworkDatasetSource
from mbff.algorithms.mb.ipcmb import AlgorithmIPCMB
import mbff.math.G_test__unoptimized
import mbff.utilities.functions as util


class TestAlgorithmIPCMBOptimizations(TestBase):

    ClassIsSetUp = False
    DatasetMatrices = None
    Omega = None
    ReferenceCITestResults = None


    def setUp(self):
        self.DatasetMatricesInUse = ['lungcancer']
        if not TestAlgorithmIPCMBOptimizations.ClassIsSetUp:
            self.prepare_datasetmatrices()
            self.prepare_reference_ci_test_results()

    
    def test_IPCMB_w_AD_tree(self):
        dm_label = 'lungcancer'

        reference_citrs = self.ReferenceCITestResults[dm_label]

        Omega = TestAlgorithmIPCMBOptimizations.Omega[dm_label]
        datasetmatrix = TestAlgorithmIPCMBOptimizations.DatasetMatrices[dm_label]

        bif_file = Path('testfiles', 'bif_files', '{}.bif'.format(dm_label))
        bn = util.read_bif_file(bif_file)
        bn.finalize()

        parameters = dict()
        parameters['target'] = 3
        parameters['ci_test_class'] = mbff.math.G_test__unoptimized.G_test
        parameters['ci_test_significance'] = 0.95
        parameters['debug'] = True
        parameters['omega'] = Omega
        parameters['source_bayesian_network'] = bn

        ipcmb = AlgorithmIPCMB(datasetmatrix, parameters)
        ipcmb.select_features()
        computed_citrs = ipcmb.CITest.ci_test_results

        self.assertEqual(reference_citrs, computed_citrs)
        print('All ok')


    def prepare_reference_ci_test_results(self):
        self.ReferenceCITestResults = dict()

        source_folder = Path('testfiles', 'tmp', 'test_ipcmb_optimizations_ci_tests')
        source_folder.mkdir(parents=True, exist_ok=True)
        for dm_label in self.DatasetMatricesInUse:
            try:
                source_path = source_folder / 'ci_test_results__{}.pickle'.format(dm_label)
                with source_path.open('rb') as f:
                    self.ReferenceCITestResults[dm_label] = pickle.load(f)
            except FileNotFoundError:
                self.ReferenceCITestResults[dm_label] = self.build_reference_ci_test_results(dm_label)
                with source_path.open('wb') as f:
                    pickle.dump(self.ReferenceCITestResults[dm_label], f)


    def build_reference_ci_test_results(self, dm_label):
        Omega = TestAlgorithmIPCMBOptimizations.Omega[dm_label]
        datasetmatrix = TestAlgorithmIPCMBOptimizations.DatasetMatrices[dm_label]
        
        bif_file = Path('testfiles', 'bif_files', '{}.bif'.format(dm_label))
        bn = util.read_bif_file(bif_file)
        bn.finalize()

        parameters = dict()
        parameters['target'] = 3
        parameters['ci_test_class'] = mbff.math.G_test__unoptimized.G_test
        parameters['ci_test_significance'] = 0.95
        parameters['debug'] = True
        parameters['omega'] = Omega
        parameters['source_bayesian_network'] = bn
        parameters['ci_test_results'] = list()

        ipcmb = AlgorithmIPCMB(datasetmatrix, parameters)
        ipcmb.select_features()
        return ipcmb.CITest.ci_test_results


    def print_ci_test_results(self, ci_test_results):
        print()
        print('==========')
        print('CI test results:')
        for result in ci_test_results:
            print(result)
        print()
        print('Total: {} CI tests'.format(len(ci_test_results)))



    def configure_datasetmatrix(self, dm_label):
        configuration = dict()
        if dm_label == 'lungcancer':
            configuration['sourcepath'] = Path('testfiles', 'bif_files', 'lungcancer.bif')
            configuration['sample_count'] = int(5e4)
            configuration['random_seed'] = 42*42
            configuration['values_as_indices'] = True
            configuration['objectives'] = []

        if dm_label == 'survey':
            configuration['sourcepath'] = Path('testfiles', 'bif_files', 'survey.bif')
            configuration['sample_count'] = int(5e4)
            configuration['random_seed'] = 42*42
            configuration['values_as_indices'] = True
            configuration['objectives'] = []
        return configuration


    def prepare_datasetmatrices(self):
        TestAlgorithmIPCMBOptimizations.DatasetMatrices = dict()
        TestAlgorithmIPCMBOptimizations.Omega = dict()

        dataset_folder = Path('testfiles', 'tmp', 'test_ipcmb_optimizations_dm')
        for dm_label in self.DatasetMatricesInUse:
            configuration = self.configure_datasetmatrix(dm_label)
            try:
                datasetmatrix = DatasetMatrix(dm_label)
                datasetmatrix.load(dataset_folder)
                TestAlgorithmIPCMBOptimizations.DatasetMatrices[dm_label] = datasetmatrix
            except:
                print('Cannot load DatasetMatrix {} from {}, must rebuild.'.format(dm_label, dataset_folder))
                bayesian_network = util.read_bif_file(configuration['sourcepath'])
                bayesian_network.finalize()
                sbnds = SampledBayesianNetworkDatasetSource(configuration)
                sbnds.reset_random_seed = True
                datasetmatrix = sbnds.create_dataset_matrix(dm_label)
                datasetmatrix.finalize()
                datasetmatrix.save(dataset_folder)
                TestAlgorithmIPCMBOptimizations.DatasetMatrices[dm_label] = datasetmatrix
                print('Dataset rebuilt.')
            TestAlgorithmIPCMBOptimizations.Omega[dm_label] = Omega(configuration['sample_count'])
        TestAlgorithmIPCMBOptimizations.ClassIsSetUp = True

