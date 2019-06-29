import unittest
from pprint import pprint
from pathlib import Path

from mbff_tests.TestBase import TestBase

from mbff.dataset.DatasetMatrix import DatasetMatrix
from mbff.math.BayesianNetwork import BayesianNetwork
from mbff.math.Variable import Variable, JointVariables, Omega
from mbff.dataset.sources.SampledBayesianNetworkDatasetSource import SampledBayesianNetworkDatasetSource
from mbff.algorithms.mb.ipcmb import algorithm_IPCMB
import mbff.math.G_test__unoptimized
import mbff.utilities.functions as util


class TestAlgorithmIPCMB(TestBase):

    ClassIsSetUp = False
    DatasetMatrices = None
    Omega = None


    def setUp(self):
        if not TestAlgorithmIPCMB.ClassIsSetUp:
            self.prepare_datasetmatrices()


    def test_finding_Markov_blankets_in_graphs(self):
        # Simple graph imitating the 'survey' Bayesian network, from
        # http://www.bnlearn.com/bnrepository/discrete-small.html#survey
        graph = {
                0: [1],
                4: [1],
                1: [2, 3],
                2: [5],
                3: [5]
                }
        bn = BayesianNetwork('testnet')
        bn.from_directed_graph(graph)

        parameters = self.make_parameters(3, bn)
        mb = algorithm_IPCMB(None, parameters)
        self.assertEqual([1, 2, 5], mb)

        parameters = self.make_parameters(1, bn)
        mb = algorithm_IPCMB(None, parameters)
        self.assertEqual([0, 2, 3, 4], mb)

        # Remove the edge 1 → 2 from the Bayesian network.
        graph[1] = [3]
        bn = BayesianNetwork('testnet')
        bn.from_directed_graph(graph)

        parameters = self.make_parameters(3, bn)
        mb = algorithm_IPCMB(None, parameters)
        self.assertEqual([1, 2, 5], mb)

        parameters = self.make_parameters(1, bn)
        mb = algorithm_IPCMB(None, parameters)
        self.assertEqual([0, 3, 4], mb)

        # Replace the edge from 1 → 3 with 1 → 2.
        graph[1] = [2]
        bn = BayesianNetwork('testnet')
        bn.from_directed_graph(graph)

        parameters = self.make_parameters(3, bn)
        mb = algorithm_IPCMB(None, parameters)
        self.assertEqual([2, 5], mb)

        parameters = self.make_parameters(1, bn)
        mb = algorithm_IPCMB(None, parameters)
        self.assertEqual([0, 2, 4], mb)

        # Test IPC-MB with the graphs proposed in the PCMB article, used to
        # illustrate the flaws of MMMB and HITON.
        graph_a = {
                0: [1, 2],
                1: [3],
                2: [3],
                3: [],
                4: [1]
                }
        bn = BayesianNetwork('testnet_a')
        bn.from_directed_graph(graph_a)

        parameters = self.make_parameters(4, bn)
        parameters['pc_only'] = True
        pc = algorithm_IPCMB(None, parameters)
        self.assertEqual([1], pc)

        parameters = self.make_parameters(4, bn)
        mb = algorithm_IPCMB(None, parameters)
        self.assertEqual([0, 1], mb)

        parameters = self.make_parameters(0, bn)
        mb = algorithm_IPCMB(None, parameters)
        self.assertEqual([1, 2, 4], mb)

        parameters = self.make_parameters(2, bn)
        mb = algorithm_IPCMB(None, parameters)
        self.assertEqual([0, 1, 3], mb)

        parameters = self.make_parameters(1, bn)
        mb = algorithm_IPCMB(None, parameters)
        self.assertEqual([0, 2, 3, 4], mb)

        # Test IPC-MB with the ALARM network.
        bn = util.read_bif_file(Path('testfiles', 'bif_files', 'alarm.bif'))
        bn.finalize()

        parameters = self.make_parameters(22, bn)
        mb = algorithm_IPCMB(None, parameters)
        self.assertEqual([18, 34], mb)

        parameters = self.make_parameters(1, bn)
        mb = algorithm_IPCMB(None, parameters)
        self.assertEqual([3, 9, 17, 29, 32, 33, 34], mb)

        parameters = self.make_parameters(17, bn)
        mb = algorithm_IPCMB(None, parameters)
        self.assertEqual([1, 3, 29, 32], mb)

        parameters = self.make_parameters(24, bn)
        mb = algorithm_IPCMB(None, parameters)
        self.assertEqual([27], mb)

        parameters = self.make_parameters(20, bn)
        mb = algorithm_IPCMB(None, parameters)
        self.assertEqual([5, 16, 21, 25], mb)

        parameters = self.make_parameters(16, bn)
        mb = algorithm_IPCMB(None, parameters)
        self.assertEqual([20, 21, 31], mb)

        parameters = self.make_parameters(34, bn)
        mb = algorithm_IPCMB(None, parameters)
        self.assertEqual([1, 9, 18, 19, 22, 33, 36], mb)


    @unittest.skipIf(TestBase.tag_excluded('ipcmb_run_with_dataset'), 'Tests running IPC-MB on datasets are excluded')
    def test_finding_Markov_blankets_in_datasetmatrix(self):
        Omega = TestAlgorithmIPCMB.Omega['lungcancer']
        datasetmatrix = TestAlgorithmIPCMB.DatasetMatrices['lungcancer']

        lungcancer_bif = Path('testfiles', 'bif_files', 'lungcancer.bif')
        bn = util.read_bif_file(lungcancer_bif)
        bn.finalize()

        parameters = dict()
        parameters['target'] = 3
        parameters['ci_test_class'] = mbff.math.G_test__unoptimized.G_test
        parameters['ci_test_significance'] = 0.99
        parameters['debug'] = True
        parameters['omega'] = Omega
        parameters['source_bayesian_network'] = bn
        parameters['ci_test_results'] = list()

        mb = algorithm_IPCMB(datasetmatrix, parameters)

        ci_test_results = parameters['ci_test_results']
        print()
        print('==========')
        print('CI test trace:')
        for result in ci_test_results:
            print(result)
        print()
        print('Total: {} CI tests'.format(len(ci_test_results)))

        # FIXME figure out what's the correct MB of the node 3 in the Bayesian
        # network 'lungcancer'
        self.assertEqual('FIXME', mb)


    def make_parameters(self, target, bn):
        return {
                'target': target,
                'all_variables': sorted(list(bn.graph_d.keys())),
                'ci_test_builder': lambda dm, param: bn.conditionally_independent,
                'pc_only': False,
                'debug': False
                }


    def configure_datasetmatrix(self, dm_label):
        configuration = {}
        if dm_label == 'lungcancer':
            configuration['sourcepath'] = Path('testfiles', 'bif_files', 'lungcancer.bif')
            configuration['sample_count'] = int(5e5)
            configuration['random_seed'] = 42*42
            configuration['values_as_indices'] = True
            configuration['objectives'] = []

        if dm_label == 'survey':
            configuration['sourcepath'] = Path('testfiles', 'bif_files', 'survey.bif')
            configuration['sample_count'] = int(1e6)
            configuration['random_seed'] = 42*42
            configuration['values_as_indices'] = True
            configuration['objectives'] = []
        return configuration


    def prepare_datasetmatrices(self):
        TestAlgorithmIPCMB.DatasetMatrices = {}
        TestAlgorithmIPCMB.Omega = {}

        dataset_folder = Path('testfiles', 'tmp', 'test_ipcmb_dm')
        for dm_label in ['lungcancer']:
            configuration = self.configure_datasetmatrix(dm_label)
            try:
                datasetmatrix = DatasetMatrix(dm_label)
                datasetmatrix.load(dataset_folder)
                TestAlgorithmIPCMB.DatasetMatrices[dm_label] = datasetmatrix
            except:
                bayesian_network = util.read_bif_file(configuration['sourcepath'])
                bayesian_network.finalize()
                sbnds = SampledBayesianNetworkDatasetSource(configuration)
                sbnds.reset_random_seed = True
                datasetmatrix = sbnds.create_dataset_matrix(dm_label)
                datasetmatrix.finalize()
                datasetmatrix.save(dataset_folder)
                TestAlgorithmIPCMB.DatasetMatrices[dm_label] = datasetmatrix
            TestAlgorithmIPCMB.Omega[dm_label] = Omega(configuration['sample_count'])
        TestAlgorithmIPCMB.ClassIsSetUp = True

