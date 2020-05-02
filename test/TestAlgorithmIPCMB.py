import unittest
from pathlib import Path


from mbff_tests.TestBase import TestBase

from mbff.structures.BayesianNetwork import BayesianNetwork
from mbff.math.DSeparationCITest import DSeparationCITest
from mbff.algorithms.mb.ipcmb import AlgorithmIPCMB

import mbff.math.G_test__unoptimized
import mbff.math.DoFCalculators
import mbff.utilities.functions as util


@unittest.skipIf(TestBase.tag_excluded('ipcmb_run'), 'Tests running IPC-MB are excluded')
class TestAlgorithmIPCMB(TestBase):

    @classmethod
    def initTestResources(testClass):
        super(TestAlgorithmIPCMB, testClass).initTestResources()
        testClass.DatasetsInUse = ['survey']
        testClass.DatasetMatrixFolder = Path('testfiles', 'tmp', 'test_ipcmb_dm')


    @classmethod
    def configure_dataset(testClass, dm_label):
        configuration = {}
        if dm_label == 'survey':
            configuration['sourcepath'] = Path('testfiles', 'bif_files', 'survey.bif')
            configuration['sample_count'] = int(2e4)
            configuration['random_seed'] = 42 * 42
            configuration['values_as_indices'] = True
            configuration['objectives'] = []
        return configuration


    def test_finding_Markov_blankets_in_datasetmatrix(self):
        Omega = self.OmegaVariables['survey']
        datasetmatrix = self.DatasetMatrices['survey']
        bn = self.BayesianNetworks['survey']

        parameters = dict()
        parameters['target'] = 3
        parameters['ci_test_class'] = mbff.math.G_test__unoptimized.G_test
        parameters['ci_test_significance'] = 0.90
        parameters['ci_test_debug'] = 0
        parameters['omega'] = Omega
        parameters['source_bayesian_network'] = bn
        parameters['ci_test_dof_calculator_class'] = mbff.math.DoFCalculators.StructuralDoF

        ipcmb = AlgorithmIPCMB(datasetmatrix, parameters)
        mb = ipcmb.select_features()
        self.assertEqual([1, 2, 5], mb)


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
        mb = AlgorithmIPCMB(None, parameters).select_features()
        self.assertEqual([1, 2, 5], mb)

        parameters = self.make_parameters(1, bn)
        mb = AlgorithmIPCMB(None, parameters).select_features()
        self.assertEqual([0, 2, 3, 4], mb)

        # Remove the edge 1 → 2 from the Bayesian network.
        graph[1] = [3]
        bn = BayesianNetwork('testnet')
        bn.from_directed_graph(graph)

        parameters = self.make_parameters(3, bn)
        mb = AlgorithmIPCMB(None, parameters).select_features()
        self.assertEqual([1, 2, 5], mb)

        parameters = self.make_parameters(1, bn)
        mb = AlgorithmIPCMB(None, parameters).select_features()
        self.assertEqual([0, 3, 4], mb)

        # Replace the edge from 1 → 3 with 1 → 2.
        graph[1] = [2]
        bn = BayesianNetwork('testnet')
        bn.from_directed_graph(graph)

        parameters = self.make_parameters(3, bn)
        mb = AlgorithmIPCMB(None, parameters).select_features()
        self.assertEqual([2, 5], mb)

        parameters = self.make_parameters(1, bn)
        mb = AlgorithmIPCMB(None, parameters).select_features()
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
        pc = AlgorithmIPCMB(None, parameters).select_features()
        self.assertEqual([1], pc)

        parameters = self.make_parameters(4, bn)
        mb = AlgorithmIPCMB(None, parameters).select_features()
        self.assertEqual([0, 1], mb)

        parameters = self.make_parameters(0, bn)
        mb = AlgorithmIPCMB(None, parameters).select_features()
        self.assertEqual([1, 2, 4], mb)

        parameters = self.make_parameters(2, bn)
        mb = AlgorithmIPCMB(None, parameters).select_features()
        self.assertEqual([0, 1, 3], mb)

        parameters = self.make_parameters(1, bn)
        mb = AlgorithmIPCMB(None, parameters).select_features()
        self.assertEqual([0, 2, 3, 4], mb)

        # Test IPC-MB with the ALARM network.
        bn = util.read_bif_file(Path('testfiles', 'bif_files', 'alarm.bif'))
        bn.finalize()

        parameters = self.make_parameters(22, bn)
        mb = AlgorithmIPCMB(None, parameters).select_features()
        self.assertEqual([18, 34], mb)

        parameters = self.make_parameters(1, bn)
        mb = AlgorithmIPCMB(None, parameters).select_features()
        self.assertEqual([3, 9, 17, 29, 32, 33, 34], mb)

        parameters = self.make_parameters(17, bn)
        mb = AlgorithmIPCMB(None, parameters).select_features()
        self.assertEqual([1, 3, 29, 32], mb)

        parameters = self.make_parameters(24, bn)
        mb = AlgorithmIPCMB(None, parameters).select_features()
        self.assertEqual([27], mb)

        parameters = self.make_parameters(34, bn)
        mb = AlgorithmIPCMB(None, parameters).select_features()
        self.assertEqual([1, 9, 18, 19, 22, 33, 36], mb)


    def make_parameters(self, target, bn):
        return {
            'target': target,
            'all_variables': sorted(list(bn.graph_d.keys())),
            'ci_test_class': DSeparationCITest,
            'source_bayesian_network': bn,
            'pc_only': False,
            'ci_test_debug': 0,
            'algorithm_debug': 0
        }