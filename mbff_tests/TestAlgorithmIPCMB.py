from pprint import pprint
from pathlib import Path

from mbff_tests.TestBase import TestBase

from mbff.math.BayesianNetwork import BayesianNetwork
from mbff.algorithms.mb.ipcmb import algorithm_IPCMB
import mbff.utilities.functions as util


class TestAlgorithmIPCMB(TestBase):

    def test_finding_Markov_blankets(self):
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


    def make_parameters(self, target, bn):
        return {
                'target': target,
                'all_variables': sorted(list(bn.graph_d.keys())),
                'ci_test_builder': lambda dm, param: bn.conditionally_independent,
                'pc_only': False,
                'debug': False
                }

