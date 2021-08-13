import pytest

from mbtk.structures.BayesianNetwork import BayesianNetwork
from mbtk.math.DSeparationCITest import DSeparationCITest
from mbtk.algorithms.mb.ipcmb import AlgorithmIPCMB

import mbtk.math.G_test__unoptimized
import mbtk.math.DoFCalculators


@pytest.mark.slow
def test_ipcmb_finding_Markov_blankets_in_datasetmatrix(ds_survey_2e4):
    omega = ds_survey_2e4.omega
    datasetmatrix = ds_survey_2e4.datasetmatrix
    bn = ds_survey_2e4.bayesiannetwork

    parameters = dict()
    parameters['target'] = 3
    parameters['ci_test_class'] = mbtk.math.G_test__unoptimized.G_test
    parameters['ci_test_significance'] = 0.90
    parameters['ci_test_debug'] = 0
    parameters['omega'] = omega
    parameters['source_bayesian_network'] = bn
    parameters['ci_test_dof_calculator_class'] = mbtk.math.DoFCalculators.StructuralDoF

    ipcmb = AlgorithmIPCMB(datasetmatrix, parameters)
    mb = ipcmb.discover_mb()
    assert mb == [1, 2, 5]



def test_ipcmb_finding_Markov_blankets_in_graphs__imitating_survey():
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

    parameters = make_parameters(3, bn)
    mb = AlgorithmIPCMB(None, parameters).discover_mb()
    assert mb == [1, 2, 5]

    parameters = make_parameters(1, bn)
    mb = AlgorithmIPCMB(None, parameters).discover_mb()
    assert mb == [0, 2, 3, 4]

    # Remove the edge 1 → 2 from the Bayesian network.
    graph[1] = [3]
    bn = BayesianNetwork('testnet')
    bn.from_directed_graph(graph)

    parameters = make_parameters(3, bn)
    mb = AlgorithmIPCMB(None, parameters).discover_mb()
    assert mb == [1, 2, 5]

    parameters = make_parameters(1, bn)
    mb = AlgorithmIPCMB(None, parameters).discover_mb()
    assert mb == [0, 3, 4]

    # Replace the edge from 1 → 3 with 1 → 2.
    graph[1] = [2]
    bn = BayesianNetwork('testnet')
    bn.from_directed_graph(graph)

    parameters = make_parameters(3, bn)
    mb = AlgorithmIPCMB(None, parameters).discover_mb()
    assert mb == [2, 5]

    parameters = make_parameters(1, bn)
    mb = AlgorithmIPCMB(None, parameters).discover_mb()
    assert mb == [0, 2, 4]



def test_ipcmb_finding_Markov_blankets_in_graphs__as_in_pcmb_article():
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

    parameters = make_parameters(4, bn)
    parameters['pc_only'] = True
    pc = AlgorithmIPCMB(None, parameters).discover_mb()
    assert pc == [1]

    parameters = make_parameters(4, bn)
    mb = AlgorithmIPCMB(None, parameters).discover_mb()
    assert mb == [0, 1]

    parameters = make_parameters(0, bn)
    mb = AlgorithmIPCMB(None, parameters).discover_mb()
    assert mb == [1, 2, 4]

    parameters = make_parameters(2, bn)
    mb = AlgorithmIPCMB(None, parameters).discover_mb()
    assert mb == [0, 1, 3]

    parameters = make_parameters(1, bn)
    mb = AlgorithmIPCMB(None, parameters).discover_mb()
    assert mb == [0, 2, 3, 4]



@pytest.mark.slow
def test_ipcmb_on_alarm(bn_alarm):
    parameters = make_parameters(22, bn_alarm)
    mb = AlgorithmIPCMB(None, parameters).discover_mb()
    assert mb == [18, 34]

    parameters = make_parameters(1, bn_alarm)
    mb = AlgorithmIPCMB(None, parameters).discover_mb()
    assert mb == [3, 9, 17, 29, 32, 33, 34]

    parameters = make_parameters(17, bn_alarm)
    mb = AlgorithmIPCMB(None, parameters).discover_mb()
    assert mb == [1, 3, 29, 32]

    parameters = make_parameters(24, bn_alarm)
    mb = AlgorithmIPCMB(None, parameters).discover_mb()
    assert mb == [27]

    parameters = make_parameters(34, bn_alarm)
    mb = AlgorithmIPCMB(None, parameters).discover_mb()
    assert mb == [1, 9, 18, 19, 22, 33, 36]



def make_parameters(target, bn):
    return {
        'target': target,
        'all_variables': sorted(list(bn.graph_d.keys())),
        'ci_test_class': DSeparationCITest,
        'source_bayesian_network': bn,
        'pc_only': False,
        'ci_test_debug': 0,
        'algorithm_debug': 0
    }
