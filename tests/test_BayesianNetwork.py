import random
from collections import Counter

import numpy
import pytest

from mbtk.structures.BayesianNetwork import BayesianNetwork, VariableNode, ProbabilityDistributionOfVariableNode
from mbtk.structures.Exceptions import BayesianNetworkNotFinalizedError


delta = 0.0000001


def almostEqual(x, y):
    return abs(x - y) < delta


def test_probability_distribution():
    variable = default_variable__unconditioned()
    probdist = variable.probdist

    assert len(probdist.probabilities) == 1
    assert len(probdist.conditioning_variable_nodes) == 0
    assert len(probdist.cummulative_probabilities) == 0

    probdist.finalize()

    assert len(probdist.probabilities) == 1
    assert len(probdist.conditioning_variable_nodes) == 0

    cprobs_compare = zip([0.2, 0.3, 0.8, 1.0], probdist.cummulative_probabilities['<unconditioned>'])
    for pair in cprobs_compare:
        assert abs(pair[0] - pair[1]) < 1e-6



def test_creating_complete_joint_pmf(bn_survey):
    bn = bn_survey

    assert bn.variable_node_names__sampling_order == ['AGE', 'SEX', 'EDU', 'OCC', 'R', 'TRN']
    assert bn.variable_node_names() == ['AGE', 'EDU', 'OCC', 'R', 'SEX', 'TRN']

    total_possible_values_in_bn = 1
    for varnode in bn.variable_nodes.values():
        total_possible_values_in_bn *= len(varnode.values)

    joint_pmf = bn.create_joint_pmf(values_as_indices=False)
    assert sum(joint_pmf.values()) == 1.0
    assert len(joint_pmf) == total_possible_values_in_bn
    assert joint_pmf.labels() == (0, 1, 2, 3, 4, 5)

    test_sample = ('young', 'uni', 'self', 'small', 'F', 'train')
    expected_probability = 0.3 * 0.51 * 0.36 * 0.08 * 0.2 * 0.36
    assert joint_pmf.p(test_sample) == expected_probability

    test_sample = ('young', 'uni', 'emp', 'small', 'F', 'other')
    expected_probability = 0.3 * 0.51 * 0.36 * 0.92 * 0.2 * 0.1
    assert joint_pmf.p(test_sample) == expected_probability

    test_sample = ('young', 'highschool', 'emp', 'small', 'M', 'other')
    expected_probability = 0.3 * 0.49 * 0.75 * 0.96 * 0.25 * 0.1
    assert joint_pmf.p(test_sample) == expected_probability



def test_creating_partial_joint_pmf_variant_1(bn_survey):
    bn = bn_survey
    AGE = bn.variable_nodes_index('AGE')
    SEX = bn.variable_nodes_index('SEX')
    EDU = bn.variable_nodes_index('EDU')

    joint_pmf = bn.create_partial_joint_pmf((AGE, SEX, EDU))
    assert sum(joint_pmf.values()) == 1

    test_sample = (0, 1, 1)  # ('young', 'F', 'uni')
    expected_probability = 0.3 * 0.51 * 0.36
    assert joint_pmf.p(test_sample) == expected_probability

    test_sample = (2, 0, 0)  # ('old', 'M', 'highschool')
    expected_probability = 0.2 * 0.49 * 0.88
    assert joint_pmf.p(test_sample) == expected_probability


def test_creating_partial_joint_pmf_variant_2(bn_survey):
    bn = bn_survey
    EDU = bn.variable_nodes_index('EDU')



    joint_pmf = bn.create_partial_joint_pmf((EDU,))
    assert almostEqual(sum(joint_pmf.values()), 1)

    test_sample = (0,)  # ('highschool',)
    expected_probability = 0 \
        + 0.3 * 0.49 * 0.75  \
        + 0.5 * 0.49 * 0.72  \
        + 0.2 * 0.49 * 0.88  \
        + 0.3 * 0.51 * 0.64  \
        + 0.5 * 0.51 * 0.70  \
        + 0.2 * 0.51 * 0.90  \

    assert joint_pmf.p(test_sample) == expected_probability

    test_sample = (1,)  # ('uni',)
    expected_probability = 0 \
        + 0.3 * 0.49 * 0.25  \
        + 0.5 * 0.49 * 0.28  \
        + 0.2 * 0.49 * 0.12  \
        + 0.3 * 0.51 * 0.36  \
        + 0.5 * 0.51 * 0.30  \
        + 0.2 * 0.51 * 0.10  \

    assert joint_pmf.p(test_sample) == expected_probability


def test_creating_partial_joint_pmf_variant_3(bn_survey):
    global delta
    delta = 0.0000001

    bn = bn_survey
    EDU = bn.variable_nodes_index('EDU')
    joint_pmf = bn.create_partial_joint_pmf((EDU,))
    EDU_p_highschool = joint_pmf.p((0,))
    EDU_p_uni = joint_pmf.p((1,))

    OCC = bn.variable_nodes_index('OCC')
    R = bn.variable_nodes_index('R')
    joint_pmf = bn.create_partial_joint_pmf((OCC, R))
    test_sample = (0, 0)  # ('emp', 'small')
    expected_probability = 0 \
        + EDU_p_highschool * 0.96 * 0.25 \
        + EDU_p_uni * 0.92 * 0.20

    assert almostEqual(joint_pmf.p(test_sample), expected_probability)

    test_sample = (0, 1)  # ('emp', 'big')
    expected_probability = 0 \
        + EDU_p_highschool * 0.96 * 0.75 \
        + EDU_p_uni * 0.92 * 0.80

    assert almostEqual(joint_pmf.p(test_sample), expected_probability)



def test_creating_partial_joint_pmf_variant_4(bn_survey):
    bn = bn_survey
    AGE = bn.variable_nodes_index('AGE')
    EDU = bn.variable_nodes_index('EDU')
    TRN = bn.variable_nodes_index('TRN')

    joint_pmf = bn.create_partial_joint_pmf((AGE, EDU, TRN))
    test_sample = (1, 1, 1)  # ('adult', 'uni', 'train')

    p_train_given_uni = 0 \
        + 0.92 * 0.2 * 0.42  \
        + 0.08 * 0.2 * 0.36  \
        + 0.92 * 0.8 * 0.24  \
        + 0.08 * 0.8 * 0.21

    p_uni_given_adult = 0 \
        + 0.28 * 0.49 \
        + 0.30 * 0.51

    p_adult = 0.5

    expected_probability = 1 \
        * p_train_given_uni \
        * p_uni_given_adult \
        * p_adult

    assert joint_pmf.p(test_sample) == expected_probability


def test_sampling_single(bn_survey):
    bn = bn_survey

    bn.finalized = False
    with pytest.raises(BayesianNetworkNotFinalizedError):
        sample = bn.sample()

    bn.finalized = True

    assert bn.variable_node_names__sampling_order == ['AGE', 'SEX', 'EDU', 'OCC', 'R', 'TRN']

    sample = bn.sample()
    assertValidExpectedSample(sample)



@pytest.mark.slow
def test_sampling_multiple(bn_survey):

    bn = bn_survey

    assert bn.variable_node_names__sampling_order == ['AGE', 'SEX', 'EDU', 'OCC', 'R', 'TRN']

    random.seed(42)
    samples = bn.sample_matrix(100000)
    assert isinstance(samples, numpy.ndarray)
    assert samples.shape == (100000, 6)

    AGEpd = calculate_probdist(samples[:, 0])
    EDUpd = calculate_probdist(samples[:, 1])
    OCCpd = calculate_probdist(samples[:, 2])
    Rpd = calculate_probdist(samples[:, 3])
    SEXpd = calculate_probdist(samples[:, 4])
    TRNpd = calculate_probdist(samples[:, 5])

    delta = 0.003

    def almostEqual(x, y):
        return abs(x - y) < delta

    assert almostEqual(AGEpd[0], 0.3)
    assert almostEqual(AGEpd[1], 0.5)
    assert almostEqual(AGEpd[2], 0.2)

    assert almostEqual(SEXpd[0], 0.49)
    assert almostEqual(SEXpd[1], 0.51)

    EDU_p_highschool = (
        0.3 * 0.49 * 0.75 +
        0.5 * 0.49 * 0.72 +
        0.2 * 0.49 * 0.88 +
        0.3 * 0.51 * 0.64 +
        0.5 * 0.51 * 0.7 +
        0.2 * 0.51 * 0.9)
    assert almostEqual(EDUpd[0], EDU_p_highschool)

    EDU_p_uni = (
        0.3 * 0.49 * 0.25 +
        0.5 * 0.49 * 0.28 +
        0.2 * 0.49 * 0.12 +
        0.3 * 0.51 * 0.36 +
        0.5 * 0.51 * 0.3 +
        0.2 * 0.51 * 0.1)
    assert almostEqual(EDUpd[1], EDU_p_uni)

    OCC_p_emp = (
        EDU_p_highschool * 0.96 +
        EDU_p_uni * 0.92)
    assert almostEqual(OCCpd[0], OCC_p_emp)

    OCC_p_self = (
        EDU_p_highschool * 0.04 +
        EDU_p_uni * 0.08)
    assert almostEqual(OCCpd[1], OCC_p_self)

    R_p_small = (
        EDU_p_highschool * 0.25 +
        EDU_p_uni * 0.2)
    assert almostEqual(Rpd[0], R_p_small)

    R_p_big = (
        EDU_p_highschool * 0.75 +
        EDU_p_uni * 0.8)
    assert almostEqual(Rpd[1], R_p_big)

    TRN_p_car = (
        OCC_p_emp * R_p_small * 0.48 +
        OCC_p_self * R_p_small * 0.56 +
        OCC_p_emp * R_p_big * 0.58 +
        OCC_p_self * R_p_big * 0.70)
    assert almostEqual(TRNpd[0], TRN_p_car)

    TRN_p_train = (
        OCC_p_emp * R_p_small * 0.42 +
        OCC_p_self * R_p_small * 0.36 +
        OCC_p_emp * R_p_big * 0.24 +
        OCC_p_self * R_p_big * 0.21)
    assert almostEqual(TRNpd[1], TRN_p_train)

    TRN_p_other = (
        OCC_p_emp * R_p_small * 0.10 +
        OCC_p_self * R_p_small * 0.08 +
        OCC_p_emp * R_p_big * 0.18 +
        OCC_p_self * R_p_big * 0.09)

    assert almostEqual(TRNpd[2], TRN_p_other)



def test_variable_IDs(bn_survey, bn_lungcancer, bn_alarm):
    bn = bn_survey
    assert bn.variable_nodes['AGE'].ID == 0
    assert bn.variable_nodes['EDU'].ID == 1
    assert bn.variable_nodes['OCC'].ID == 2
    assert bn.variable_nodes['R'].ID == 3
    assert bn.variable_nodes['SEX'].ID == 4
    assert bn.variable_nodes['TRN'].ID == 5

    bn = bn_lungcancer
    assert bn.variable_nodes['ASIA'].ID == 0
    assert bn.variable_nodes['BRONC'].ID == 1
    assert bn.variable_nodes['DYSP'].ID == 2
    assert bn.variable_nodes['EITHER'].ID == 3
    assert bn.variable_nodes['LUNG'].ID == 4
    assert bn.variable_nodes['SMOKE'].ID == 5
    assert bn.variable_nodes['TUB'].ID == 6
    assert bn.variable_nodes['XRAY'].ID == 7

    bn = bn_alarm
    assert bn.variable_nodes['ANAPHYLAXIS'].ID == 0
    assert bn.variable_nodes['ARTCO2'].ID == 1
    assert bn.variable_nodes['BP'].ID == 2
    assert bn.variable_nodes['CATECHOL'].ID == 3
    assert bn.variable_nodes['CO'].ID == 4
    assert bn.variable_nodes['CVP'].ID == 5
    assert bn.variable_nodes['DISCONNECT'].ID == 6
    assert bn.variable_nodes['ERRCAUTER'].ID == 7
    assert bn.variable_nodes['ERRLOWOUTPUT'].ID == 8
    assert bn.variable_nodes['EXPCO2'].ID == 9
    assert bn.variable_nodes['FIO2'].ID == 10
    assert bn.variable_nodes['HISTORY'].ID == 11
    assert bn.variable_nodes['HR'].ID == 12
    assert bn.variable_nodes['HRBP'].ID == 13
    assert bn.variable_nodes['HREKG'].ID == 14
    assert bn.variable_nodes['HRSAT'].ID == 15
    assert bn.variable_nodes['HYPOVOLEMIA'].ID == 16
    assert bn.variable_nodes['INSUFFANESTH'].ID == 17
    assert bn.variable_nodes['INTUBATION'].ID == 18
    assert bn.variable_nodes['KINKEDTUBE'].ID == 19
    assert bn.variable_nodes['LVEDVOLUME'].ID == 20
    assert bn.variable_nodes['LVFAILURE'].ID == 21
    assert bn.variable_nodes['MINVOL'].ID == 22
    assert bn.variable_nodes['MINVOLSET'].ID == 23
    assert bn.variable_nodes['PAP'].ID == 24
    assert bn.variable_nodes['PCWP'].ID == 25
    assert bn.variable_nodes['PRESS'].ID == 26
    assert bn.variable_nodes['PULMEMBOLUS'].ID == 27
    assert bn.variable_nodes['PVSAT'].ID == 28
    assert bn.variable_nodes['SAO2'].ID == 29
    assert bn.variable_nodes['SHUNT'].ID == 30
    assert bn.variable_nodes['STROKEVOLUME'].ID == 31
    assert bn.variable_nodes['TPR'].ID == 32
    assert bn.variable_nodes['VENTALV'].ID == 33
    assert bn.variable_nodes['VENTLUNG'].ID == 34
    assert bn.variable_nodes['VENTMACH'].ID == 35
    assert bn.variable_nodes['VENTTUBE'].ID == 36



def test_directed_graph_building(bn_survey, bn_lungcancer, bn_alarm):
    bn = bn_survey
    expected_directed_graph = {
        0: [1],
        1: [2, 3],
        2: [5],
        3: [5],
        4: [1],
        5: []
    }
    assert bn.graph_d == expected_directed_graph

    bn = bn_lungcancer
    bn.finalize()
    expected_directed_graph = {
        0: [6],
        1: [2],
        2: [],
        3: [2, 7],
        4: [3],
        5: [1, 4],
        6: [3],
        7: []
    }
    assert bn.graph_d == expected_directed_graph

    bn = bn_alarm
    expected_directed_graph = {
        0: [32],
        1: [3, 9],
        2: [],
        3: [12],
        4: [2],
        5: [],
        6: [36],
        7: [14, 15],
        8: [13],
        9: [],
        10: [28],
        11: [],
        12: [4, 13, 14, 15],
        13: [],
        14: [],
        15: [],
        16: [20, 31],
        17: [3],
        18: [22, 26, 30, 33, 34],
        19: [26, 34],
        20: [5, 25],
        21: [11, 20, 31],
        22: [],
        23: [35],
        24: [],
        25: [],
        26: [],
        27: [24, 30],
        28: [29],
        29: [3],
        30: [29],
        31: [4],
        32: [2, 3],
        33: [1, 28],
        34: [9, 22, 33],
        35: [36],
        36: [26, 34],
    }
    assert bn.graph_d == expected_directed_graph

    expected_paths = [
        [0, 32, 2],
        [0, 32, 3, 12, 4, 2]
    ]
    assert bn.find_all_directed_paths(0, 2) == expected_paths

    expected_paths = [
        [18, 30, 29, 3, 12, 4],
        [18, 33, 1, 3, 12, 4],
        [18, 33, 28, 29, 3, 12, 4],
        [18, 34, 33, 1, 3, 12, 4],
        [18, 34, 33, 28, 29, 3, 12, 4]
    ]
    assert bn.find_all_directed_paths(18, 4) == expected_paths

    expected_paths = []
    assert bn.find_all_directed_paths(18, 23) == expected_paths

    expected_paths = [[6, 36]]
    assert bn.find_all_directed_paths(6, 36) == expected_paths



def test_undirected_graph_building(bn_alarm):
    bn = bn_alarm
    expected_undirected_graph = {
        0: [32],
        1: [3, 9, 33],
        2: [4, 32],
        3: [1, 12, 17, 29, 32],
        4: [2, 12, 31],
        5: [20],
        6: [36],
        7: [14, 15],
        8: [13],
        9: [1, 34],
        10: [28],
        11: [21],
        12: [3, 4, 13, 14, 15],
        13: [8, 12],
        14: [7, 12],
        15: [7, 12],
        16: [20, 31],
        17: [3],
        18: [22, 26, 30, 33, 34],
        19: [26, 34],
        20: [5, 16, 21, 25],
        21: [11, 20, 31],
        22: [18, 34],
        23: [35],
        24: [27],
        25: [20],
        26: [18, 19, 36],
        27: [24, 30],
        28: [10, 29, 33],
        29: [3, 28, 30],
        30: [18, 27, 29],
        31: [4, 16, 21],
        32: [0, 2, 3],
        33: [1, 18, 28, 34],
        34: [9, 18, 19, 22, 33, 36],
        35: [23, 36],
        36: [6, 26, 34, 35],
    }
    assert bn.graph_u == expected_undirected_graph


def test_building_from_directed_graph():
    graph = {
        0: [32],
        1: [3, 9],
        2: [],
        3: [12],
        4: [2],
        5: [],
        6: [36],
        7: [14, 15],
        8: [13],
        9: [],
        10: [28],
        11: [],
        12: [4, 13, 14, 15],
        13: [],
        14: [],
        15: [],
        16: [20, 31],
        17: [3],
        18: [22, 26, 30, 33, 34],
        19: [26, 34],
        20: [5, 25],
        21: [11, 20, 31],
        22: [],
        23: [35],
        24: [],
        25: [],
        26: [],
        27: [24, 30],
        28: [29],
        29: [3],
        30: [29],
        31: [4],
        32: [2, 3],
        33: [1, 28],
        34: [9, 22, 33],
        35: [36],
        36: [26, 34],
    }
    expected_undirected_graph = {
        0: [32],
        1: [3, 9, 33],
        2: [4, 32],
        3: [1, 12, 17, 29, 32],
        4: [2, 12, 31],
        5: [20],
        6: [36],
        7: [14, 15],
        8: [13],
        9: [1, 34],
        10: [28],
        11: [21],
        12: [3, 4, 13, 14, 15],
        13: [8, 12],
        14: [7, 12],
        15: [7, 12],
        16: [20, 31],
        17: [3],
        18: [22, 26, 30, 33, 34],
        19: [26, 34],
        20: [5, 16, 21, 25],
        21: [11, 20, 31],
        22: [18, 34],
        23: [35],
        24: [27],
        25: [20],
        26: [18, 19, 36],
        27: [24, 30],
        28: [10, 29, 33],
        29: [3, 28, 30],
        30: [18, 27, 29],
        31: [4, 16, 21],
        32: [0, 2, 3],
        33: [1, 18, 28, 34],
        34: [9, 18, 19, 22, 33, 36],
        35: [23, 36],
        36: [6, 26, 34, 35],
    }
    bn = BayesianNetwork('testnet_graph')
    bn.from_directed_graph(graph)
    assert bn.graph_d == graph
    assert bn.graph_u == expected_undirected_graph

    graph = {
        1: [2, 3],
        2: [4],
        3: [4],
        4: [5],
        5: []
    }
    expected_undirected_graph = {
        1: [2, 3],
        2: [1, 4],
        3: [1, 4],
        4: [2, 3, 5],
        5: [4]
    }

    bn = BayesianNetwork('testnet_graph')
    bn.from_directed_graph(graph)
    assert bn.graph_d == graph
    assert bn.graph_u == expected_undirected_graph



def test_d_separation__custom_graph_1():
    # Simple graph, from "Probabilistic Reasoning in Intelligent Systems"
    # by Judea Pearl, 1988
    graph = {
        1: [2, 3],
        2: [4],
        3: [4],
        4: [5],
        5: []
    }
    bn = BayesianNetwork('testnet')
    bn.from_directed_graph(graph)

    assert bn.d_separated(2, [1], 3) is True
    assert bn.d_separated(2, [1, 5], 3) is False
    assert bn.d_separated(1, [], 2) is False
    assert bn.d_separated(1, [], 3) is False
    assert bn.d_separated(1, [], 4) is False
    assert bn.d_separated(1, [], 5) is False

    assert bn.d_separated(1, [4], 5) is True
    assert bn.d_separated(2, [], 3) is False
    assert bn.d_separated(2, [4], 3) is False

    assert bn.d_separated(5, [2], 1) is False
    assert bn.d_separated(5, [3], 1) is False
    assert bn.d_separated(5, [2, 3], 1) is True

    assert bn.d_separated(3, [1, 2], 5) is False



def test_d_separation__custom_graph_2():
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

    assert bn.d_separated(3, [0], 1) is False
    assert bn.d_separated(3, [], 2) is False
    assert bn.d_separated(3, [5], 2) is False
    assert bn.d_separated(3, [1], 2) is True



def test_d_separation__custom_graph_3():
    # Simple graph taken from the PCMB article, where authors provide
    # examples to illustrate the flaws found in MMMB and HITON.
    graph_a = {
        0: [1, 2],
        1: [3],
        2: [3],
        3: [],
        4: [1]
    }
    bn = BayesianNetwork('testnet_a')
    bn.from_directed_graph(graph_a)
    bn.debug = True
    assert bn.d_separated(4, [1], 3) is False
    # Assert that [0, 1] is the Markov blanket of 4, by the Intersection
    # Property.
    assert bn.d_separated(4, [0, 1, 3], 2) is True
    assert bn.d_separated(4, [0, 1, 2], 3) is True
    # Assert that [0, 1] is the Markov blanket of 4, by the Contraction
    # Property.
    assert bn.d_separated(4, [0, 1, 3], 2) is True
    assert bn.d_separated(4, [0, 1], 3) is True



def test_d_separation__lc_repaired(bn_lc_repaired):
    # Simple graph, similar to 'lungcancer' (a.k.a. 'asia'), but with no
    # deterministic nodes.
    bn = bn_lc_repaired

    assert bn.d_separated(3, [0], 6) is False



def test_d_separation__alarm(bn_alarm):
    # Moderately complex graph, as generated by the 'alarm' Bayesian
    # network, from
    # http://www.bnlearn.com/bnrepository/discrete-medium.html#alarm
    bn = bn_alarm

    assert bn.d_separated(0, [], 1) is True
    assert bn.d_separated(0, [12], 1) is False
    assert bn.d_separated(0, [32, 12], 1) is True

    assert bn.d_separated(24, [], 26) is True
    assert bn.d_separated(24, [29], 26) is False

    assert bn.d_separated(17, [], 18) is True
    assert bn.d_separated(17, [3], 18) is False
    assert bn.d_separated(17, [12], 18) is False

    assert bn.d_separated(10, [], 26) is True
    assert bn.d_separated(10, [29], 26) is False
    assert bn.d_separated(10, [29, 19], 26) is False
    assert bn.d_separated(10, [29, 19, 36], 26) is False
    assert bn.d_separated(10, [18, 29, 19, 36], 26) is True



def test_d_separation__lungcancer(bn_lungcancer):
    # Another simple graph, as generated by the 'asia' (or 'lungcancer')
    # Bayesian network, from
    # http://www.bnlearn.com/bnrepository/discrete-medium.html#alarm
    bn = bn_lungcancer

    assert bn.d_separated(0, [], 6) is False



def test_find_all_paths(bn_alarm):
    bn = bn_alarm

    start_variable = 'MINVOL'
    end_variable = 'SAO2'

    start_index = bn.variable_nodes_index(start_variable)
    end_index = bn.variable_nodes_index(end_variable)

    paths = bn.find_all_undirected_paths(start_index, end_index)
    expected_paths = expected_paths_bn_alarm_MINVOL_to_SAO2()

    assert paths == expected_paths



def calculate_probdist(column):
    counter = Counter()
    counter.update(column.T.tolist())
    probdist = {}
    for value in counter.keys():
        probdist[value] = 1.0 * counter[value] / column.size
    return probdist



def default_variable__unconditioned():
    variable = VariableNode('ASDF')
    variable.values = ['rocket', 'carbohydrate', 'albatross', 'oxygen']
    variable.probdist = ProbabilityDistributionOfVariableNode(variable)
    variable.probdist.probabilities = {'<unconditioned>': [0.2, 0.1, 0.5, 0.2]}
    return variable



def assertValidExpectedSample(sample):
    assert isinstance(sample, dict)
    assert len(sample) == 6
    assert sample['AGE'] in ['young', 'adult', 'old']
    assert sample['SEX'] in ['M', 'F']
    assert sample['EDU'] in ['highschool', 'uni']
    assert sample['OCC'] in ['emp', 'self']
    assert sample['R'] in ['small', 'big']
    assert sample['TRN'] in ['car', 'train', 'other']



def expected_paths_bn_alarm_MINVOL_to_SAO2():
    return [
        (22, 18, 26, 19, 34, 9, 1, 3, 29),
        (22, 18, 26, 19, 34, 9, 1, 33, 28, 29),
        (22, 18, 26, 19, 34, 33, 1, 3, 29),
        (22, 18, 26, 19, 34, 33, 28, 29),
        (22, 18, 26, 36, 34, 9, 1, 3, 29),
        (22, 18, 26, 36, 34, 9, 1, 33, 28, 29),
        (22, 18, 26, 36, 34, 33, 1, 3, 29),
        (22, 18, 26, 36, 34, 33, 28, 29),
        (22, 18, 30, 29),
        (22, 18, 33, 1, 3, 29),
        (22, 18, 33, 28, 29),
        (22, 18, 33, 34, 9, 1, 3, 29),
        (22, 18, 34, 9, 1, 3, 29),
        (22, 18, 34, 9, 1, 33, 28, 29),
        (22, 18, 34, 33, 1, 3, 29),
        (22, 18, 34, 33, 28, 29),
        (22, 34, 9, 1, 3, 29),
        (22, 34, 9, 1, 33, 18, 30, 29),
        (22, 34, 9, 1, 33, 28, 29),
        (22, 34, 18, 30, 29),
        (22, 34, 18, 33, 1, 3, 29),
        (22, 34, 18, 33, 28, 29),
        (22, 34, 19, 26, 18, 30, 29),
        (22, 34, 19, 26, 18, 33, 1, 3, 29),
        (22, 34, 19, 26, 18, 33, 28, 29),
        (22, 34, 33, 1, 3, 29),
        (22, 34, 33, 18, 30, 29),
        (22, 34, 33, 28, 29),
        (22, 34, 36, 26, 18, 30, 29),
        (22, 34, 36, 26, 18, 33, 1, 3, 29),
        (22, 34, 36, 26, 18, 33, 28, 29)
    ]
