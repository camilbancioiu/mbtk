import random
from pathlib import Path
from collections import Counter

import numpy
import unittest

from mbff_tests.TestBase import TestBase

import mbff.utilities.functions as util
from mbff.math.BayesianNetwork import *
from mbff.math.Exceptions import *


class TestBayesianNetwork(TestBase):

    def test_probability_distribution(self):
        variable = self.default_variable__unconditioned()
        probdist = variable.probdist

        self.assertEqual(1, len(probdist.probabilities))
        self.assertEqual(0, len(probdist.conditioning_variable_nodes))
        self.assertEqual(0, len(probdist.cummulative_probabilities))

        probdist.finalize()

        self.assertEqual(1, len(probdist.probabilities))
        self.assertEqual(0, len(probdist.conditioning_variable_nodes))

        cprobs_compare = zip([0.2, 0.3, 0.8, 1.0], probdist.cummulative_probabilities['<unconditioned>'])
        for pair in cprobs_compare:
            self.assertAlmostEqual(pair[0], pair[1])


    def test_sampling_single(self):
        survey_bif = Path('testfiles', 'bif_files', 'survey.bif')
        bn = util.read_bif_file(survey_bif)

        bn.finalize()
        bn.finalized = False
        with self.assertRaises(BayesianNetworkNotFinalizedError):
            sample = bn.sample()

        bn.finalized = True

        self.assertListEqual(['AGE', 'SEX', 'EDU', 'OCC', 'R', 'TRN'], bn.variable_node_names__sampling_order)

        sample = bn.sample()
        self.assertValidExpectedSample(sample)


    @unittest.skipIf(TestBase.tag_excluded('sampling'), 'Sampling tests excluded')
    def test_sampling_multiple(self):
        survey_bif = Path('testfiles', 'bif_files', 'survey.bif')
        bn = util.read_bif_file(survey_bif)

        bn.finalize()
        self.assertListEqual(['AGE', 'SEX', 'EDU', 'OCC', 'R', 'TRN'], bn.variable_node_names__sampling_order)

        random.seed(42)
        samples = bn.sample_matrix(100000)
        self.assertIsInstance(samples, numpy.ndarray)
        self.assertEqual((100000, 6), samples.shape)

        AGEpd = self.calculate_probdist(samples[:, 0])
        EDUpd = self.calculate_probdist(samples[:, 1])
        OCCpd = self.calculate_probdist(samples[:, 2])
        Rpd   = self.calculate_probdist(samples[:, 3])
        SEXpd = self.calculate_probdist(samples[:, 4])
        TRNpd = self.calculate_probdist(samples[:, 5])

        delta = 0.003

        self.assertAlmostEqual(0.3, AGEpd[0], delta=delta)   # Pr(AGE = young) = 0.3
        self.assertAlmostEqual(0.5, AGEpd[1], delta=delta)   # Pr(AGE = adult) = 0.5
        self.assertAlmostEqual(0.2, AGEpd[2], delta=delta)   # Pr(AGE = old)   = 0.2

        self.assertAlmostEqual(0.49, SEXpd[0], delta=delta)   # Pr(SEX = M) = 0.49
        self.assertAlmostEqual(0.51, SEXpd[1], delta=delta)   # Pr(SEX = F) = 0.51

        EDU_p_highschool = (
                0.3 * 0.49 * 0.75
              + 0.5 * 0.49 * 0.72
              + 0.2 * 0.49 * 0.88
              + 0.3 * 0.51 * 0.64
              + 0.5 * 0.51 * 0.7
              + 0.2 * 0.51 * 0.9 )
        self.assertAlmostEqual(EDU_p_highschool, EDUpd[0], delta=delta)

        EDU_p_uni = (
                0.3 * 0.49 * 0.25
              + 0.5 * 0.49 * 0.28
              + 0.2 * 0.49 * 0.12
              + 0.3 * 0.51 * 0.36
              + 0.5 * 0.51 * 0.3
              + 0.2 * 0.51 * 0.1 )
        self.assertAlmostEqual(EDU_p_uni, EDUpd[1], delta=delta)

        OCC_p_emp = (
                EDU_p_highschool * 0.96
              + EDU_p_uni        * 0.92)
        self.assertAlmostEqual(OCC_p_emp, OCCpd[0], delta=delta)

        OCC_p_self = (
                EDU_p_highschool * 0.04
              + EDU_p_uni        * 0.08)
        self.assertAlmostEqual(OCC_p_self, OCCpd[1], delta=delta)

        R_p_small = (
                EDU_p_highschool * 0.25
              + EDU_p_uni        * 0.2 )
        self.assertAlmostEqual(R_p_small, Rpd[0], delta=delta)

        R_p_big = (
                EDU_p_highschool * 0.75
              + EDU_p_uni        * 0.8 )
        self.assertAlmostEqual(R_p_big, Rpd[1], delta=delta)

        TRN_p_car = (
                OCC_p_emp   * R_p_small * 0.48
              + OCC_p_self  * R_p_small * 0.56
              + OCC_p_emp   * R_p_big   * 0.58
              + OCC_p_self  * R_p_big   * 0.70)
        self.assertAlmostEqual(TRN_p_car, TRNpd[0], delta=delta)

        TRN_p_train = (
                OCC_p_emp   * R_p_small * 0.42
              + OCC_p_self  * R_p_small * 0.36
              + OCC_p_emp   * R_p_big   * 0.24
              + OCC_p_self  * R_p_big   * 0.21)
        self.assertAlmostEqual(TRN_p_train, TRNpd[1], delta=delta)

        TRN_p_other = (
                OCC_p_emp   * R_p_small * 0.10
              + OCC_p_self  * R_p_small * 0.08
              + OCC_p_emp   * R_p_big   * 0.18
              + OCC_p_self  * R_p_big   * 0.09)

        self.assertAlmostEqual(TRN_p_other, TRNpd[2], delta=delta)


    def test_variable_IDs(self):
        survey_bif = Path('testfiles', 'bif_files', 'survey.bif')
        bn = util.read_bif_file(survey_bif)
        bn.finalize()
        self.assertEqual(0, bn.variable_nodes['AGE'].ID)
        self.assertEqual(1, bn.variable_nodes['EDU'].ID)
        self.assertEqual(2, bn.variable_nodes['OCC'].ID)
        self.assertEqual(3, bn.variable_nodes['R'].ID)
        self.assertEqual(4, bn.variable_nodes['SEX'].ID)
        self.assertEqual(5, bn.variable_nodes['TRN'].ID)

        lungcancer_bif = Path('testfiles', 'bif_files', 'lungcancer.bif')
        bn = util.read_bif_file(lungcancer_bif)
        bn.finalize()
        self.assertEqual(0, bn.variable_nodes['ASIA'].ID)
        self.assertEqual(1, bn.variable_nodes['BRONC'].ID)
        self.assertEqual(2, bn.variable_nodes['DYSP'].ID)
        self.assertEqual(3, bn.variable_nodes['EITHER'].ID)
        self.assertEqual(4, bn.variable_nodes['LUNG'].ID)
        self.assertEqual(5, bn.variable_nodes['SMOKE'].ID)
        self.assertEqual(6, bn.variable_nodes['TUB'].ID)
        self.assertEqual(7, bn.variable_nodes['XRAY'].ID)

        alarm_bif = Path('testfiles', 'bif_files', 'alarm.bif')
        bn = util.read_bif_file(alarm_bif)
        bn.finalize()
        self.assertEqual(0,  bn.variable_nodes['ANAPHYLAXIS'].ID)
        self.assertEqual(1,  bn.variable_nodes['ARTCO2'].ID)
        self.assertEqual(2,  bn.variable_nodes['BP'].ID)
        self.assertEqual(3,  bn.variable_nodes['CATECHOL'].ID)
        self.assertEqual(4,  bn.variable_nodes['CO'].ID)
        self.assertEqual(5,  bn.variable_nodes['CVP'].ID)
        self.assertEqual(6,  bn.variable_nodes['DISCONNECT'].ID)
        self.assertEqual(7,  bn.variable_nodes['ERRCAUTER'].ID)
        self.assertEqual(8,  bn.variable_nodes['ERRLOWOUTPUT'].ID)
        self.assertEqual(9,  bn.variable_nodes['EXPCO2'].ID)
        self.assertEqual(10, bn.variable_nodes['FIO2'].ID)
        self.assertEqual(11, bn.variable_nodes['HISTORY'].ID)
        self.assertEqual(12, bn.variable_nodes['HR'].ID)
        self.assertEqual(13, bn.variable_nodes['HRBP'].ID)
        self.assertEqual(14, bn.variable_nodes['HREKG'].ID)
        self.assertEqual(15, bn.variable_nodes['HRSAT'].ID)
        self.assertEqual(16, bn.variable_nodes['HYPOVOLEMIA'].ID)
        self.assertEqual(17, bn.variable_nodes['INSUFFANESTH'].ID)
        self.assertEqual(18, bn.variable_nodes['INTUBATION'].ID)
        self.assertEqual(19, bn.variable_nodes['KINKEDTUBE'].ID)
        self.assertEqual(20, bn.variable_nodes['LVEDVOLUME'].ID)
        self.assertEqual(21, bn.variable_nodes['LVFAILURE'].ID)
        self.assertEqual(22, bn.variable_nodes['MINVOL'].ID)
        self.assertEqual(23, bn.variable_nodes['MINVOLSET'].ID)
        self.assertEqual(24, bn.variable_nodes['PAP'].ID)
        self.assertEqual(25, bn.variable_nodes['PCWP'].ID)
        self.assertEqual(26, bn.variable_nodes['PRESS'].ID)
        self.assertEqual(27, bn.variable_nodes['PULMEMBOLUS'].ID)
        self.assertEqual(28, bn.variable_nodes['PVSAT'].ID)
        self.assertEqual(29, bn.variable_nodes['SAO2'].ID)
        self.assertEqual(30, bn.variable_nodes['SHUNT'].ID)
        self.assertEqual(31, bn.variable_nodes['STROKEVOLUME'].ID)
        self.assertEqual(32, bn.variable_nodes['TPR'].ID)
        self.assertEqual(33, bn.variable_nodes['VENTALV'].ID)
        self.assertEqual(34, bn.variable_nodes['VENTLUNG'].ID)
        self.assertEqual(35, bn.variable_nodes['VENTMACH'].ID)
        self.assertEqual(36, bn.variable_nodes['VENTTUBE'].ID)


    def test_directed_graph_building(self):
        survey_bif = Path('testfiles', 'bif_files', 'survey.bif')
        bn = util.read_bif_file(survey_bif)
        bn.finalize()
        expected_directed_graph = {
                0: [1],
                1: [2, 3],
                2: [5],
                3: [5],
                4: [1],
                5: []
                }
        self.assertEqual(expected_directed_graph, bn.graph_d)

        lungcancer_bif = Path('testfiles', 'bif_files', 'lungcancer.bif')
        bn = util.read_bif_file(lungcancer_bif)
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
        self.assertDictEqual(expected_directed_graph, bn.graph_d)

        alarm_bif = Path('testfiles', 'bif_files', 'alarm.bif')
        bn = util.read_bif_file(alarm_bif)
        bn.finalize()
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
        self.assertDictEqual(expected_directed_graph, bn.graph_d)

        expected_paths = [
                [0, 32, 2],
                [0, 32, 3, 12, 4, 2]
                ]
        self.assertEqual(expected_paths, bn.find_all_directed_paths(0, 2))

        expected_paths = [
                [18, 30, 29, 3, 12, 4],
                [18, 33, 1, 3, 12, 4],
                [18, 33, 28, 29, 3, 12, 4],
                [18, 34, 33, 1, 3, 12, 4],
                [18, 34, 33, 28, 29, 3, 12, 4]
                ]
        self.assertEqual(expected_paths, bn.find_all_directed_paths(18, 4))

        expected_paths = []
        self.assertEqual(expected_paths, bn.find_all_directed_paths(18, 23))

        expected_paths = [[6, 36]]
        self.assertEqual(expected_paths, bn.find_all_directed_paths(6, 36))


    def test_undirected_graph_building(self):
        alarm_bif = Path('testfiles', 'bif_files', 'alarm.bif')
        bn = util.read_bif_file(alarm_bif)
        bn.finalize()
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
        self.assertDictEqual(expected_undirected_graph, bn.graph_u)


    def test_building_from_directed_graph(self):
        self.maxDiff = None
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
        self.assertEqual(graph, bn.graph_d)
        self.assertEqual(expected_undirected_graph, bn.graph_u)

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
        self.assertEqual(graph, bn.graph_d)
        self.assertEqual(expected_undirected_graph, bn.graph_u)


    def test_d_separation(self):
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

        self.assertTrue(bn.d_separated(2, [1], 3))
        self.assertFalse(bn.d_separated(2, [1, 5], 3))
        self.assertFalse(bn.d_separated(1, [], 2))
        self.assertFalse(bn.d_separated(1, [], 3))
        self.assertFalse(bn.d_separated(1, [], 4))
        self.assertFalse(bn.d_separated(1, [], 5))

        self.assertTrue(bn.d_separated(1, [4], 5))
        self.assertFalse(bn.d_separated(2, [], 3))
        self.assertFalse(bn.d_separated(2, [4], 3))

        self.assertFalse(bn.d_separated(5, [2], 1))
        self.assertFalse(bn.d_separated(5, [3], 1))
        self.assertTrue(bn.d_separated(5, [2, 3], 1))

        self.assertFalse(bn.d_separated(3, [1, 2], 5))

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

        self.assertFalse(bn.d_separated(3, [], 2))
        self.assertFalse(bn.d_separated(3, [5], 2))
        self.assertTrue(bn.d_separated(3, [1], 2))

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
        self.assertFalse(bn.d_separated(4, [1], 3))
        # Assert that [0, 1] is the Markov blanket of 4, by the Intersection
        # Property.
        self.assertTrue(bn.d_separated(4, [0, 1, 3], 2))
        self.assertTrue(bn.d_separated(4, [0, 1, 2], 3))
        # Assert that [0, 1] is the Markov blanket of 4, by the Contraction
        # Property.
        self.assertTrue(bn.d_separated(4, [0, 1, 3], 2))
        self.assertTrue(bn.d_separated(4, [0, 1], 3))

        # Moderately complex graph, as generated by the 'alarm' Bayesian
        # network, from
        # http://www.bnlearn.com/bnrepository/discrete-medium.html#alarm
        alarm_bif = Path('testfiles', 'bif_files', 'alarm.bif')
        bn = util.read_bif_file(alarm_bif)
        bn.finalize()

        self.assertTrue(bn.d_separated(0, [], 1))
        self.assertFalse(bn.d_separated(0, [12], 1))
        self.assertTrue(bn.d_separated(0, [32, 12], 1))

        self.assertTrue(bn.d_separated(24, [], 26))
        self.assertFalse(bn.d_separated(24, [29], 26))

        self.assertTrue(bn.d_separated(17, [], 18))
        self.assertFalse(bn.d_separated(17, [3], 18))
        self.assertFalse(bn.d_separated(17, [12], 18))

        self.assertTrue(bn.d_separated(10, [], 26))
        self.assertFalse(bn.d_separated(10, [29], 26))
        self.assertFalse(bn.d_separated(10, [29, 19], 26))
        self.assertFalse(bn.d_separated(10, [29, 19, 36], 26))
        self.assertTrue(bn.d_separated(10, [18, 29, 19, 36], 26))


    def calculate_probdist(self, column):
        counter = Counter()
        counter.update(column.T.tolist())
        probdist = {}
        for value in counter.keys():
            probdist[value] = 1.0 * counter[value] / column.size
        return probdist


    def default_variable__unconditioned(self):
        variable = VariableNode('ASDF')
        variable.values = ['rocket', 'carbohydrate', 'albatross', 'oxygen']
        variable.probdist = ProbabilityDistributionOfVariableNode(variable)
        variable.probdist.probabilities = {'<unconditioned>' : [0.2, 0.1, 0.5, 0.2]}
        return variable


    def assertValidExpectedSample(self, sample):
        self.assertIsInstance(sample, dict)
        self.assertEqual(6, len(sample))
        self.assertIn(sample['AGE'], ['young', 'adult', 'old'])
        self.assertIn(sample['SEX'], ['M', 'F'])
        self.assertIn(sample['EDU'], ['highschool', 'uni'])
        self.assertIn(sample['OCC'], ['emp', 'self'])
        self.assertIn(sample['R'],   ['small', 'big'])
        self.assertIn(sample['TRN'], ['car', 'train', 'other'])
