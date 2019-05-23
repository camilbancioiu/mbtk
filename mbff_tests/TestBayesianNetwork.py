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


