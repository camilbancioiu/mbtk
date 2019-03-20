import random
from pathlib import Path
from collections import Counter

import numpy
import unittest

from mbff_tests.TestBase import TestBase

import mbff.utilities.functions as util
from mbff.dataset.BayesianNetwork import *

class TestBayesianNetwork(TestBase):

    def test_probability_distribution(self):
        variable = self.default_variable__unconditioned()
        probdist = variable.probdist

        self.assertEqual(1, len(probdist.probabilities))
        self.assertEqual(0, len(probdist.conditioning_variables))
        self.assertEqual(0, len(probdist.cummulative_probabilities))

        probdist.finalize()

        self.assertEqual(1, len(probdist.probabilities))
        self.assertEqual(0, len(probdist.conditioning_variables))

        cprobs_compare = zip([0.2, 0.3, 0.8, 1.0], probdist.cummulative_probabilities['<unconditioned>'])
        for pair in cprobs_compare:
            self.assertAlmostEqual(pair[0], pair[1])


    def test_sample_roll(self):
        variable = self.default_variable__unconditioned()
        # TODO


    def test_sampling_single(self):
        survey_bif = Path('testfiles', 'bif_files', 'survey.bif')
        bn = util.read_bif_file(survey_bif)

        bn.finalize()

        sample = bn.sample()
        self.assertValidExpectedSample(sample)


    @unittest.skip
    def test_sampling_multiple(self):
        survey_bif = Path('testfiles', 'bif_files', 'survey.bif')
        bn = util.read_bif_file(survey_bif)

        bn.finalize()

        random.seed(42)
        samples = bn.sample_matrix(1000)
        self.assertIsInstance(samples, numpy.ndarray)
        self.assertEqual((1000, 6), samples.shape)

        AGEpd = self.calculate_probdist(samples[:, 0])
        EDUpd = self.calculate_probdist(samples[:, 1])
        OCCpd = self.calculate_probdist(samples[:, 2])
        Rpd   = self.calculate_probdist(samples[:, 3])
        SEXpd = self.calculate_probdist(samples[:, 4])
        TRNpd = self.calculate_probdist(samples[:, 5])

        print(AGEpd)
        self.assertAlmostEqual(0.3, AGEpd[0])   # Pr(AGE = young) = 0.3
        self.assertAlmostEqual(0.5, AGEpd[1])   # Pr(AGE = adult) = 0.5
        self.assertAlmostEqual(0.2, AGEpd[2])   # Pr(AGE = old)   = 0.2

        self.assertAlmostEqual(0.49, SEXpd[0])   # Pr(SEX = M) = 0.49
        self.assertAlmostEqual(0.51, SEXpd[1])   # Pr(SEX = F) = 0.51

        # TODO EDU, OCC, R, TRN


    def calculate_probdist(self, column):
        counter = Counter()
        counter.update(column.T.tolist())
        probdist = {}
        for value in counter.keys():
            probdist[value] = 1.0 * counter[value] / column.size
        return probdist


    def default_variable__unconditioned(self):
        variable = Variable('ASDF')
        variable.values = ['rocket', 'carbohydrate', 'albatross', 'oxygen']
        variable.probdist = ProbabilityDistribution(variable)
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


