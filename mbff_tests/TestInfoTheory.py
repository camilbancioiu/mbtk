import numpy

import mbff.math.infotheory as infotheory

from mbff_tests.TestBase import TestBase


class TestInfoTheory(TestBase):

    def test_MI__binary(self):
        # Test calculating the mutual information of two binary variables.

        # Both variables lack any information at all (H = 0).
        # Only one pair in the joint pmf: (0,0)
        X = numpy.array([0, 0, 0, 0,  0, 0, 0, 0])
        Y = numpy.array([0, 0, 0, 0,  0, 0, 0, 0])
        expected_MI = 0
        calculated_MI = infotheory.MI__binary(X, Y)
        self.assertEqual(expected_MI, calculated_MI)

        # X has some entropy, but Y contains no information of any kind.
        # Two pairs in the joint pmf: (0,0) (1,0)
        X = numpy.array([1, 0, 0, 0,  0, 0, 0, 0])
        Y = numpy.array([0, 0, 0, 0,  0, 0, 0, 0])
        expected_MI = 0
        calculated_MI = infotheory.MI__binary(X, Y)
        self.assertEqual(expected_MI, calculated_MI)

        # Both variables have some entropy, and are identical.
        # Two pairs in the joint pmf: (0,0) (1,1)
        X = numpy.array([1, 0, 0, 0,  0, 0, 0, 0])
        Y = numpy.array([1, 0, 0, 0,  0, 0, 0, 0])
        expected_MI = 0.5435644431
        calculated_MI = infotheory.MI__binary(X, Y)
        self.assertAlmostEqual(expected_MI, calculated_MI, delta=1e-10)

        # Both variables have some information, and are almost identical (a
        # single difference).
        # Three pairs in the joint pmf: (0,0) (1,0) (1,1)
        X = numpy.array([1, 0, 0, 0,  1, 0, 0, 0])
        Y = numpy.array([1, 0, 0, 0,  0, 0, 0, 0])
        expected_MI = 0.2935644431
        calculated_MI = infotheory.MI__binary(X, Y)
        self.assertAlmostEqual(expected_MI, calculated_MI, delta=1e-10)

        # Both variables contain information, but the information in one
        # cannot be used to predict the other.
        # All four pairs in the joint pmf: (0,0) (0,1) (1,0) (1,1)
        X = numpy.array([1, 0, 0, 0,  1, 1, 0, 0])
        Y = numpy.array([1, 0, 1, 1,  0, 0, 0, 0])
        expected_MI = 0.0032289436
        calculated_MI = infotheory.MI__binary(X, Y)
        self.assertAlmostEqual(expected_MI, calculated_MI, delta=1e-10)

        # Both variables contain information, with some (not much) mutual
        # information as well.
        # Three pairs in the joint pmf: (0,0) (0,1) (1,1)
        X = numpy.array([1, 0, 0, 0,  1, 1, 0, 0])
        Y = numpy.array([1, 0, 1, 1,  1, 1, 0, 0])
        expected_MI = 0.3475898813
        calculated_MI = infotheory.MI__binary(X, Y)
        self.assertAlmostEqual(expected_MI, calculated_MI, delta=1e-10)

        # Both variables contain information, almost all of it is mutual
        # information.
        # Three pairs in the joint pmf: (0,0) (0,1) (1,0)
        X = numpy.array([0, 1, 0, 0,  1, 1, 0, 0])
        Y = numpy.array([1, 0, 1, 1,  0, 0, 1, 0])
        expected_MI = 0.5487949406
        calculated_MI = infotheory.MI__binary(X, Y)
        self.assertAlmostEqual(expected_MI, calculated_MI, delta=1e-10)

        # Both variables contain information,  all of it is mutual information.
        # Two pairs in the joint pmf: (0,1) (1,0)
        X = numpy.array([0, 1, 0, 0,  1, 1, 0, 1])
        Y = numpy.array([1, 0, 1, 1,  0, 0, 1, 0])
        expected_MI = 1
        calculated_MI = infotheory.MI__binary(X, Y)
        self.assertAlmostEqual(expected_MI, calculated_MI, delta=1e-10)


    def test_pmf__binary(self):
        # Test calculating the probability mass function of a single binary
        # variable.
        X = numpy.array([0, 0, 0, 0,  0, 0, 0, 0])
        expected_pmf = {0: 1, 1: 0}
        calculated_pmf = infotheory.calculate_pmf__binary(X)
        self.assertDictEqual(expected_pmf, calculated_pmf)

        X = numpy.array([0, 0, 0, 1,  0, 0, 0, 0])
        expected_pmf = {0: 7/8, 1: 1/8}
        calculated_pmf = infotheory.calculate_pmf__binary(X)
        self.assertDictEqual(expected_pmf, calculated_pmf)

        X = numpy.array([1, 1, 1, 1,  0, 0, 0, 0])
        expected_pmf = {0: 4/8, 1: 4/8}
        calculated_pmf = infotheory.calculate_pmf__binary(X)
        self.assertDictEqual(expected_pmf, calculated_pmf)

        X = numpy.array([1, 1, 1, 1,  1, 1, 1, 1])
        expected_pmf = {0: 0, 1: 1}
        calculated_pmf = infotheory.calculate_pmf__binary(X)
        self.assertDictEqual(expected_pmf, calculated_pmf)

        # Test calculating the joint probability mass function of two binary
        # variables.
        X = numpy.array([0, 0, 0, 0,  0, 0, 0, 0])
        Y = numpy.array([0, 0, 0, 0,  0, 0, 0, 0])
        expected_pmf = {
                (0, 0): 1,
                (0, 1): 0,
                (1, 0): 0,
                (1, 1): 0
                }
        calculated_pmf = infotheory.calculate_joint_pmf2__binary(X, Y)
        self.assertDictEqual(expected_pmf, calculated_pmf)

        X = numpy.array([1, 0, 0, 0,  0, 0, 0, 0])
        Y = numpy.array([0, 0, 0, 0,  0, 0, 0, 0])
        expected_pmf = {
                (0, 0): 7/8,
                (0, 1): 0,
                (1, 0): 1/8,
                (1, 1): 0
                }
        calculated_pmf = infotheory.calculate_joint_pmf2__binary(X, Y)
        self.assertDictEqual(expected_pmf, calculated_pmf)

        X = numpy.array([1, 0, 0, 0,  0, 0, 0, 0])
        Y = numpy.array([0, 0, 0, 1,  0, 0, 0, 0])
        expected_pmf = {
                (0, 0): 6/8,
                (0, 1): 1/8,
                (1, 0): 1/8,
                (1, 1): 0
                }
        calculated_pmf = infotheory.calculate_joint_pmf2__binary(X, Y)
        self.assertDictEqual(expected_pmf, calculated_pmf)

        X = numpy.array([1, 1, 1, 0,  0, 0, 0, 1])
        Y = numpy.array([0, 0, 1, 1,  0, 0, 0, 1])
        expected_pmf = {
                (0, 0): 3/8,
                (0, 1): 1/8,
                (1, 0): 2/8,
                (1, 1): 2/8
                }
        calculated_pmf = infotheory.calculate_joint_pmf2__binary(X, Y)
        self.assertDictEqual(expected_pmf, calculated_pmf)

        X = numpy.array([1, 1, 1, 1,  1, 1, 1, 1])
        Y = numpy.array([0, 0, 1, 1,  0, 0, 0, 1])
        expected_pmf = {
                (0, 0): 0/8,
                (0, 1): 0/8,
                (1, 0): 5/8,
                (1, 1): 3/8
                }
        calculated_pmf = infotheory.calculate_joint_pmf2__binary(X, Y)
        self.assertDictEqual(expected_pmf, calculated_pmf)

        X = numpy.array([0, 0, 1, 1,  0, 0, 0, 1])
        Y = numpy.array([1, 1, 1, 1,  1, 1, 1, 1])
        expected_pmf = {
                (0, 0): 0/8,
                (0, 1): 5/8,
                (1, 0): 0/8,
                (1, 1): 3/8
                }
        calculated_pmf = infotheory.calculate_joint_pmf2__binary(X, Y)
        self.assertDictEqual(expected_pmf, calculated_pmf)

        X = numpy.array([1, 1, 1, 1,  1, 1, 0, 1])
        Y = numpy.array([1, 1, 1, 1,  1, 1, 1, 1])
        expected_pmf = {
                (0, 0): 0/8,
                (0, 1): 1/8,
                (1, 0): 0/8,
                (1, 1): 7/8
                }
        calculated_pmf = infotheory.calculate_joint_pmf2__binary(X, Y)
        self.assertDictEqual(expected_pmf, calculated_pmf)

        X = numpy.array([1, 1, 1, 1,  1, 1, 1, 1])
        Y = numpy.array([1, 1, 1, 1,  1, 1, 1, 1])
        expected_pmf = {
                (0, 0): 0/8,
                (0, 1): 0/8,
                (1, 0): 0/8,
                (1, 1): 8/8
                }
        calculated_pmf = infotheory.calculate_joint_pmf2__binary(X, Y)
        self.assertDictEqual(expected_pmf, calculated_pmf)
