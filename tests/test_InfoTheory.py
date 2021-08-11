import numpy

import mbtk.math.infotheory as infotheory
from mbtk.math.Variable import Variable


def almostEqual(x: float, y: float) -> bool:
    return abs(x - y) < 1e-10



def test_MI__binary() -> None:
    # Test calculating the mutual information of two binary variables.
    expected_MI: float

    # Both variables lack any information at all (H = 0).
    # Only one pair in the joint pmf: (0,0)
    X = Variable(numpy.array([0, 0, 0, 0, 0, 0, 0, 0]))
    Y = Variable(numpy.array([0, 0, 0, 0, 0, 0, 0, 0]))
    expected_MI = 0
    calculated_MI = infotheory.mutual_information(*infotheory.calculate_pmf_for_mi(X, Y))
    assert expected_MI == calculated_MI

    # X has some entropy, but Y contains no information of any kind.
    # Two pairs in the joint pmf: (0,0) (1,0)
    X = Variable(numpy.array([1, 0, 0, 0, 0, 0, 0, 0]))
    Y = Variable(numpy.array([0, 0, 0, 0, 0, 0, 0, 0]))
    expected_MI = 0
    calculated_MI = infotheory.mutual_information(*infotheory.calculate_pmf_for_mi(X, Y))
    assert expected_MI == calculated_MI

    # Both variables have some entropy, and are identical.
    # Two pairs in the joint pmf: (0,0) (1,1)
    X = Variable(numpy.array([1, 0, 0, 0, 0, 0, 0, 0]))
    Y = Variable(numpy.array([1, 0, 0, 0, 0, 0, 0, 0]))
    expected_MI = 0.5435644431
    calculated_MI = infotheory.mutual_information(*infotheory.calculate_pmf_for_mi(X, Y))
    assert almostEqual(expected_MI, calculated_MI)

    # Both variables have some information, and are almost identical (a
    # single difference).
    # Three pairs in the joint pmf: (0,0) (1,0) (1,1)
    X = Variable(numpy.array([1, 0, 0, 0, 1, 0, 0, 0]))
    Y = Variable(numpy.array([1, 0, 0, 0, 0, 0, 0, 0]))
    expected_MI = 0.2935644431
    calculated_MI = infotheory.mutual_information(*infotheory.calculate_pmf_for_mi(X, Y))
    assert almostEqual(expected_MI, calculated_MI)

    # Both variables contain information, but the information in one
    # cannot be used to predict the other.
    # All four pairs in the joint pmf: (0,0) (0,1) (1,0) (1,1)
    X = Variable(numpy.array([1, 0, 0, 0, 1, 1, 0, 0]))
    Y = Variable(numpy.array([1, 0, 1, 1, 0, 0, 0, 0]))
    expected_MI = 0.0032289436
    calculated_MI = infotheory.mutual_information(*infotheory.calculate_pmf_for_mi(X, Y))
    assert almostEqual(expected_MI, calculated_MI)

    # Both variables contain information, with some (not much) mutual
    # information as well.
    # Three pairs in the joint pmf: (0,0) (0,1) (1,1)
    X = Variable(numpy.array([1, 0, 0, 0, 1, 1, 0, 0]))
    Y = Variable(numpy.array([1, 0, 1, 1, 1, 1, 0, 0]))
    expected_MI = 0.3475898813
    calculated_MI = infotheory.mutual_information(*infotheory.calculate_pmf_for_mi(X, Y))
    assert almostEqual(expected_MI, calculated_MI)

    # Both variables contain information, almost all of it is mutual
    # information.
    # Three pairs in the joint pmf: (0,0) (0,1) (1,0)
    X = Variable(numpy.array([0, 1, 0, 0, 1, 1, 0, 0]))
    Y = Variable(numpy.array([1, 0, 1, 1, 0, 0, 1, 0]))
    expected_MI = 0.5487949406
    calculated_MI = infotheory.mutual_information(*infotheory.calculate_pmf_for_mi(X, Y))
    assert almostEqual(expected_MI, calculated_MI)

    # Both variables contain information,  all of it is mutual information.
    # Two pairs in the joint pmf: (0,1) (1,0)
    X = Variable(numpy.array([0, 1, 0, 0, 1, 1, 0, 1]))
    Y = Variable(numpy.array([1, 0, 1, 1, 0, 0, 1, 0]))
    expected_MI = 1
    calculated_MI = infotheory.mutual_information(*infotheory.calculate_pmf_for_mi(X, Y))
    assert almostEqual(expected_MI, calculated_MI)
