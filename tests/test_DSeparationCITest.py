import pytest
import time
import collections
import operator

from pprint import pprint

from mbtk.math.DSeparationCITest import DSeparationCITest
from mbtk.math.DSeparationCITest import DSeparationAsCorrelationHeuristic
from mbtk.algorithms.mb.iamb import AlgorithmIAMB
from mbtk.algorithms.mb.pcmb import AlgorithmPCMB
from mbtk.algorithms.mb.ipcmb import AlgorithmIPCMB


@pytest.mark.slow
def test_dsep_full_alarm_ipcmb(bn_alarm):
    bn = bn_alarm
    variables = sorted(list(bn.graph_d.keys()))
    print()

    expected_boundaries = expected_markov_boundaries_alarm()

    total_start_time = time.time()
    for target in variables:
        target_start_time = time.time()
        parameters = make_parameters_ipcmb(target, bn, variables)
        mb = AlgorithmIPCMB(None, parameters).discover_mb()
        target_end_time = time.time()
        duration = target_end_time - target_start_time
        print(f'[{duration:6.2f} s] Variable {target:3}, Markov boundary {mb}')
        assert mb == expected_boundaries[target]

    total_end_time = time.time()
    duration = total_end_time - total_start_time
    print(f'Total duration {duration}s')


def test_dsep_alarm_ipcmb(bn_alarm):
    bn = bn_alarm
    variables = sorted(list(bn.graph_d.keys()))
    print()

    expected_boundaries = expected_markov_boundaries_alarm()

    target = 17
    target_start_time = time.time()
    parameters = make_parameters_ipcmb(target, bn, variables)
    ipcmb = AlgorithmIPCMB(None, parameters)
    mb = ipcmb.discover_mb()
    target_end_time = time.time()
    duration = target_end_time - target_start_time
    print(f'[{duration:6.2f} s] Variable {target:3}, Markov boundary {mb}')
    assert mb == expected_boundaries[target]

    citr = ipcmb.CITest.ci_test_results
    print('ci tests', len(citr))

    condset = operator.attrgetter('Z')
    cond_counter = collections.Counter(map(len, map(condset, citr)))
    pprint(cond_counter)


def make_parameters_ipcmb(target, bn, variables):
    return {
        'target': target,
        'all_variables': variables,
        'ci_test_class': DSeparationCITest,
        'source_bayesian_network': bn,
        'pc_only': False,
        'ci_test_debug': 0,
        'algorithm_debug': 0
    }


@pytest.mark.slow
def test_dsep_full_alarm_pcmb(bn_alarm):
    bn = bn_alarm
    variables = sorted(list(bn.graph_d.keys()))
    print()

    expected_boundaries = expected_markov_boundaries_alarm()

    total_start_time = time.time()
    for target in [0]:
        target_start_time = time.time()
        parameters = make_parameters_pcmb(target, bn, variables)
        mb = AlgorithmPCMB(None, parameters).discover_mb()
        target_end_time = time.time()
        duration = target_end_time - target_start_time
        print(f'[{duration:6.2f} s] Variable {target:3}, Markov boundary {mb}')
        assert mb == expected_boundaries[target]

    total_end_time = time.time()
    duration = total_end_time - total_start_time
    print(f'Total duration {duration}s')


def make_parameters_pcmb(target, bn, variables):
    return {
        'target': target,
        'all_variables': variables,
        'ci_test_class': DSeparationCITest,
        'source_bayesian_network': bn,
        'pc_only': False,
        'ci_test_debug': 0,
        'algorithm_debug': 0
    }


@pytest.mark.slow
def test_dsep_full_alarm_iamb(bn_alarm):
    bn = bn_alarm
    variables = sorted(list(bn.graph_d.keys()))
    print()

    expected_boundaries = expected_markov_boundaries_alarm()

    total_start_time = time.time()
    for target in variables:
        target_start_time = time.time()
        parameters = make_parameters_iamb(target, bn, variables)
        mb = AlgorithmIAMB(None, parameters).discover_mb()
        target_end_time = time.time()
        duration = target_end_time - target_start_time
        print(f'[{duration:6.2f} s] Variable {target:3}, Markov boundary {mb}')
        assert mb == expected_boundaries[target]

    total_end_time = time.time()
    duration = total_end_time - total_start_time
    print(f'Total duration {duration}s')


def make_parameters_iamb(target, bn, variables):
    return {
        'target': target,
        'all_variables': variables,
        'ci_test_class': DSeparationCITest,
        'correlation_heuristic_class': DSeparationAsCorrelationHeuristic,
        'source_bayesian_network': bn,
        'pc_only': False,
        'ci_test_debug': 0,
        'algorithm_debug': 0
    }


def expected_markov_boundaries_alarm():
    return {
        0: [32],
        1: [3, 9, 17, 29, 32, 33, 34],
        2: [4, 32],
        3: [1, 12, 17, 29, 32],
        4: [2, 12, 31, 32],
        5: [20],
        6: [35, 36],
        7: [12, 14, 15],
        8: [12, 13],
        9: [1, 34],
        10: [28, 33],
        11: [21],
        12: [3, 4, 7, 8, 13, 14, 15, 31],
        13: [8, 12],
        14: [7, 12],
        15: [7, 12],
        16: [20, 21, 31],
        17: [1, 3, 29, 32],
        18: [19, 22, 26, 27, 30, 33, 34, 36],
        19: [18, 26, 34, 36],
        20: [5, 16, 21, 25],
        21: [11, 16, 20, 31],
        22: [18, 34],
        23: [35],
        24: [27],
        25: [20],
        26: [18, 19, 36],
        27: [18, 24, 30],
        28: [10, 29, 30, 33],
        29: [1, 3, 17, 28, 30, 32],
        30: [18, 27, 28, 29],
        31: [4, 12, 16, 21],
        32: [0, 1, 2, 3, 4, 17, 29],
        33: [1, 10, 18, 28, 34],
        34: [1, 9, 18, 19, 22, 33, 36],
        35: [6, 23, 36],
        36: [6, 18, 19, 26, 34, 35]
    }
