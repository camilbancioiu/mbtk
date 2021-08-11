
from mbtk.math.CMICalculator import CMICalculator


def test_basic_functionality(bn_alarm):
    bn = bn_alarm

    parameters = dict()
    parameters['source_bayesian_network'] = bn

    X = bn.variable_nodes_index('INTUBATION')
    Y = bn.variable_nodes_index('MINVOL')

    cmi = CMICalculator(None, parameters)
    value = cmi.compute(X, Y, [])

    assert value > 0
