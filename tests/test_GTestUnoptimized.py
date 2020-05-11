from mbff.math.Variable import JointVariables
import mbff.math.Variable
import mbff.math.G_test__unoptimized
import mbff.math.DoFCalculators

import pytest


@pytest.mark.slow
def test_G_value__alarm(ds_alarm_8e3):
    Omega = ds_alarm_8e3.omega
    dataset = ds_alarm_8e3.datasetmatrix
    bn = ds_alarm_8e3.bayesiannetwork

    parameters = dict()
    parameters['ci_test_significance'] = 0.95
    parameters['ci_test_debug'] = 0
    parameters['omega'] = Omega
    parameters['source_bayesian_network'] = bn
    parameters['ci_test_dof_calculator_class'] = mbff.math.DoFCalculators.UnadjustedDoF

    X = 35
    Y = 3
    Z = {32, 1, 34, 36}

    G_test = mbff.math.G_test__unoptimized.G_test(dataset, parameters)
    G_test.conditionally_independent(X, Y, Z)



@pytest.mark.slow
def test_G_value__lungcancer(ds_lungcancer_4e4):
    Omega = ds_lungcancer_4e4.omega
    lungcancer = ds_lungcancer_4e4.datasetmatrix
    bn = ds_lungcancer_4e4.bayesiannetwork

    ASIA = lungcancer.get_variable('X', 0)
    BRONC = lungcancer.get_variable('X', 1)
    DYSP = lungcancer.get_variable('X', 2)
    EITHER = lungcancer.get_variable('X', 3)
    LUNG = lungcancer.get_variable('X', 4)
    SMOKE = lungcancer.get_variable('X', 5)
    TUB = lungcancer.get_variable('X', 6)
    XRAY = lungcancer.get_variable('X', 7)

    parameters = dict()
    parameters['ci_test_significance'] = 0.95
    parameters['ci_test_debug'] = 0
    parameters['omega'] = Omega
    parameters['source_bayesian_network'] = bn
    parameters['ci_test_dof_calculator_class'] = mbff.math.DoFCalculators.StructuralDoF

    G_test = mbff.math.G_test__unoptimized.G_test(lungcancer, parameters)

    assertCITestAccurate(G_test, ASIA, SMOKE, Omega)
    assertCITestAccurate(G_test, ASIA, LUNG, Omega)
    assertCITestAccurate(G_test, ASIA, BRONC, Omega)
    assertCITestAccurate(G_test, ASIA, TUB, Omega)
    assertCITestAccurate(G_test, ASIA, EITHER, Omega)
    assertCITestAccurate(G_test, ASIA, XRAY, Omega)
    assertCITestAccurate(G_test, EITHER, ASIA, JointVariables(TUB, LUNG))
    assertCITestAccurate(G_test, EITHER, SMOKE, JointVariables(TUB, LUNG))
    assertCITestAccurate(G_test, DYSP, SMOKE, JointVariables(EITHER, BRONC))
    assertCITestAccurate(G_test, DYSP, LUNG, JointVariables(EITHER, BRONC))
    assertCITestAccurate(G_test, DYSP, TUB, JointVariables(EITHER, BRONC))
    assertCITestAccurate(G_test, XRAY, TUB, EITHER)
    assertCITestAccurate(G_test, XRAY, LUNG, EITHER)
    assertCITestAccurate(G_test, XRAY, ASIA, EITHER)
    assertCITestAccurate(G_test, XRAY, SMOKE, EITHER)
    assertCITestAccurate(G_test, XRAY, DYSP, EITHER)
    assertCITestAccurate(G_test, XRAY, BRONC, EITHER)
    assertCITestAccurate(G_test, XRAY, EITHER, Omega)
    assertCITestAccurate(G_test, XRAY, LUNG, Omega)
    assertCITestAccurate(G_test, XRAY, SMOKE, Omega)
    assertCITestAccurate(G_test, XRAY, TUB, Omega)
    # assertCITestAccurate(G_test, ASIA, DYSP, Omega)



def assertCITestAccurate(G_test, X, Y, Z):
    if isinstance(Z, mbff.math.Variable.Omega):
        Z_ID = []
    elif isinstance(Z, JointVariables):
        Z_ID = Z.variableIDs
    elif isinstance(Z, mbff.math.Variable.Variable):
        Z_ID = [Z.ID]

    result = G_test.G_test_conditionally_independent(X.ID, Y.ID, Z_ID)
    if G_test.source_bn is not None:
        result.computed_d_separation = G_test.source_bn.d_separated(X.ID, Z_ID, Y.ID)
    result.index = len(G_test.ci_test_results)
    G_test.ci_test_results.append(result)
    # print(result, result.test_distribution_parameters)
    assert result.accurate() is True
