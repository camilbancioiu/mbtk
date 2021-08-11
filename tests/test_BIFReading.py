from collections import OrderedDict
from pathlib import Path

import tests.utilities as testutil
from mbtk.structures.BayesianNetwork import BayesianNetwork, VariableNode, ProbabilityDistributionOfVariableNode


def test_reading_bif_file():
    survey_bif = Path(testutil.bif_folder, 'survey.bif')

    bn_expected = default_Bayesian_network()
    bn_read = BayesianNetwork.from_bif_file(survey_bif, use_cache=False)

    assertBayesianNetworkEqual(bn_expected, bn_read)


def assertBayesianNetworkEqual(bnA, bnB):
    assert bnA.name == bnB.name
    assert bnA.properties == bnB.properties
    varnamesA = bnA.variable_node_names()
    varnamesB = bnB.variable_node_names()
    assert varnamesA == varnamesB
    for varname in varnamesA:
        varA = bnA.variable_nodes[varname]
        varB = bnB.variable_nodes[varname]
        assertVariableNodesEqual(varA, varB)



def assertVariableNodesEqual(A, B):
    assert A.name == B.name
    assert A.values == B.values
    assert A.properties == B.properties
    assertProbabilityDistributionsOfVariableNodesEqual(A.probdist, B.probdist)



def assertProbabilityDistributionsOfVariableNodesEqual(pA, pB):
    assert len(pA.conditioning_variable_nodes) == len(pB.conditioning_variable_nodes)
    for ourVar, otherVar in zip(pA.conditioning_variable_nodes.values(), pB.conditioning_variable_nodes.values()):
        assert ourVar.values == otherVar.values
    assert pA.probabilities == pB.probabilities
    assert pA.properties == pB.properties



def default_Bayesian_network():
    AGE = VariableNode('AGE')
    AGE.values = ['young', 'adult', 'old']
    AGE.properties = {'label': 'age'}
    AGE.probdist = ProbabilityDistributionOfVariableNode(AGE)
    AGE.probdist.conditioning_variable_nodes = OrderedDict()
    AGE.probdist.probabilities = {'<unconditioned>': [0.3, 0.5, 0.2]}

    SEX = VariableNode('SEX')
    SEX.values = ['M', 'F']
    SEX.properties = {'label': 'sex'}
    SEX.probdist = ProbabilityDistributionOfVariableNode(SEX)
    SEX.probdist.conditioning_variable_nodes = OrderedDict()
    SEX.probdist.probabilities = {'<unconditioned>': [0.49, 0.51]}

    EDU = VariableNode('EDU')
    EDU.values = ['highschool', 'uni']
    EDU.properties = {'label': 'education'}
    EDU.probdist = ProbabilityDistributionOfVariableNode(AGE)
    EDU.probdist.conditioning_variable_nodes = OrderedDict([('AGE', AGE), ('SEX', SEX)])
    EDU.probdist.probabilities = {
        ('young', 'M'): [0.75, 0.25],
        ('young', 'F'): [0.64, 0.36],
        ('adult', 'M'): [0.72, 0.28],
        ('adult', 'F'): [0.70, 0.30],
        ('old', 'M'): [0.88, 0.12],
        ('old', 'F'): [0.90, 0.10]
    }

    OCC = VariableNode('OCC')
    OCC.values = ['emp', 'self']
    OCC.properties = {'label': 'occupation'}
    OCC.probdist = ProbabilityDistributionOfVariableNode(OCC)
    OCC.probdist.conditioning_variable_nodes = OrderedDict([('EDU', EDU)])
    OCC.probdist.probabilities = {
        ('highschool',): [0.96, 0.04],
        ('uni',): [0.92, 0.08]
    }

    R = VariableNode('R')
    R.values = ['small', 'big']
    R.properties = {'label': 'unknown'}
    R.probdist = ProbabilityDistributionOfVariableNode(R)
    R.probdist.conditioning_variable_nodes = OrderedDict([('EDU', EDU)])
    R.probdist.probabilities = {
        ('highschool',): [0.25, 0.75],
        ('uni',): [0.2, 0.8]
    }

    TRN = VariableNode('TRN')
    TRN.values = ['car', 'train', 'other']
    TRN.properties = {'label': 'transportation'}
    TRN.probdist = ProbabilityDistributionOfVariableNode(TRN)
    TRN.probdist.conditioning_variable_nodes = OrderedDict([('OCC', OCC), ('R', R)])
    TRN.probdist.probabilities = {
        ('emp', 'small'): [0.48, 0.42, 0.10],
        ('self', 'small'): [0.56, 0.36, 0.08],
        ('emp', 'big'): [0.58, 0.24, 0.18],
        ('self', 'big'): [0.70, 0.21, 0.09]
    }

    BN = BayesianNetwork('survey')
    BN.properties = {'testing': 'yes'}
    BN.variable_nodes = {
        'AGE': AGE,
        'SEX': SEX,
        'EDU': EDU,
        'OCC': OCC,
        'R': R,
        'TRN': TRN
    }

    return BN
