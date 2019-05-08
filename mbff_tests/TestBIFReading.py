from collections import OrderedDict
from pathlib import Path

from mbff_tests.TestBase import TestBase

import mbff.utilities.functions as util
from mbff.math.BayesianNetwork import *

class TestBIFReading(TestBase):

    def test_reading_bif_file(self):
        survey_bif = Path('testfiles', 'bif_files', 'survey.bif')

        bn_expected = self.default_Bayesian_network()
        bn_read = util.read_bif_file(survey_bif)

        self.assertBayesianNetworkEqual(bn_expected, bn_read)


    def assertBayesianNetworkEqual(self, bnA, bnB):
        self.assertEqual(bnA.name, bnB.name)
        self.assertDictEqual(bnA.properties, bnB.properties)
        varnamesA = bnA.variable_names()
        varnamesB = bnB.variable_names()
        self.assertListEqual(varnamesA, varnamesB)
        for varname in varnamesA:
            varA = bnA.variables[varname]
            varB = bnB.variables[varname]
            self.assertVariablesEqual(varA, varB)


    def assertVariablesEqual(self, A, B):
        self.assertEqual(A.name, B.name)
        self.assertListEqual(A.values, B.values)
        self.assertDictEqual(A.properties, B.properties)
        self.assertProbabilityDistributionsEqual(A.probdist, B.probdist)


    def assertProbabilityDistributionsEqual(self, pA, pB):
        self.assertEqual(len(pA.conditioning_variables), len(pB.conditioning_variables))
        for ourVar, otherVar in zip(pA.conditioning_variables.values(), pB.conditioning_variables.values()):
            self.assertListEqual(ourVar.values, otherVar.values)
        self.assertDictEqual(pA.probabilities, pB.probabilities)
        self.assertDictEqual(pA.properties, pB.properties)


    def default_Bayesian_network(self):
        AGE = Variable('AGE')
        AGE.values = ['young', 'adult', 'old']
        AGE.properties = {'label' : 'age'}
        AGE.probdist = ProbabilityDistribution(AGE)
        AGE.probdist.conditioning_variables = OrderedDict()
        AGE.probdist.probabilities = { '<unconditioned>' : [0.3, 0.5, 0.2] }

        SEX = Variable('SEX')
        SEX.values = ['M', 'F']
        SEX.properties = {'label' : 'sex'}
        SEX.probdist = ProbabilityDistribution(SEX)
        SEX.probdist.conditioning_variables = OrderedDict()
        SEX.probdist.probabilities = { '<unconditioned>' : [0.49, 0.51] }

        EDU = Variable('EDU')
        EDU.values = ['highschool', 'uni']
        EDU.properties = {'label': 'education'}
        EDU.probdist = ProbabilityDistribution(AGE)
        EDU.probdist.conditioning_variables = OrderedDict([('AGE', AGE), ('SEX', SEX)])
        EDU.probdist.probabilities = {
                ('young', 'M')    : [0.75, 0.25],
                ('young', 'F')    : [0.64, 0.36],
                ('adult', 'M')    : [0.72, 0.28],
                ('adult', 'F')    : [0.70, 0.30],
                ('old', 'M')  : [0.88, 0.12],
                ('old', 'F')  : [0.90, 0.10]
                }

        OCC = Variable('OCC')
        OCC.values = ['emp', 'self']
        OCC.properties = {'label' : 'occupation'}
        OCC.probdist = ProbabilityDistribution(OCC)
        OCC.probdist.conditioning_variables = OrderedDict([('EDU', EDU)])
        OCC.probdist.probabilities = {
                ('highschool',) : [0.96, 0.04],
                ('uni',)  : [0.92, 0.08]
                }

        R = Variable('R')
        R.values = ['small', 'big']
        R.properties = {'label' : 'unknown'}
        R.probdist = ProbabilityDistribution(R)
        R.probdist.conditioning_variables = OrderedDict([('EDU', EDU)])
        R.probdist.probabilities = {
                ('highschool',) : [0.25, 0.75],
                ('uni',)  : [0.2, 0.8]
                }

        TRN = Variable('TRN')
        TRN.values = ['car', 'train', 'other']
        TRN.properties = {'label' : 'transportation'}
        TRN.probdist = ProbabilityDistribution(TRN)
        TRN.probdist.conditioning_variables = OrderedDict([('OCC', OCC), ('R', R)])
        TRN.probdist.probabilities = {
                ('emp', 'small')  : [0.48, 0.42, 0.10],
                ('self', 'small') : [0.56, 0.36, 0.08],
                ('emp', 'big')    : [0.58, 0.24, 0.18],
                ('self', 'big')  : [0.70, 0.21, 0.09]
                }

        BN = BayesianNetwork('survey')
        BN.properties = { 'testing' : 'yes' }
        BN.variables = {
                'AGE' : AGE,
                'SEX' : SEX,
                'EDU' : EDU,
                'OCC' : OCC,
                'R'   : R,
                'TRN' : TRN
                }

        return BN


