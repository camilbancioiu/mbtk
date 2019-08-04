import unittest

from pathlib import Path

from mbff.math.Variable import Variable, JointVariables
import mbff.math.Variable
from mbff.math.PMF import PMF
import mbff.math.G_test__unoptimized

from mbff_tests.TestBase import TestBase


@unittest.skipIf(TestBase.tag_excluded('conditional_independence'), 'Conditional independence tests excluded')
class TestGTestUnoptimized(TestBase):

    @classmethod
    def initTestResources(testClass):
        super(TestGTestUnoptimized, testClass).initTestResources()
        # TODO revert DatasetsInUse
        testClass.DatasetsInUse = ['lungcancer', 'alarm']
        testClass.DatasetsInUse = []
        testClass.DatasetMatrixFolder = Path('testfiles', 'tmp', 'test_gstat_dm')
        testClass.G_test = None


    @classmethod
    def configure_dataset(testClass, dm_label):
        configuration = {}
        if dm_label == 'lungcancer':
            configuration['sourcepath'] = Path('testfiles', 'bif_files', 'lungcancer.bif')
            configuration['sample_count'] = int(5e4)
            configuration['random_seed'] = 129
            configuration['values_as_indices'] = False
            configuration['objectives'] = []

        if dm_label == 'alarm':
            configuration['sourcepath'] = Path('testfiles', 'bif_files', 'alarm.bif')
            configuration['sample_count'] = int(8e3)
            configuration['random_seed'] = 129
            configuration['values_as_indices'] = True
            configuration['objectives'] = []
        return configuration


    def test_calculating_effective_dof(self):
        self.G_test = mbff.math.G_test__unoptimized.G_test(None, dict())

        VarX = Variable([0, 0, 0, 0, 0, 0, 0, 0])
        VarY = Variable([0, 0, 0, 0, 0, 0, 0, 0])
        self.assertJointPMFDoF(VarX, VarY, 0)

        VarX = Variable([1, 0, 0, 0, 0, 0, 0, 0])
        VarY = Variable([0, 0, 0, 0, 0, 0, 0, 0])
        self.assertJointPMFDoF(VarX, VarY, 0)

        VarX = Variable([1, 0, 0, 0, 0, 0, 0, 0])
        VarY = Variable([0, 0, 1, 1, 0, 0, 0, 0])
        self.assertJointPMFDoF(VarX, VarY, 0)

        VarX = Variable([1, 0, 0, 0, 0, 0, 0, 1])
        VarY = Variable([0, 0, 1, 1, 0, 0, 0, 1])
        self.assertJointPMFDoF(VarX, VarY, 1)


    def test_G_value__alarm(self):
        Omega = self.OmegaVariables['alarm']
        dataset = self.DatasetMatrices['alarm']

        parameters = dict()
        parameters['ci_test_significance'] = 0.95
        parameters['ci_test_debug'] = 0
        parameters['omega'] = Omega
        parameters['source_bayesian_network'] = self.BayesianNetworks['alarm']

        X = 35
        Y = 3
        Z = {32, 1, 34, 36}

        self.G_test = mbff.math.G_test__unoptimized.G_test(dataset, parameters)
        self.G_test.conditionally_independent(X, Y, Z)


    def test_G_value__lungcancer(self):
        Omega = self.OmegaVariables['lungcancer']
        lungcancer = self.DatasetMatrices['lungcancer']

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
        parameters['ci_test_debug'] = 1
        parameters['omega'] = Omega
        parameters['source_bayesian_network'] = self.BayesianNetworks['lungcancer']

        self.G_test = mbff.math.G_test__unoptimized.G_test(lungcancer, parameters)

        self.assertCITestAccurate(ASIA, SMOKE, Omega)
        self.assertCITestAccurate(ASIA, LUNG, Omega)
        self.assertCITestAccurate(ASIA, BRONC, Omega)
        self.assertCITestAccurate(ASIA, TUB, Omega)
        self.assertCITestAccurate(ASIA, EITHER, Omega)
        self.assertCITestAccurate(ASIA, XRAY, Omega)
        # self.assertCITestAccurate(ASIA, DYSP, Omega)
        self.assertCITestAccurate(EITHER, ASIA, JointVariables(TUB, LUNG))
        self.assertCITestAccurate(EITHER, SMOKE, JointVariables(TUB, LUNG))
        self.assertCITestAccurate(DYSP, SMOKE, JointVariables(EITHER, BRONC))
        self.assertCITestAccurate(DYSP, LUNG, JointVariables(EITHER, BRONC))
        self.assertCITestAccurate(DYSP, TUB, JointVariables(EITHER, BRONC))
        self.assertCITestAccurate(XRAY, TUB, EITHER)
        self.assertCITestAccurate(XRAY, LUNG, EITHER)
        self.assertCITestAccurate(XRAY, ASIA, EITHER)
        self.assertCITestAccurate(XRAY, SMOKE, EITHER)
        self.assertCITestAccurate(XRAY, DYSP, EITHER)
        self.assertCITestAccurate(XRAY, BRONC, EITHER)
        self.assertCITestAccurate(XRAY, EITHER, Omega)
        self.assertCITestAccurate(XRAY, LUNG, Omega)
        self.assertCITestAccurate(XRAY, SMOKE, Omega)
        self.assertCITestAccurate(XRAY, TUB, Omega)


    def assertJointPMFDoF(self, VarX, VarY, expected_dof):
        VarXY = JointVariables(VarX, VarY)

        PrX = PMF(VarX)
        PrY = PMF(VarY)
        PrXY = PMF(VarXY)

        print()
        print('PrX:', len(PrX))
        print('PrY:', len(PrY))
        print('PrXY:', len(PrXY))

        dof = self.G_test.calculate_degrees_of_freedom__joint_pmf(PrXY, PrX, PrY)
        self.assertEqual(expected_dof, dof)


    def assertCITestAccurate(self, X, Y, Z):
        if isinstance(Z, mbff.math.Variable.Omega):
            Z_ID = []
        elif isinstance(Z, JointVariables):
            Z_ID = Z.variableIDs
        elif isinstance(Z, mbff.math.Variable.Variable):
            Z_ID = [Z.ID]

        result = self.G_test.G_test_conditionally_independent(X, Y, Z, X.ID, Y.ID, Z_ID)
        if self.G_test.source_bn is not None:
            result.computed_d_separation = self.G_test.source_bn.d_separated(X.ID, Z_ID, Y.ID)
        result.index = len(self.G_test.ci_test_results)
        self.G_test.ci_test_results.append(result)
        print(result, result.test_distribution_parameters)
        print()
        self.assertTrue(result.accurate())
