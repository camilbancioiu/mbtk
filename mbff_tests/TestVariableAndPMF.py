import math
import unittest
import numpy

from pathlib import Path

import mbff.utilities.functions as util

from mbff.math.Variable import Variable, JointVariables
from mbff.math.PMF import PMF, CPMF, process_pmf_key
from mbff.math.Exceptions import VariableInstancesOfUnequalCount
from mbff.dataset.sources.SampledBayesianNetworkDatasetSource import SampledBayesianNetworkDatasetSource

from mbff_tests.TestBase import TestBase


class TestVariableAndPMF(TestBase):

    def test_single_variable_pmf(self):
        variable = Variable(numpy.array([3, 5, 1, 1, 4, 3, 7, 0, 2, 1, 0, 5, 4, 7, 2, 4]))
        variable.ID = 1
        variable.name = 'test_variable_1'

        variable.update_values()
        self.assertListEqual([0, 1, 2, 3, 4, 5, 7], variable.values)

        PrVariable = PMF(variable)
        self.assertDictEqual({
            0: 2,
            1: 3,
            2: 2,
            3: 2,
            4: 3,
            5: 2,
            7: 2}, PrVariable.value_counts)
        self.assertDictEqual({
            0: 2 / 16,
            1: 3 / 16,
            2: 2 / 16,
            3: 2 / 16,
            4: 3 / 16,
            5: 2 / 16,
            7: 2 / 16}, PrVariable.probabilities)
        self.assertEqual(1, sum(PrVariable.values()))

        self.assertEqual(2 / 16, PrVariable.p(3))
        self.assertEqual(2 / 16, PrVariable.p(2))
        self.assertEqual(2 / 16, PrVariable.p(5))

        ev = 0
        for (v, pv) in PrVariable.items():
            ev += pv * v

        self.assertEqual(3.0625, ev)


    def test_joint_variables_pmf(self):
        animals = Variable(['cat', 'dog', 'cat', 'mouse', 'dog', 'cat'])
        animals.ID = 3
        animals.name = 'animals'

        colors = Variable(['gray', 'yellow', 'brown', 'silver', 'white', 'gray'])
        colors.ID = 2
        colors.name = 'colors'

        sizes = Variable(['small', 'small', 'large', 'small', 'normal', 'small'])
        sizes.ID = 1
        sizes.name = 'sizes'

        fauna = JointVariables(sizes, colors, animals)
        fauna.update_values()
        self.assertListEqual([1, 2, 3], fauna.variableIDs)
        self.assertIs(sizes, fauna.variables[0])
        self.assertIs(colors, fauna.variables[1])
        self.assertIs(animals, fauna.variables[2])
        self.assertListEqual([
            ('large', 'brown', 'cat'),
            ('normal', 'white', 'dog'),
            ('small', 'gray', 'cat'),
            ('small', 'silver', 'mouse'),
            ('small', 'yellow', 'dog')],
            fauna.values)

        PrFauna = PMF(fauna)

        self.assertEqual(2 / 6, PrFauna.p('small', 'gray', 'cat'))
        self.assertEqual(1 / 6, PrFauna.p('small', 'silver', 'mouse'))
        self.assertEqual(0, PrFauna.p('small', 'silver', 'dog'))

        singleton_joint = JointVariables(animals)
        self.assertEqual(['cat', 'dog', 'cat', 'mouse', 'dog', 'cat'], singleton_joint.instances())


    def test_pmf_expected_values(self):
        animals = Variable(['cat', 'dog', 'cat', 'mouse', 'dog', 'cat', 'cat', 'dog'])
        PrAnimals = PMF(animals)

        self.assertAlmostEqual(1.0, PrAnimals.expected_value(lambda v, p: 1))

        # Test calculations of base-e entropy and base-2 entropy.
        self.assertAlmostEqual(0.97431475, (-1) * PrAnimals.expected_value(lambda v, p: math.log(p)))
        self.assertAlmostEqual(1.40563906, (-1) * PrAnimals.expected_value(lambda v, p: math.log2(p)))

        # Expected word length.
        self.assertEqual(3.25, PrAnimals.expected_value(lambda v, p: len(v)))


    def test_conditional_pmf__binary(self):
        V0 = Variable([0, 1, 0, 1, 0, 1, 0, 1])
        V1 = Variable([0, 0, 1, 1, 0, 0, 1, 1])
        V2 = Variable([0, 0, 0, 0, 1, 1, 1, 1])
        V78 = Variable([0, 0, 0, 0, 0, 0, 1, 1])

        Pr = CPMF(V0, V78)
        self.assertEqual(3 / 6, Pr.given(0).p(0))
        self.assertEqual(3 / 6, Pr.given(0).p(1))
        self.assertEqual(1 / 2, Pr.given(1).p(0))
        self.assertEqual(1 / 2, Pr.given(1).p(1))

        Pr = CPMF(V2, V78)
        self.assertEqual(4 / 6, Pr.given(0).p(0))
        self.assertEqual(2 / 6, Pr.given(0).p(1))
        self.assertEqual(0 / 2, Pr.given(1).p(0))
        self.assertEqual(2 / 2, Pr.given(1).p(1))

        Pr = CPMF(V78, V1)
        self.assertEqual(4 / 4, Pr.given(0).p(0))
        self.assertEqual(0 / 4, Pr.given(0).p(1))
        self.assertEqual(2 / 4, Pr.given(1).p(0))
        self.assertEqual(2 / 4, Pr.given(1).p(1))

        Pr = CPMF(V1, JointVariables(V2, V78))
        self.assertEqual(2 / 4, Pr.given(0, 0).p(0))
        self.assertEqual(2 / 4, Pr.given(0, 0).p(1))
        self.assertEqual(0 / 1, Pr.given(0, 1).p(0))
        self.assertEqual(0 / 1, Pr.given(0, 1).p(1))
        self.assertEqual(2 / 2, Pr.given(1, 0).p(0))
        self.assertEqual(0 / 2, Pr.given(1, 0).p(1))
        self.assertEqual(0 / 2, Pr.given(1, 1).p(0))
        self.assertEqual(2 / 2, Pr.given(1, 1).p(1))


    def test_conditional_pmf__multiple_values(self):
        sizes = Variable(['small', 'small', 'large', 'small', 'normal', 'small'])
        sizes.ID = 1
        sizes.name = 'sizes'

        colors = Variable(['gray', 'yellow', 'brown', 'silver', 'white', 'gray'])
        colors.ID = 2
        colors.name = 'colors'

        animals = Variable(['cat', 'dog', 'cat', 'snake', 'dog', 'cat'])
        animals.ID = 3
        animals.name = 'animals'

        is_pet = Variable(['yes', 'yes', 'yes', 'maybe', 'yes', 'yes'])
        is_pet.ID = 4
        is_pet.name = 'is_pet'

        Pr = CPMF(JointVariables(colors, is_pet), JointVariables(sizes, animals))

        self.assertEqual(2 / 2, Pr.given('small', 'cat').p('gray', 'yes'))
        self.assertEqual(0 / 1, Pr.given('small', 'cat').p('yellow', 'yes'))
        self.assertEqual(0 / 1, Pr.given('small', 'cat').p('brown', 'maybe'))

        self.assertEqual(1 / 1, Pr.given('small', 'dog').p('yellow', 'yes'))
        self.assertEqual(0 / 1, Pr.given('small', 'dog').p('yellow', 'maybe'))
        self.assertEqual(0 / 1, Pr.given('small', 'dog').p('silver', 'maybe'))

        self.assertEqual(1 / 1, Pr.given('large', 'cat').p('brown', 'yes'))
        self.assertEqual(0 / 1, Pr.given('large', 'cat').p('yellow', 'yes'))

        self.assertEqual(1 / 1, Pr.given('small', 'snake').p('silver', 'maybe'))
        self.assertEqual(0 / 1, Pr.given('small', 'snake').p('silver', 'no'))

        self.assertEqual(1 / 1, Pr.given('normal', 'dog').p('white', 'yes'))
        self.assertEqual(0 / 1, Pr.given('normal', 'dog').p('silver', 'yes'))
        self.assertEqual(0 / 1, Pr.given('normal', 'dog').p('yellow', 'maybe'))

        SA = JointVariables(sizes, animals)
        PrAll = CPMF(JointVariables(colors, is_pet), SA)
        PrSA = PMF(SA)
        PrCcSA = CPMF(colors, SA)
        PrIPcSA = CPMF(is_pet, SA)

        test_p_all = 0.0
        test_p_c = 0.0
        test_p_ip = 0.0

        for (sa, psa) in PrSA.items():
            for (c, pcsa) in PrCcSA.given(sa).items():
                test_p_c += pcsa * PrSA.p(sa)
                for (ip, pipsa) in PrIPcSA.given(sa).items():
                    pall = PrAll.given(sa).p(c, ip)
                    test_p_all += pall * PrSA.p(sa)
                    test_p_ip += pipsa * PrSA.p(sa)

        self.assertAlmostEqual(1, test_p_all)
        self.assertAlmostEqual(1, test_p_c)
        self.assertAlmostEqual(1, test_p_ip)


    @unittest.skipIf(TestBase.tag_excluded('sampling'), 'Sampling tests excluded')
    def test_conditional_pmf__from_bayesian_network(self):
        configuration = {}
        configuration['sourcepath'] = Path('testfiles', 'bif_files', 'survey.bif')
        configuration['sample_count'] = int(4e4)
        # Using a random seed of 42 somehow requires 2e6 samples to pass, but
        # with the seed 1984, it is sufficient to generate only 4e4. Maybe the
        # random generator is biased somehow?
        configuration['random_seed'] = 1984
        configuration['values_as_indices'] = False
        configuration['objectives'] = ['R', 'TRN']

        bayesian_network = util.read_bif_file(configuration['sourcepath'])
        bayesian_network.finalize()

        sbnds = SampledBayesianNetworkDatasetSource(configuration)
        sbnds.reset_random_seed = True
        datasetmatrix = sbnds.create_dataset_matrix('test_sbnds')

        self.assertEqual(['AGE', 'EDU', 'OCC', 'SEX'], datasetmatrix.column_labels_X)
        self.assertEqual(['R', 'TRN'], datasetmatrix.column_labels_Y)

        delta = 0.008

        AGE = Variable(datasetmatrix.get_column_by_label('X', 'AGE'))
        PrAge = PMF(AGE)

        SEX = Variable(datasetmatrix.get_column_by_label('X', 'SEX'))
        PrSex = PMF(SEX)

        self.assert_PMF_AlmostEquals_BNProbDist(
            bayesian_network.variable_nodes['AGE'].probdist,
            PrAge,
            delta=delta)

        self.assert_PMF_AlmostEquals_BNProbDist(
            bayesian_network.variable_nodes['SEX'].probdist,
            PrSex,
            delta=delta)

        EDU = Variable(datasetmatrix.get_column_by_label('X', 'EDU'))
        PrEdu = CPMF(EDU, given=JointVariables(AGE, SEX))

        self.assert_CPMF_AlmostEquals_BNProbDist(
            bayesian_network.variable_nodes['EDU'].probdist,
            PrEdu,
            delta=delta)

        OCC = Variable(datasetmatrix.get_column_by_label('X', 'OCC'))
        PrOcc = CPMF(OCC, given=EDU)

        self.assert_CPMF_AlmostEquals_BNProbDist(
            bayesian_network.variable_nodes['OCC'].probdist,
            PrOcc,
            delta=delta)

        R = Variable(datasetmatrix.get_column_by_label('Y', 'R'))
        PrR = CPMF(R, given=EDU)

        self.assert_CPMF_AlmostEquals_BNProbDist(
            bayesian_network.variable_nodes['R'].probdist,
            PrR,
            delta=delta)

        TRN = Variable(datasetmatrix.get_column_by_label('Y', 'TRN'))
        PrTRN = CPMF(TRN, given=JointVariables(OCC, R))

        self.assert_CPMF_AlmostEquals_BNProbDist(
            bayesian_network.variable_nodes['TRN'].probdist,
            PrTRN,
            delta=delta)


    def test_pmf_key_processing(self):
        key = (0, 1)
        expected = (0, 1)
        self.assertEqual(expected, process_pmf_key(key))

        key = (2,)
        expected = 2
        self.assertEqual(expected, process_pmf_key(key))

        key = (0, (1,))
        expected = (0, 1)
        self.assertEqual(expected, process_pmf_key(key))

        key = ((0, 1), 2)
        expected = (0, 1, 2)
        self.assertEqual(expected, process_pmf_key(key))

        key = ((0, 1), (5, 6, (7,)))
        expected = (0, 1, 5, 6, 7)
        self.assertEqual(expected, process_pmf_key(key))

        key = (0, 1, [2, 3, 4, (5, 6), 7])
        expected = (0, 1, 2, 3, 4, 5, 6, 7)
        self.assertEqual(expected, process_pmf_key(key))

        key = [0, 1]
        expected = (0, 1)
        self.assertEqual(expected, process_pmf_key(key))

        key = [(0, 1), 2]
        expected = (0, 1, 2)
        self.assertEqual(expected, process_pmf_key(key))

        key = [(0, 1), [2], [3, 4, (5,)]]
        expected = (0, 1, 2, 3, 4, 5)
        self.assertEqual(expected, process_pmf_key(key))


    def test_joint_variables__unequal_numbers_of_instances(self):
        # Variable animals has 6 instances.
        animals = Variable(['cat', 'dog', 'cat', 'mouse', 'dog', 'cat'])
        animals.ID = 3
        animals.name = 'animals'

        # Variable colors has 6 instances.
        colors = Variable(['gray', 'yellow', 'brown', 'silver', 'white', 'gray'])
        colors.ID = 2
        colors.name = 'colors'

        # Variable sizes has only 5 instances, which will cause an error.
        sizes = Variable(['small', 'small', 'large', 'small', 'normal'])
        sizes.ID = 1
        sizes.name = 'sizes'

        with self.assertRaises(VariableInstancesOfUnequalCount):
            fauna = JointVariables(animals, colors, sizes)

        sizes = Variable(['small', 'small', 'large', 'small', 'normal', 'small'])
        sizes.ID = 1
        sizes.name = 'sizes'

        fauna = JointVariables(animals, colors, sizes)

        can_fly = Variable([False, False, False, False, False])
        can_fly.ID = 4
        can_fly.name = 'can_fly'

        with self.assertRaises(VariableInstancesOfUnequalCount):
            JointVariables(fauna, can_fly)


    def assert_PMF_AlmostEquals_BNProbDist(self, probdist, pmf, delta):
        probdist_dict = dict(enumerate(probdist.probabilities['<unconditioned>']))
        pmf_dict = pmf.probabilities
        self.assertEqual(set(probdist_dict.keys()), set(pmf_dict.keys()))
        for key in probdist_dict.keys():
            self.assertAlmostEqual(probdist_dict[key], pmf_dict[key], delta=delta)


    def assert_CPMF_AlmostEquals_BNProbDist(self, probdist, cpmf, delta):
        indexed_probdist = probdist.probabilities_with_indexed_conditioning
        for conditioning_key, probabilities in indexed_probdist.items():
            probdist_dict = dict(enumerate(indexed_probdist[conditioning_key]))
            pmf_dict = cpmf.given(conditioning_key).probabilities
            self.assertEqual(set(probdist_dict.keys()), set(pmf_dict.keys()))
            for key in probdist_dict.keys():
                try:
                    self.assertAlmostEqual(probdist_dict[key], pmf_dict[key], delta=delta)
                except AssertionError:
                    print("Key", key)
                    print("Conditioning key", conditioning_key)
                    raise
