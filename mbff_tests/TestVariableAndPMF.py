import math
import numpy

from mbff.math.Variable import Variable, JointVariables
from mbff.math.PMF import PMF, CPMF
from mbff.math.Exceptions import VariableInstancesOfUnequalCount

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
            0 : 2,
            1 : 3,
            2 : 2,
            3 : 2,
            4 : 3,
            5 : 2,
            7 : 2 }, PrVariable.value_counts)
        self.assertDictEqual({
            0 : 2/16,
            1 : 3/16,
            2 : 2/16,
            3 : 2/16,
            4 : 3/16,
            5 : 2/16,
            7 : 2/16}, PrVariable.probabilities)
        self.assertEqual(1, sum(PrVariable.values()))

        self.assertEqual(2/16, PrVariable.p(3))
        self.assertEqual(2/16, PrVariable.p(2))
        self.assertEqual(2/16, PrVariable.p(5))

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

        fauna = JointVariables(animals, colors, sizes)
        self.assertListEqual([1, 2, 3], fauna.variableIDs)
        self.assertIs(sizes, fauna.variables[0])
        self.assertIs(colors, fauna.variables[1])
        self.assertIs(animals, fauna.variables[2])
        self.assertListEqual([
            ('large',  'brown',  'cat'),
            ('normal', 'white',  'dog'),
            ('small',  'gray',   'cat'),
            ('small',  'silver', 'mouse'),
            ('small',  'yellow', 'dog')],
            fauna.values)

        PrFauna = PMF(fauna)

        self.assertEqual(2/6, PrFauna.p('small', 'gray', 'cat'))
        self.assertEqual(1/6, PrFauna.p('small', 'silver', 'mouse'))
        self.assertEqual(0,   PrFauna.p('small', 'silver', 'dog'))


    def test_pmf_expected_values(self):
        animals = Variable(['cat', 'dog', 'cat', 'mouse', 'dog', 'cat', 'cat', 'dog'])
        PrAnimals = PMF(animals)

        self.assertAlmostEqual(1.0, PrAnimals.expected_value(lambda v, p: 1))

        # Test calculations of base-e entropy and base-2 entropy.
        self.assertAlmostEqual(-0.97431475, PrAnimals.expected_value(lambda v, p: math.log(p)))
        self.assertAlmostEqual(-1.40563906, PrAnimals.expected_value(lambda v, p: math.log2(p)))

        # Expected word length.
        self.assertEqual(3.25, PrAnimals.expected_value(lambda v, p: len(v)))


    def test_conditional_pmf__binary(self):
        V0 =  Variable([0, 1, 0, 1, 0, 1, 0, 1])
        V1 =  Variable([0, 0, 1, 1, 0, 0, 1, 1])
        V2 =  Variable([0, 0, 0, 0, 1, 1, 1, 1])
        V78 = Variable([0, 0, 0, 0, 0, 0, 1, 1])

        Pr = CPMF(V0, V78)
        self.assertEqual(3/6, Pr.given(0).p(0))
        self.assertEqual(3/6, Pr.given(0).p(1))
        self.assertEqual(1/2, Pr.given(1).p(0))
        self.assertEqual(1/2, Pr.given(1).p(1))

        Pr = CPMF(V2, V78)
        self.assertEqual(4/6, Pr.given(0).p(0))
        self.assertEqual(2/6, Pr.given(0).p(1))
        self.assertEqual(0/2, Pr.given(1).p(0))
        self.assertEqual(2/2, Pr.given(1).p(1))

        Pr = CPMF(V78, V1)
        self.assertEqual(4/4, Pr.given(0).p(0))
        self.assertEqual(0/4, Pr.given(0).p(1))
        self.assertEqual(2/4, Pr.given(1).p(0))
        self.assertEqual(2/4, Pr.given(1).p(1))

        Pr = CPMF(V1, JointVariables(V2, V78))
        self.assertEqual(2/4, Pr.given(0, 0).p(0))
        self.assertEqual(2/4, Pr.given(0, 0).p(1))
        self.assertEqual(0/1, Pr.given(0, 1).p(0))
        self.assertEqual(0/1, Pr.given(0, 1).p(1))
        self.assertEqual(2/2, Pr.given(1, 0).p(0))
        self.assertEqual(0/2, Pr.given(1, 0).p(1))
        self.assertEqual(0/2, Pr.given(1, 1).p(0))
        self.assertEqual(2/2, Pr.given(1, 1).p(1))


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

        self.assertEqual(2/2, Pr.given('small', 'cat').p('gray', 'yes'))
        self.assertEqual(0/1, Pr.given('small', 'cat').p('yellow', 'yes'))
        self.assertEqual(0/1, Pr.given('small', 'cat').p('brown', 'maybe'))

        self.assertEqual(1/1, Pr.given('small', 'dog').p('yellow', 'yes'))
        self.assertEqual(0/1, Pr.given('small', 'dog').p('yellow', 'maybe'))
        self.assertEqual(0/1, Pr.given('small', 'dog').p('silver', 'maybe'))

        self.assertEqual(1/1, Pr.given('large', 'cat').p('brown', 'yes'))
        self.assertEqual(0/1, Pr.given('large', 'cat').p('yellow', 'yes'))

        self.assertEqual(1/1, Pr.given('small', 'snake').p('silver', 'maybe'))
        self.assertEqual(0/1, Pr.given('small', 'snake').p('silver', 'no'))

        self.assertEqual(1/1, Pr.given('normal', 'dog').p('white', 'yes'))
        self.assertEqual(0/1, Pr.given('normal', 'dog').p('silver', 'yes'))
        self.assertEqual(0/1, Pr.given('normal', 'dog').p('yellow', 'maybe'))

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

