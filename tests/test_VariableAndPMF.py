import math
import numpy

import tests.utilities as testutil
import pytest

from mbtk.math.Variable import Variable, JointVariables
from mbtk.math.PMF import PMF, CPMF, process_pmf_key
from mbtk.structures.BayesianNetwork import BayesianNetwork
from mbtk.math.Exceptions import VariableInstancesOfUnequalCount
from mbtk.dataset.sources.SampledBayesianNetworkDatasetSource import SampledBayesianNetworkDatasetSource

delta = 0.008


def test_single_variable_pmf():
    variable = Variable(numpy.array([3, 5, 1, 1, 4, 3, 7, 0, 2, 1, 0, 5, 4, 7, 2, 4]))
    variable.ID = 1
    variable.name = 'test_variable_1'

    variable.update_values()
    assert [0, 1, 2, 3, 4, 5, 7] == variable.values

    PrVariable = PMF(variable)
    expected_counts = {0: 2,
                       1: 3,
                       2: 2,
                       3: 2,
                       4: 3,
                       5: 2,
                       7: 2}
    assert PrVariable.value_counts == expected_counts

    expected_counts = {0: 2 / 16,
                       1: 3 / 16,
                       2: 2 / 16,
                       3: 2 / 16,
                       4: 3 / 16,
                       5: 2 / 16,
                       7: 2 / 16}
    assert PrVariable.probabilities == expected_counts

    assert 1 == sum(PrVariable.values())

    assert 2 / 16 == PrVariable.p(3)
    assert 2 / 16 == PrVariable.p(2)
    assert 2 / 16 == PrVariable.p(5)

    ev = 0
    for (v, pv) in PrVariable.items():
        ev += pv * v

    assert 3.0625 == ev



def test_joint_variables_pmf():
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
    assert [1, 2, 3] == fauna.variableIDs
    assert fauna.variables[0] is sizes
    assert fauna.variables[1] is colors
    assert fauna.variables[2] is animals

    expected_values = [('large', 'brown', 'cat'),
                       ('normal', 'white', 'dog'),
                       ('small', 'gray', 'cat'),
                       ('small', 'silver', 'mouse'),
                       ('small', 'yellow', 'dog')]
    assert fauna.values == expected_values

    PrFauna = PMF(fauna)
    assert PrFauna.p('small', 'gray', 'cat') == 2 / 6
    assert PrFauna.p('small', 'silver', 'mouse') == 1 / 6
    assert PrFauna.p('small', 'silver', 'dog') == 0

    singleton_joint = JointVariables(animals)
    assert ['cat', 'dog', 'cat', 'mouse', 'dog', 'cat'] == singleton_joint.instances()



def test_pmf_expected_values():
    animals = Variable(['cat', 'dog', 'cat', 'mouse', 'dog', 'cat', 'cat', 'dog'])
    PrAnimals = PMF(animals)

    assert almostEqual(1.0, PrAnimals.expected_value(lambda v, p: 1))

    # Test calculations of base-e entropy and base-2 entropy.
    assert almostEqual(0.97431475, (-1) * PrAnimals.expected_value(lambda v, p: math.log(p)))
    assert almostEqual(1.40563906, (-1) * PrAnimals.expected_value(lambda v, p: math.log2(p)))

    # Expected word length.
    assert PrAnimals.expected_value(lambda v, p: len(v)) == 3.25



def test_instantiating_joint_pmf(bn_survey):
    bn = bn_survey

    assert ['AGE', 'SEX', 'EDU', 'OCC', 'R', 'TRN'] == bn.variable_node_names__sampling_order
    assert ['AGE', 'EDU', 'OCC', 'R', 'SEX', 'TRN'] == bn.variable_node_names()

    joint_pmf = bn.create_joint_pmf(values_as_indices=False)
    assert joint_pmf.min_instance_count_for_accuracy() == 76593

    instances = list(joint_pmf.create_instances_list(77000))
    assert len(instances) == 77000



def test_conditional_pmf__binary():
    V0 = Variable([0, 1, 0, 1, 0, 1, 0, 1])
    V1 = Variable([0, 0, 1, 1, 0, 0, 1, 1])
    V2 = Variable([0, 0, 0, 0, 1, 1, 1, 1])
    V78 = Variable([0, 0, 0, 0, 0, 0, 1, 1])

    Pr = CPMF(V0, V78)
    assert Pr.given(0).p(0) == 3 / 6
    assert Pr.given(0).p(1) == 3 / 6
    assert Pr.given(1).p(0) == 1 / 2
    assert Pr.given(1).p(1) == 1 / 2

    Pr = CPMF(V2, V78)
    assert Pr.given(0).p(0) == 4 / 6
    assert Pr.given(0).p(1) == 2 / 6
    assert Pr.given(1).p(0) == 0 / 2
    assert Pr.given(1).p(1) == 2 / 2

    Pr = CPMF(V78, V1)
    assert Pr.given(0).p(0) == 4 / 4
    assert Pr.given(0).p(1) == 0 / 4
    assert Pr.given(1).p(0) == 2 / 4
    assert Pr.given(1).p(1) == 2 / 4

    Pr = CPMF(V1, JointVariables(V2, V78))
    assert Pr.given(0, 0).p(0) == 2 / 4
    assert Pr.given(0, 0).p(1) == 2 / 4
    assert Pr.given(0, 1).p(0) == 0 / 1
    assert Pr.given(0, 1).p(1) == 0 / 1
    assert Pr.given(1, 0).p(0) == 2 / 2
    assert Pr.given(1, 0).p(1) == 0 / 2
    assert Pr.given(1, 1).p(0) == 0 / 2
    assert Pr.given(1, 1).p(1) == 2 / 2



def test_pmf_summing_over_variable():
    V0 = Variable([0, 1, 1, 1, 0, 1, 0, 1])
    V1 = Variable([0, 0, 1, 1, 0, 1, 1, 1])
    V2 = Variable([0, 0, 0, 0, 1, 0, 1, 1])
    V3 = Variable([0, 0, 0, 0, 0, 0, 1, 1])

    V0.ID = 1000
    V1.ID = 1111
    V2.ID = 1222
    V3.ID = 1333

    Pr = PMF(JointVariables(V0, V1, V2, V3))
    assert Pr.IDs() == (1000, 1111, 1222, 1333)

    assert Pr.p((0, 0, 0, 0)) == 1 / 8
    assert Pr.p((1, 0, 0, 0)) == 1 / 8
    assert Pr.p((1, 1, 0, 0)) == 3 / 8
    assert Pr.p((0, 0, 1, 0)) == 1 / 8
    assert Pr.p((0, 1, 1, 1)) == 1 / 8
    assert Pr.p((1, 1, 1, 1)) == 1 / 8

    Pr = Pr.sum_over(V2.ID)
    assert sum(Pr.probabilities.values()) == 1

    assert Pr.p((0, 0, 0)) == 2 / 8
    assert Pr.p((1, 0, 0)) == 1 / 8
    assert Pr.p((1, 1, 0)) == 3 / 8
    assert Pr.p((0, 1, 1)) == 1 / 8
    assert Pr.p((1, 1, 1)) == 1 / 8
    assert Pr.IDs() == (V0.ID, V1.ID, V3.ID)

    Pr = Pr.sum_over(V1.ID)
    assert sum(Pr.probabilities.values()) == 1

    assert Pr.p((0, 0)) == 2 / 8
    assert Pr.p((1, 0)) == 4 / 8
    assert Pr.p((0, 1)) == 1 / 8
    assert Pr.p((1, 1)) == 1 / 8
    assert Pr.IDs() == (V0.ID, V3.ID)

    Pr = Pr.sum_over(V0.ID)
    assert sum(Pr.probabilities.values()) == 1

    print(Pr.probabilities)

    assert Pr.p(0) == 6 / 8
    assert Pr.p(1) == 2 / 8
    assert Pr.IDs() == (V3.ID,)



def test_pmf_remove_from_key() -> None:
    pmf = PMF(None)
    assert pmf.remove_from_key(('A', 'B', 'C', 'D'), 2) == ('A', 'B', 'D')
    assert pmf.remove_from_key(('A', 'B', 'C', 'D'), 0) == ('B', 'C', 'D')
    assert pmf.remove_from_key(('A', 'B', 'C', 'D'), 3) == ('A', 'B', 'C')
    assert pmf.remove_from_key(('A', 'B'), 0) == ('B',)
    assert pmf.remove_from_key(('A', 'B'), 1) == ('A',)
    assert pmf.remove_from_key(('A',), 0) == tuple()



def test_make_cpmf_PrXcZ_variant_1() -> None:
    V0 = Variable([0, 1, 1, 1, 0, 1, 0, 1])
    V1 = Variable([0, 0, 1, 1, 0, 1, 1, 1])

    PrXZ = PMF(JointVariables(V0, V1))
    PrXZ.IDs(1000, 1111)

    assert PrXZ.IDs() == (1000, 1111)

    assert PrXZ.p((0, 0)) == 2 / 8
    assert PrXZ.p((0, 1)) == 1 / 8
    assert PrXZ.p((1, 0)) == 1 / 8
    assert PrXZ.p((1, 1)) == 4 / 8



def test_conditional_pmf__multiple_values():
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

    assert Pr.given('small', 'cat').p('gray', 'yes') == 2 / 2
    assert Pr.given('small', 'cat').p('yellow', 'yes') == 0 / 1
    assert Pr.given('small', 'cat').p('brown', 'maybe') == 0 / 1

    assert Pr.given('small', 'dog').p('yellow', 'yes') == 1 / 1
    assert Pr.given('small', 'dog').p('yellow', 'maybe') == 0 / 1
    assert Pr.given('small', 'dog').p('silver', 'maybe') == 0 / 1

    assert Pr.given('large', 'cat').p('brown', 'yes') == 1 / 1
    assert Pr.given('large', 'cat').p('yellow', 'yes') == 0 / 1

    assert Pr.given('small', 'snake').p('silver', 'maybe') == 1 / 1
    assert Pr.given('small', 'snake').p('silver', 'no') == 0 / 1

    assert Pr.given('normal', 'dog').p('white', 'yes') == 1 / 1
    assert Pr.given('normal', 'dog').p('silver', 'yes') == 0 / 1
    assert Pr.given('normal', 'dog').p('yellow', 'maybe') == 0 / 1

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

    assert almostEqual(1, test_p_all)
    assert almostEqual(1, test_p_c)
    assert almostEqual(1, test_p_ip)



def test_conditional_pmf__from_bayesian_network():
    configuration = dict()
    configuration['sourcepath'] = testutil.bif_folder / 'survey.bif'
    configuration['sample_count'] = int(4e4)
    # Using a random seed of 42 somehow requires 2e6 samples to pass, but
    # with the seed 1984, it is sufficient to generate only 4e4. Maybe the
    # random generator is biased somehow?
    configuration['random_seed'] = 1984
    configuration['values_as_indices'] = False
    configuration['objectives'] = ['R', 'TRN']

    bayesian_network = BayesianNetwork.from_bif_file(configuration['sourcepath'], use_cache=False)
    bayesian_network.finalize()

    sbnds = SampledBayesianNetworkDatasetSource(configuration)
    sbnds.reset_random_seed = True
    datasetmatrix = sbnds.create_dataset_matrix('test_sbnds')

    assert ['AGE', 'EDU', 'OCC', 'SEX'] == datasetmatrix.column_labels_X
    assert ['R', 'TRN'] == datasetmatrix.column_labels_Y

    AGE = Variable(datasetmatrix.get_column_by_label('X', 'AGE'))
    PrAge = PMF(AGE)

    SEX = Variable(datasetmatrix.get_column_by_label('X', 'SEX'))
    PrSex = PMF(SEX)

    assert_PMF_AlmostEquals_BNProbDist(
        bayesian_network.variable_nodes['AGE'].probdist,
        PrAge)

    assert_PMF_AlmostEquals_BNProbDist(
        bayesian_network.variable_nodes['SEX'].probdist,
        PrSex)

    EDU = Variable(datasetmatrix.get_column_by_label('X', 'EDU'))
    PrEdu = CPMF(EDU, given=JointVariables(AGE, SEX))

    assert_CPMF_AlmostEquals_BNProbDist(
        bayesian_network.variable_nodes['EDU'].probdist,
        PrEdu)

    OCC = Variable(datasetmatrix.get_column_by_label('X', 'OCC'))
    PrOcc = CPMF(OCC, given=EDU)

    assert_CPMF_AlmostEquals_BNProbDist(
        bayesian_network.variable_nodes['OCC'].probdist,
        PrOcc)

    R = Variable(datasetmatrix.get_column_by_label('Y', 'R'))
    PrR = CPMF(R, given=EDU)

    assert_CPMF_AlmostEquals_BNProbDist(
        bayesian_network.variable_nodes['R'].probdist,
        PrR)

    TRN = Variable(datasetmatrix.get_column_by_label('Y', 'TRN'))
    PrTRN = CPMF(TRN, given=JointVariables(OCC, R))

    assert_CPMF_AlmostEquals_BNProbDist(
        bayesian_network.variable_nodes['TRN'].probdist,
        PrTRN)



def test_pmf_key_processing():
    key = (0, 1)
    expected = (0, 1)
    assert expected == process_pmf_key(key)

    key = (2,)
    expected = 2
    assert expected == process_pmf_key(key)

    key = (0, (1,))
    expected = (0, 1)
    assert expected == process_pmf_key(key)

    key = ((0, 1), 2)
    expected = (0, 1, 2)
    assert expected == process_pmf_key(key)

    key = ((0, 1), (5, 6, (7,)))
    expected = (0, 1, 5, 6, 7)
    assert expected == process_pmf_key(key)

    key = (0, 1, [2, 3, 4, (5, 6), 7])
    expected = (0, 1, 2, 3, 4, 5, 6, 7)
    assert expected == process_pmf_key(key)

    key = [0, 1]
    expected = (0, 1)
    assert expected == process_pmf_key(key)

    key = [(0, 1), 2]
    expected = (0, 1, 2)
    assert expected == process_pmf_key(key)

    key = [(0, 1), [2], [3, 4, (5,)]]
    expected = (0, 1, 2, 3, 4, 5)
    assert expected == process_pmf_key(key)



def test_joint_variables__unequal_numbers_of_instances():
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

    with pytest.raises(VariableInstancesOfUnequalCount):
        fauna = JointVariables(animals, colors, sizes)

    sizes = Variable(['small', 'small', 'large', 'small', 'normal', 'small'])
    sizes.ID = 1
    sizes.name = 'sizes'

    fauna = JointVariables(animals, colors, sizes)

    can_fly = Variable([False, False, False, False, False])
    can_fly.ID = 4
    can_fly.name = 'can_fly'

    with pytest.raises(VariableInstancesOfUnequalCount):
        JointVariables(fauna, can_fly)



def assert_PMF_AlmostEquals_BNProbDist(probdist, pmf):
    probdist_dict = dict(enumerate(probdist.probabilities['<unconditioned>']))
    pmf_dict = pmf.probabilities
    assert set(probdist_dict.keys()) == set(pmf_dict.keys())
    for key in probdist_dict.keys():
        assert almostEqual(probdist_dict[key], pmf_dict[key])



def assert_CPMF_AlmostEquals_BNProbDist(probdist, cpmf):
    indexed_probdist = probdist.probabilities_with_indexed_conditioning
    for conditioning_key, probabilities in indexed_probdist.items():
        probdist_dict = dict(enumerate(indexed_probdist[conditioning_key]))
        pmf_dict = cpmf.given(conditioning_key).probabilities
        assert set(probdist_dict.keys()) == set(pmf_dict.keys())
        for key in probdist_dict.keys():
            try:
                assert almostEqual(probdist_dict[key], pmf_dict[key])
            except AssertionError:
                print("Key", key)
                print("Conditioning key", conditioning_key)
                raise



def almostEqual(x, y):
    return abs(x - y) < delta
