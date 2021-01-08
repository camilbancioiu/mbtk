from mbtk.structures.ADTree import ADTree
from mbtk.math.PMF import PMF, CPMF
import mbtk.math.DoFCalculators
import mbtk.math.G_test__unoptimized
import mbtk.math.G_test__with_AD_tree


def test_simple_ADTree_query_count(data_small_1):
    dataset, column_values = data_small_1
    adtree = ADTree(dataset, column_values)

    assert adtree.query_count({}) == 8
    assert adtree.query_count({0: 1}) == 1
    assert adtree.query_count({0: 2}) == 3
    assert adtree.query_count({0: 3}) == 4

    assert adtree.query_count({0: 1, 1: 1}) == 0
    assert adtree.query_count({0: 1, 1: 2}) == 1

    assert adtree.query_count({0: 2, 1: 1}) == 2
    assert adtree.query_count({0: 2, 1: 2}) == 1

    assert adtree.query_count({0: 3, 1: 1}) == 1
    assert adtree.query_count({0: 3, 1: 2}) == 3

    assert adtree.query_count({1: 1}) == 3
    assert adtree.query_count({1: 2}) == 5



def test_simple_ADTree_query_count2(data_small_3):
    dataset, column_values = data_small_3
    adtree = ADTree(dataset, column_values)

    assert adtree.query_count({}) == 16
    assert adtree.query_count({0: 1}) == 6
    assert adtree.query_count({1: 3}) == 5
    assert adtree.query_count({0: 2, 1: 3}) == 3
    assert adtree.query_count({0: 2, 1: 2, 2: 1}) == 1



def test_simple_ADTree_query(data_small_1):
    dataset, column_values = data_small_1
    adtree = ADTree(dataset, column_values)

    assert adtree.query({}) == 1 / 1

    assert adtree.query({0: 1}) == 1 / 8
    assert adtree.query({0: 2}) == 3 / 8
    assert adtree.query({0: 3}) == 4 / 8

    assert adtree.query({0: 1, 1: 1}) == 0 / 8
    assert adtree.query({0: 1, 1: 2}) == 1 / 8

    assert adtree.query({0: 2, 1: 1}) == 2 / 8
    assert adtree.query({0: 2, 1: 2}) == 1 / 8

    assert adtree.query({0: 3, 1: 1}) == 1 / 8
    assert adtree.query({0: 3, 1: 2}) == 3 / 8

    assert adtree.query({1: 1}) == 3 / 8
    assert adtree.query({1: 2}) == 5 / 8

    assert adtree.query({1: 1}, given={0: 1}) == 0 / 1
    assert adtree.query({1: 2}, given={0: 1}) == 1 / 1

    assert adtree.query({1: 1}, given={0: 2}) == 2 / 3
    assert adtree.query({1: 2}, given={0: 2}) == 1 / 3

    assert adtree.query({1: 1}, given={0: 3}) == 1 / 4
    assert adtree.query({1: 2}, given={0: 3}) == 3 / 4



def test_making_pmf(data_small_2):
    dataset, column_values = data_small_2
    adtree = ADTree(dataset, column_values)

    pmf = adtree.make_pmf([0])
    assert sorted(list(pmf.items())) == [(1, 1 / 2), (2, 1 / 2)]

    pmf = adtree.make_pmf([0, 1])
    expected_keys = sorted(zip([1] * 4 + [2] * 4, [1, 2, 3, 4] * 2))
    expected_pmf = {k: 2 / 16 for k in expected_keys}
    assert sorted(list(pmf.probabilities.keys())) == expected_keys
    assert pmf.probabilities == expected_pmf



def test_simple_ADTree_structure(data_small_1):
    dataset, column_values = data_small_1
    adtree = ADTree(dataset, column_values)

    assert adtree.root is not None
    assert adtree.root.count == 8
    assert len(adtree.root.Vary_children) == 2

    vary1 = adtree.root.Vary_children[0]
    assertVaryNodeCorrect(vary1, 0, [1, 2, 3], 3, 3)

    child0 = vary1.AD_children[0]
    assertADNodeCorrect(child0, 0, 1, 1, 1)

    child1 = vary1.AD_children[1]
    assertADNodeCorrect(child1, 0, 2, 3, 1)

    child2 = vary1.AD_children[2]
    assert child2 is None

    vary2 = adtree.root.Vary_children[1]
    assertVaryNodeCorrect(vary2, 1, [1, 2], 2, 2)

    child0 = vary2.AD_children[0]
    assertADNodeCorrect(child0, 1, 1, 3, 0)

    child1 = vary2.AD_children[1]
    assert child1 is None

    vary3 = adtree.root.Vary_children[0].AD_children[0].Vary_children[0]
    assertVaryNodeCorrect(vary3, 1, [1, 2], 2, 2)
    assert vary3.AD_children == [None, None]

    vary4 = adtree.root.Vary_children[0].AD_children[1].Vary_children[0]
    assertVaryNodeCorrect(vary4, 1, [1, 2], 1, 2)

    child0 = vary4.AD_children[0]
    assert child0 is None

    child1 = vary4.AD_children[1]
    assertADNodeCorrect(child1, 1, 2, 1, 0)



def test_compare_g_tests__survey(ds_survey_5e2, adtree_survey_5e2_llta20):
    ds = ds_survey_5e2
    adtree = adtree_survey_5e2_llta20

    parameters = dict()
    parameters['ci_test_debug'] = 0
    parameters['ci_test_significance'] = 0.95
    parameters['ci_test_ad_tree_class'] = ADTree
    parameters['ci_test_ad_tree_leaf_list_threshold'] = 20
    parameters['ci_test_ad_tree_preloaded'] = adtree
    parameters['omega'] = ds.omega
    parameters['source_bayesian_network'] = ds.bayesiannetwork
    parameters['ci_test_dof_calculator_class'] = mbtk.math.DoFCalculators.StructuralDoF

    G_unoptimized = mbtk.math.G_test__unoptimized.G_test(ds.datasetmatrix, parameters)
    G_with_AD_tree = mbtk.math.G_test__with_AD_tree.G_test(ds.datasetmatrix, parameters)

    tests = [
        (0, 1, set()),
        (4, 3, set()),
        (4, 3, {1}),
        (5, 3, {1, 2}),
        (0, 1, {2, 3, 4, 5}),
    ]

    for test in tests:
        X, Y, Z = test

        G_unoptimized.conditionally_independent(X, Y, Z)
        G_with_AD_tree.conditionally_independent(X, Y, Z)

        assertEqualLastTests(G_unoptimized, G_with_AD_tree)



def test_compare_g_tests__alarm(ds_alarm_5e2, adtree_alarm_5e2_llta0):
    ds = ds_alarm_5e2
    adtree = adtree_alarm_5e2_llta0

    parameters = dict()
    parameters['ci_test_debug'] = 0
    parameters['ci_test_significance'] = 0.95
    parameters['ci_test_ad_tree_class'] = ADTree
    parameters['ci_test_ad_tree_leaf_list_threshold'] = 20
    parameters['ci_test_ad_tree_preloaded'] = adtree
    parameters['omega'] = ds.omega
    parameters['source_bayesian_network'] = ds.bayesiannetwork
    parameters['ci_test_dof_calculator_class'] = mbtk.math.DoFCalculators.StructuralDoF

    G_unoptimized = mbtk.math.G_test__unoptimized.G_test(ds.datasetmatrix, parameters)
    G_with_AD_tree = mbtk.math.G_test__with_AD_tree.G_test(ds.datasetmatrix, parameters)

    tests = [
        (0, 1, set()),
        (4, 3, set()),
        (4, 3, {1}),
        (5, 3, {1, 2}),
        (0, 1, {2, 3, 4, 5}),
        (1, 3, {28, 33}),
        (33, 3, {2, 28, 36})
    ]

    for test in tests:
        X, Y, Z = test

        G_unoptimized.conditionally_independent(X, Y, Z)
        G_with_AD_tree.conditionally_independent(X, Y, Z)

        assertEqualLastTests(G_unoptimized, G_with_AD_tree)



def test_making_pmf_larger_dataset(ds_survey_5e2, adtree_survey_5e2_llta20):
    ds = ds_survey_5e2
    adtree = adtree_survey_5e2_llta20

    assert_pmf_adtree_vs_datasetmatrix(ds, adtree, [0])
    assert_pmf_adtree_vs_datasetmatrix(ds, adtree, [1])
    assert_pmf_adtree_vs_datasetmatrix(ds, adtree, [2])
    assert_pmf_adtree_vs_datasetmatrix(ds, adtree, [3])
    assert_pmf_adtree_vs_datasetmatrix(ds, adtree, [4])
    assert_pmf_adtree_vs_datasetmatrix(ds, adtree, [5])
    assert_pmf_adtree_vs_datasetmatrix(ds, adtree, [1, 2])
    assert_pmf_adtree_vs_datasetmatrix(ds, adtree, [2, 3])
    assert_pmf_adtree_vs_datasetmatrix(ds, adtree, [3, 4])
    assert_pmf_adtree_vs_datasetmatrix(ds, adtree, [4, 5])
    assert_pmf_adtree_vs_datasetmatrix(ds, adtree, [0, 2, 3])
    assert_pmf_adtree_vs_datasetmatrix(ds, adtree, [1, 3, 4])
    assert_pmf_adtree_vs_datasetmatrix(ds, adtree, [1, 4, 5])
    assert_pmf_adtree_vs_datasetmatrix(ds, adtree, [0, 1, 2, 3, 4, 5])



def test_simple_ADTree_structure_2(data_small_2):
    dataset, column_values = data_small_2
    adtree = ADTree(dataset, column_values)

    assert adtree.root is not None
    assert adtree.root.count == 16
    assert len(adtree.root.Vary_children) == 3

    vary1 = adtree.root.Vary_children[0]
    assertVaryNodeCorrect(vary1, 0, [1, 2], 1, 2)

    child11 = vary1.AD_children[0]
    assert child11 is None

    child12 = vary1.AD_children[1]
    assertADNodeCorrect(child12, 0, 2, 8, 2)

    vary121 = child12.Vary_children[0]
    assertVaryNodeCorrect(vary121, 1, [1, 2, 3, 4], 1, 4)

    child1211 = vary121.AD_children[0]
    assert child1211 is None

    child1212 = vary121.AD_children[1]
    assertADNodeCorrect(child1212, 1, 2, 2, 1)
    vary12121 = child1212.Vary_children[0]
    assertVaryNodeCorrect(vary12121, 2, [1, 2], 1, 2)
    child121211 = vary12121.AD_children[0]
    assert child121211 is None
    child121212 = vary12121.AD_children[1]
    assertADNodeCorrect(child121212, 2, 2, 1, 0)

    child1213 = vary121.AD_children[2]
    assertADNodeCorrect(child1213, 1, 3, 2, 1)
    vary12131 = child1213.Vary_children[0]
    assertVaryNodeCorrect(vary12131, 2, [1, 2], 1, 2)
    child121311 = vary12131.AD_children[0]
    assert child121311 is None
    child121312 = vary12131.AD_children[1]
    assertADNodeCorrect(child121312, 2, 2, 1, 0)

    child1214 = vary121.AD_children[3]
    assertADNodeCorrect(child1214, 1, 4, 2, 1)
    vary12141 = child1214.Vary_children[0]
    assertVaryNodeCorrect(vary12141, 2, [1, 2], 1, 2)
    child121411 = vary12141.AD_children[0]
    assert child121411 is None
    child121412 = vary12141.AD_children[1]
    assertADNodeCorrect(child121412, 2, 2, 1, 0)

    vary122 = child12.Vary_children[1]
    assertVaryNodeCorrect(vary122, 2, [1, 2], 1, 2)
    child1221 = vary122.AD_children[0]
    assert child1221 is None
    child1222 = vary122.AD_children[1]
    assertADNodeCorrect(child1222, 2, 2, 4, 0)

    vary2 = adtree.root.Vary_children[1]
    assertVaryNodeCorrect(vary2, 1, [1, 2, 3, 4], 1, 4)
    child21 = vary2.AD_children[0]
    assert child21 is None
    child22 = vary2.AD_children[1]
    assertADNodeCorrect(child22, 1, 2, 4, 1)
    vary221 = child22.Vary_children[0]
    assertVaryNodeCorrect(vary221, 2, [1, 2], 1, 2)
    child2211 = vary221.AD_children[0]
    assert child2211 is None
    child2212 = vary221.AD_children[1]
    assertADNodeCorrect(child2212, 2, 2, 2, 0)

    child23 = vary2.AD_children[2]
    assertADNodeCorrect(child23, 1, 3, 4, 1)
    vary231 = child23.Vary_children[0]
    assertVaryNodeCorrect(vary231, 2, [1, 2], 1, 2)
    child2311 = vary231.AD_children[0]
    assert child2311 is None
    child2312 = vary231.AD_children[1]
    assertADNodeCorrect(child2312, 2, 2, 2, 0)

    child24 = vary2.AD_children[3]
    assertADNodeCorrect(child24, 1, 4, 4, 1)
    vary241 = child24.Vary_children[0]
    assertVaryNodeCorrect(vary241, 2, [1, 2], 1, 2)
    child2411 = vary241.AD_children[0]
    assert child2411 is None
    child2412 = vary241.AD_children[1]
    assertADNodeCorrect(child2412, 2, 2, 2, 0)

    vary3 = adtree.root.Vary_children[2]
    assertVaryNodeCorrect(vary3, 2, [1, 2], 1, 2)
    child31 = vary3.AD_children[0]
    assert child31 is None
    child32 = vary3.AD_children[1]
    assertADNodeCorrect(child32, 2, 2, 8, 0)



def assertEqualLastTests(Gtest_left, Gtest_right):
    result_left = Gtest_left.ci_test_results[-1:]
    result_right = Gtest_left.ci_test_results[-1:]
    assert result_left == result_right



def assert_pmf_adtree_vs_datasetmatrix(ds, adtree, variables):
    dm = ds.datasetmatrix
    if isinstance(variables, int):
        variables = [variables]

    calculated_pmf = adtree.make_pmf(variables)
    calculated_pmf.remove_zeros()

    variables = dm.get_variables('X', variables)
    expected_pmf = PMF(variables)

    assert expected_pmf.probabilities == calculated_pmf.probabilities



def assertVaryNodeCorrect(vary, column_index, values, mcv, children_count):
    assert vary is not None
    assert vary.column_index == column_index
    assert vary.values == values
    assert vary.most_common_value == mcv
    assert len(vary.AD_children) == children_count



def assertADNodeCorrect(adNode, column_index, value, count, children_count):
    assert adNode is not None
    assert adNode.column_index == column_index
    assert adNode.value == value
    assert adNode.count == count
    assert len(adNode.Vary_children) == children_count



def assert_cpmf_adtree_vs_dm(dm, adtree, cd_vars, cn_vars):
    if isinstance(cd_vars, int):
        cd_vars = [cd_vars]

    if isinstance(cn_vars, int):
        cn_vars = [cn_vars]

    cd_vars = dm.get_variables('X', cd_vars)
    cn_vars = dm.get_variables('X', cn_vars)
    expected_cpmf = CPMF(cd_vars, cn_vars)

    calculated_cpmf = adtree.make_cpmf(cd_vars, cn_vars)

    assert expected_cpmf == calculated_cpmf
