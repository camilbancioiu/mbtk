from mbff.structures.DynamicADTree import DynamicADTree
from tests.test_ADTree import assert_pmf_adtree_vs_datasetmatrix


def test_simple_DynamicADTree_query_count(data_small_1):
    dataset, column_values = data_small_1
    adtree = DynamicADTree(dataset, column_values)

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



def test_simple_DynamicADTree_query_count2(data_small_3):
    dataset, column_values = data_small_3
    adtree = DynamicADTree(dataset, column_values)

    assert adtree.query_count({}) == 16
    assert adtree.query_count({0: 1}) == 6
    assert adtree.query_count({1: 3}) == 5
    assert adtree.query_count({0: 2, 1: 3}) == 3
    assert adtree.query_count({0: 2, 1: 2, 2: 1}) == 1



def test_simple_DynamicADTree_query(data_small_1):
    dataset, column_values = data_small_1
    adtree = DynamicADTree(dataset, column_values)

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
    adtree = DynamicADTree(dataset, column_values)

    pmf = adtree.make_pmf([0])
    assert sorted(list(pmf.items())) == [(1, 1 / 2), (2, 1 / 2)]

    pmf = adtree.make_pmf([0, 1])
    expected_keys = sorted(zip([1] * 4 + [2] * 4, [1, 2, 3, 4] * 2))
    expected_pmf = {k: 2 / 16 for k in expected_keys}
    assert sorted(list(pmf.probabilities.keys())) == expected_keys
    assert pmf.probabilities == expected_pmf



def test_making_pmf_larger_dataset(ds_survey_5e2):
    ds = ds_survey_5e2
    matrix = ds.datasetmatrix.X
    column_values = ds.datasetmatrix.get_values_per_column('X')
    adtree = DynamicADTree(matrix, column_values)

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
