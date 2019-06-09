import unittest
import numpy
import scipy

from mbff.dataset.DatasetMatrix import DatasetMatrix
from mbff.structures.ADTree import ADTree, ADNode, VaryNode

from mbff_tests.TestBase import TestBase

class TestADTree(TestBase):

    def default_small_dataset(self):
        dataset = scipy.sparse.csr_matrix(numpy.array([
            [1, 2, 3, 2, 2, 3, 3, 3],
            [2, 1, 1, 2, 1, 2, 2, 2]]).transpose())

        column_values = {
                0: [1, 2, 3],
                1: [1, 2]}
        return (dataset, column_values)


    def default_small_dataset_2(self):
        dataset = scipy.sparse.csr_matrix(numpy.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
            [1, 1, 2, 2, 3, 3, 4, 4, 1, 1, 2, 2, 3, 3, 4, 4],
            [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]]).transpose())

        column_values = {
                0: [1, 2],
                1: [1, 2, 3, 4],
                2: [1, 2]}
        return (dataset, column_values)


    def default_small_dataset_3(self):
        dataset = scipy.sparse.csr_matrix(numpy.array([
            [1, 1, 2, 1, 1, 2, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2],
            [2, 2, 1, 3, 3, 2, 2, 1, 1, 2, 3, 3, 1, 1, 2, 3],
            [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]]).transpose())

        column_values = {
                0: [1, 2],
                1: [1, 2, 3, 4],
                2: [1, 2]}
        return (dataset, column_values)


    def assertVaryNodeCorrect(self, vary, column_index, values, mcv, children_count):
        self.assertIsNotNone(vary)
        self.assertEqual(column_index, vary.column_index)
        self.assertEqual(values, vary.values)
        self.assertEqual(mcv, vary.most_common_value)
        self.assertEqual(children_count, len(vary.AD_children))


    def assertADNodeCorrect(self, adNode, column_index, value, count, children_count):
        self.assertIsNotNone(adNode)
        self.assertEqual(column_index, adNode.column_index)
        self.assertEqual(value, adNode.value)
        self.assertEqual(count, adNode.count)
        self.assertEqual(children_count, len(adNode.Vary_children))


    def test_simple_ADTree_query_count(self):
        (dataset, column_values) = self.default_small_dataset()
        adtree = ADTree(dataset, column_values)
        
        self.assertEqual(8, adtree.query_count({}))
        self.assertEqual(1, adtree.query_count({0:1}))
        self.assertEqual(3, adtree.query_count({0:2}))
        self.assertEqual(4, adtree.query_count({0:3}))
        
        self.assertEqual(0, adtree.query_count({0:1, 1:1}))
        self.assertEqual(1, adtree.query_count({0:1, 1:2}))

        self.assertEqual(2, adtree.query_count({0:2, 1:1}))
        self.assertEqual(1, adtree.query_count({0:2, 1:2}))

        self.assertEqual(1, adtree.query_count({0:3, 1:1}))
        self.assertEqual(3, adtree.query_count({0:3, 1:2}))

        self.assertEqual(3, adtree.query_count({1:1}))
        self.assertEqual(5, adtree.query_count({1:2}))


    def test_simple_ADTree_query_count2(self):
        (dataset, column_values) = self.default_small_dataset_3()
        adtree = ADTree(dataset, column_values)

        self.assertEqual(16, adtree.query_count({}))
        self.assertEqual(6, adtree.query_count({0:1}))
        self.assertEqual(5, adtree.query_count({1:3}))
        self.assertEqual(3, adtree.query_count({0:2, 1:3}))
        self.assertEqual(1, adtree.query_count({0:2, 1:2, 2:1}))


    def test_simple_ADTree_query(self):
        (dataset, column_values) = self.default_small_dataset()
        adtree = ADTree(dataset, column_values)
        
        self.assertEqual(1/1, adtree.query({}))

        self.assertEqual(1/8, adtree.query({0:1}))
        self.assertEqual(3/8, adtree.query({0:2}))
        self.assertEqual(4/8, adtree.query({0:3}))

        self.assertEqual(0/8, adtree.query({0:1, 1:1}))
        self.assertEqual(1/8, adtree.query({0:1, 1:2}))

        self.assertEqual(2/8, adtree.query({0:2, 1:1}))
        self.assertEqual(1/8, adtree.query({0:2, 1:2}))

        self.assertEqual(1/8, adtree.query({0:3, 1:1}))
        self.assertEqual(3/8, adtree.query({0:3, 1:2}))

        self.assertEqual(3/8, adtree.query({1:1}))
        self.assertEqual(5/8, adtree.query({1:2}))

        self.assertEqual(0/1, adtree.query({1:1}, given={0:1}))
        self.assertEqual(1/1, adtree.query({1:2}, given={0:1}))

        self.assertEqual(2/3, adtree.query({1:1}, given={0:2}))
        self.assertEqual(1/3, adtree.query({1:2}, given={0:2}))

        self.assertEqual(1/4, adtree.query({1:1}, given={0:3}))
        self.assertEqual(3/4, adtree.query({1:2}, given={0:3}))


    def test_simple_ADTree_structure(self):
        (dataset, column_values) = self.default_small_dataset()
        adtree = ADTree(dataset, column_values)

        self.assertIsNotNone(adtree.root)
        self.assertEqual(8, adtree.root.count)
        self.assertEqual(2, len(adtree.root.Vary_children))

        vary1 = adtree.root.Vary_children[0]
        self.assertVaryNodeCorrect(vary1, 0, [1, 2, 3], 3, 3)

        child0 = vary1.AD_children[0]
        self.assertADNodeCorrect(child0, 0, 1, 1, 1)

        child1 = vary1.AD_children[1]
        self.assertADNodeCorrect(child1, 0, 2, 3, 1)

        child2 = vary1.AD_children[2]
        self.assertIsNone(child2)

        vary2 = adtree.root.Vary_children[1]
        self.assertVaryNodeCorrect(vary2, 1, [1, 2], 2, 2)

        child0 = vary2.AD_children[0]
        self.assertADNodeCorrect(child0, 1, 1, 3, 0)

        child1 = vary2.AD_children[1]
        self.assertIsNone(child1)

        vary3 = adtree.root.Vary_children[0].AD_children[0].Vary_children[0]
        self.assertVaryNodeCorrect(vary3, 1, [1, 2], 2, 2)
        self.assertEqual([None, None], vary3.AD_children)

        vary4 = adtree.root.Vary_children[0].AD_children[1].Vary_children[0]
        self.assertVaryNodeCorrect(vary4, 1, [1, 2], 1, 2)

        child0 = vary4.AD_children[0]
        self.assertIsNone(child0)

        child1 = vary4.AD_children[1]
        self.assertADNodeCorrect(child1, 1, 2, 1, 0)


    def test_simple_ADTree2_structure(self):
        (dataset, column_values) = self.default_small_dataset_2()
        adtree = ADTree(dataset, column_values)

        self.assertIsNotNone(adtree.root)
        self.assertEqual(16, adtree.root.count)
        self.assertEqual(3, len(adtree.root.Vary_children))

        vary1 = adtree.root.Vary_children[0]
        self.assertVaryNodeCorrect(vary1, 0, [1, 2], 1, 2)

        child11 = vary1.AD_children[0]
        self.assertIsNone(child11)

        child12 = vary1.AD_children[1]
        self.assertADNodeCorrect(child12, 0, 2, 8, 2) 

        vary121 = child12.Vary_children[0]
        self.assertVaryNodeCorrect(vary121, 1, [1, 2, 3, 4], 1, 4)

        child1211 = vary121.AD_children[0]
        self.assertIsNone(child1211)

        child1212 = vary121.AD_children[1]
        self.assertADNodeCorrect(child1212, 1, 2, 2, 1)
        vary12121 = child1212.Vary_children[0]
        self.assertVaryNodeCorrect(vary12121, 2, [1, 2], 1, 2)
        child121211 = vary12121.AD_children[0]
        self.assertIsNone(child121211)
        child121212 = vary12121.AD_children[1]
        self.assertADNodeCorrect(child121212, 2, 2, 1, 0)

        child1213 = vary121.AD_children[2]
        self.assertADNodeCorrect(child1213, 1, 3, 2, 1)
        vary12131 = child1213.Vary_children[0]
        self.assertVaryNodeCorrect(vary12131, 2, [1, 2], 1, 2)
        child121311 = vary12131.AD_children[0]
        self.assertIsNone(child121311)
        child121312 = vary12131.AD_children[1]
        self.assertADNodeCorrect(child121312, 2, 2, 1, 0)

        child1214 = vary121.AD_children[3]
        self.assertADNodeCorrect(child1214, 1, 4, 2, 1)
        vary12141 = child1214.Vary_children[0]
        self.assertVaryNodeCorrect(vary12141, 2, [1, 2], 1, 2)
        child121411 = vary12141.AD_children[0]
        self.assertIsNone(child121411)
        child121412 = vary12141.AD_children[1]
        self.assertADNodeCorrect(child121412, 2, 2, 1, 0)

        vary122 = child12.Vary_children[1]
        self.assertVaryNodeCorrect(vary122, 2, [1, 2], 1, 2)
        child1221 = vary122.AD_children[0]
        self.assertIsNone(child1221)
        child1222 = vary122.AD_children[1]
        self.assertADNodeCorrect(child1222, 2, 2, 4, 0)

        vary2 = adtree.root.Vary_children[1]
        self.assertVaryNodeCorrect(vary2, 1, [1, 2, 3, 4], 1, 4)
        child21 = vary2.AD_children[0]
        self.assertIsNone(child21)
        child22 = vary2.AD_children[1]
        self.assertADNodeCorrect(child22, 1, 2, 4, 1)
        vary221 = child22.Vary_children[0]
        self.assertVaryNodeCorrect(vary221, 2, [1, 2], 1, 2)
        child2211 = vary221.AD_children[0]
        self.assertIsNone(child2211)
        child2212 = vary221.AD_children[1]
        self.assertADNodeCorrect(child2212, 2, 2, 2, 0)

        child23 = vary2.AD_children[2]
        self.assertADNodeCorrect(child23, 1, 3, 4, 1)
        vary231 = child23.Vary_children[0]
        self.assertVaryNodeCorrect(vary231, 2, [1, 2], 1, 2)
        child2311 = vary231.AD_children[0]
        self.assertIsNone(child2311)
        child2312 = vary231.AD_children[1]
        self.assertADNodeCorrect(child2312, 2, 2, 2, 0)
        
        child24 = vary2.AD_children[3]
        self.assertADNodeCorrect(child24, 1, 4, 4, 1)
        vary241 = child24.Vary_children[0]
        self.assertVaryNodeCorrect(vary241, 2, [1, 2], 1, 2)
        child2411 = vary241.AD_children[0]
        self.assertIsNone(child2411)
        child2412 = vary241.AD_children[1]
        self.assertADNodeCorrect(child2412, 2, 2, 2, 0)

        vary3 = adtree.root.Vary_children[2]
        self.assertVaryNodeCorrect(vary3, 2, [1, 2], 1, 2)
        child31 = vary3.AD_children[0]
        self.assertIsNone(child31)
        child32 = vary3.AD_children[1]
        self.assertADNodeCorrect(child32, 2, 2, 8, 0)
