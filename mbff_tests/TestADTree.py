import unittest
import numpy
import scipy

from mbff.dataset.DatasetMatrix import DatasetMatrix
from mbff.structures.ADTree import ADTree

from mbff_tests.TestBase import TestBase

class TestADTree(TestBase):

    def test_simple_ADTree(self):
        dataset = self.default_small_dataset()
        dataset.finalize()
        adtree = ADTree(dataset)

        self.assertIsNotNone(adtree.Root)
        self.assertEqual(8, adtree.Root.count)
        self.assertEqual(2, len(adtree.Root.Vary_children))

        vary1 = adtree.Root.Vary_children[0]
        self.assertEqual(0, vary1.variableID)
        self.assertEqual([1, 2, 3], vary1.values)
        self.assertEqual(3, vary1.most_common_value)
        self.assertEqual(3, len(vary1.AD_children))

        child0 = vary1.AD_children[0]
        self.assertIsNotNone(child0)
        self.assertEqual(1, child0.value)
        self.assertEqual(1, child0.count)
        self.assertEqual(1, len(child0.Vary_children))

        child1 = vary1.AD_children[1]
        self.assertIsNotNone(child1)
        self.assertEqual(2, child1.value)
        self.assertEqual(3, child1.count)
        self.assertEqual(1, len(child1.Vary_children))

        child2 = vary1.AD_children[2]
        self.assertIsNone(child2)

        vary2 = adtree.Root.Vary_children[1]
        self.assertEqual(1, vary1.variableID)
        self.assertEqual([1, 2], vary1.values)
        self.assertEqual(2, vary1.most_common_value)
        self.assertEqual(2, len(vary1.AD_children))

        child0 = vary2.AD_children[0]
        self.assertIsNotNone(child0)
        self.assertEqual(1, child0.value)
        self.assertEqual(3, child0.count)
        self.assertEqual(0, len(child0.Vary_children))

        child1 = vary2.AD_children[1]
        self.assertIsNone(child1)

        vary3 = adtree.Root.Vary_children[0].AD_children[0].Vary_children[0]
        self.assertEqual(2, vary3.variableID)
        self.assertEqual(2, vary3.most_common_value)
        self.assertEqual([None, None], vary.AD_children)

        vary4 = adtree.Root.Vary_children[0].AD_children[1].Vary_children[0]
        self.assertEqual(2, vary4.variableID)
        self.assertEqual(1, vary4.most_common_value)
        self.assertEqual(2, len(vary.AD_children))

        child0 = vary4.AD_children[0]
        self.assertIsNone(child0)

        child1 = vary4.AD_children[1]
        self.assertIsNotNone(child1)
        self.assertEqual(2, child1.value)
        self.assertEqual(1, child1.count)
        self.assertEqual(0, len(child1.Vary_children))


    def default_small_dataset(self):
        datasetmatrix = DatasetMatrix('test_small')
        datasetmatrix.X = scipy.sparse.csr_matrix(numpy.array([
            [1, 2, 2, 2, 3, 3, 3, 3],
            [2, 1, 1, 2, 1, 2, 2, 2]]).transpose())
        datasetmatrix.Y = scipy.sparse.csr_matrix(numpy.empty((8, 2)))
        datasetmatrix.row_labels = ['row{}'.format(i) for i in range(0, 8)]
        datasetmatrix.column_labels_X = ['a1', 'a2']
        datasetmatrix.column_labels_Y = []

        return datasetmatrix
