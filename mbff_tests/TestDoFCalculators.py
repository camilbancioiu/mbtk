import itertools

from mbff.math.PMF import PMF

from mbff.math.DoFCalculators import CachedStructuralDoF

from mbff_tests.TestBase import TestBase


class TestDoFCalculators(TestBase):

    def test_CachedStructuralDoF__computing_for_pmf__single_var(self):
        raise NotImplementedError()


    def test_CachedStructuralDoF__computing_for_pmf__two_vars(self):
        raise NotImplementedError()


    def test_CachedStructuralDoF__computing_for_pmf__three_vars(self):
        column_values = [
            [1, 2, 3, 4],
            [1, 2],
            [1, 2, 3]
        ]

        keys = list(itertools.product(*column_values))

        print()
        pmf = PMF(None)
        for key in keys:
            pmf.probabilities[key] = 1

        self.assertEqual(24, len(pmf))

        expected_dofs = {
            (0, 1): 9,
            (1, 0): 9,
            (0, 2): 12,
            (2, 0): 12,
            (1, 2): 8,
            (2, 1): 8
        }

        csdof = CachedStructuralDoF(None)
        calculated_dofs = csdof.calculate_pairwise_DoFs(pmf, len(column_values))

        self.assertEqual(expected_dofs, calculated_dofs)


    def test_CachedStructuralDoF__computing_for_pmf__four_vars(self):
        raise NotImplementedError()
