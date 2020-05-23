import math
import pickle

from mbff.math.CITestResult import CITestResult
from mbff.math.PMF import PMF

import mbff.math.G_test__unoptimized
from scipy.stats import chi2


class G_test(mbff.math.G_test__unoptimized.G_test):

    def __init__(self, datasetmatrix, parameters):
        super().__init__(datasetmatrix, parameters)

        if self.DoF_calculator.requires_cpmfs:
            raise ValueError("Cannot use a DoF calculator that requires CPMFs.")

        self.JHT = dict()
        self.JHT_reads = 0
        self.JHT_misses = 0
        self.prepare_JHT()


    def prepare_JHT(self):
        preloaded_JHT = self.parameters.get('ci_test_jht_preloaded', None)
        if preloaded_JHT is not None:
            self.JHT = preloaded_JHT
            self.JHT_reads = self.JHT['reads']
            self.JHT_misses = self.JHT['misses']
            print('using preloaded JHT')
            return

        jht_load_path = self.parameters.get('ci_test_jht_path__load', None)
        if jht_load_path is not None and jht_load_path.exists():
            with jht_load_path.open('rb') as f:
                self.JHT = pickle.load(f)
            self.JHT_reads = self.JHT['reads']
            self.JHT_misses = self.JHT['misses']


    def G_test_conditionally_independent(self, X, Y, Z):
        result = CITestResult()
        result.start_timing()

        self.DoF_calculator.set_context_variables(X, Y, Z)

        # In the other two implementations of the G-test (unoptimized and with
        # AD-tree), the DoF could be calculated before calculating G. But this
        # implementation of the G-test (using dcMI) requires the
        # CachingStructuralDoF calculator, which must have received the joint
        # probability distribution of X, Y and Z sometime in the past. The only
        # time it would have been possible to pass this joint distribution to
        # the CachingStructuralDoF calculator would be inside the method
        # self.G_value(), and even then, only when there is a miss in the JHT.
        # Therefore self.G_value() must run before the DoF calculator, to be
        # sure that it has received the required joint distributions beforehand.
        G = self.G_value(X, Y, Z)

        DoF = self.DoF_calculator.calculate_DoF(X, Y, Z)

        if not self.sufficient_samples(DoF):
            result.end_timing()
            result.index = self.ci_test_counter + 1
            result.set_insufficient_samples()
            result.set_variables(X, Y, Z)
            result.extra_info = ' DoF {}'.format(DoF)
            return result

        p = chi2.cdf(G, DoF)
        independent = None
        if p < self.significance:
            independent = True
        else:
            independent = False

        result.end_timing()
        result.index = self.ci_test_counter + 1
        result.set_independent(independent, self.significance)
        result.set_variables(X, Y, Z)
        result.set_statistic('G', G, dict())
        result.set_distribution('chi2', p, {'DoF': DoF})

        result.extra_info = ' DoF {}'.format(DoF)

        return result


    def G_value(self, X, Y, Z):
        HXYZ = self.get_joint_entropy_term(X, Y, Z)
        HXZ = self.get_joint_entropy_term(X, Z)
        HYZ = self.get_joint_entropy_term(Y, Z)
        HZ = self.get_joint_entropy_term(Z)
        cMI = HYZ + HXZ - HXYZ - HZ
        return 2 * self.N * cMI


    def get_joint_entropy_term(self, *variables):
        jht_key = self.create_flat_variable_set(*variables)
        if len(jht_key) == 0:
            return 0

        self.JHT_reads += 1

        try:
            H = self.JHT[jht_key]
        except KeyError:
            self.JHT_misses += 1
            joint_variables = self.datasetmatrix.get_variables('X', jht_key)
            pmf = PMF(joint_variables)
            H = - pmf.expected_value(lambda v, p: math.log(p))
            self.JHT[jht_key] = H
            if self.DoF_calculator.requires_pmfs:
                self.DoF_calculator.set_context_pmfs(pmf, None, None, None)

        return H


    def end(self):
        super().end()

        jht_save_path = self.parameters.get('ci_test_jht_path__save', None)
        if jht_save_path is not None:
            self.JHT['reads'] = self.JHT_reads
            self.JHT['misses'] = self.JHT_misses
            with jht_save_path.open('wb') as f:
                pickle.dump(self.JHT, f)
