import math
import pickle

from mbff.math.CITestResult import CITestResult
from mbff.math.PMF import PMF

import mbff.math.G_test__unoptimized
from scipy.stats import chi2


class G_test(mbff.math.G_test__unoptimized.G_test):

    def __init__(self, datasetmatrix, parameters):
        super().__init__(datasetmatrix, parameters)

        self.JHT = dict()
        self.JHT_reads = 0
        self.JHT_hits = 0
        self.prepare_JHT()


    def prepare_JHT(self):
        jht_load_path = self.parameters.get('ci_test_jht_path__load', None)
        if jht_load_path is not None and jht_load_path.exists():
            if self.debug >= 1: print('Loading the JHT from {} ...'.format(jht_load_path))
            with jht_load_path.open('rb') as f:
                self.JHT = pickle.load(f)
            self.JHT_reads = self.JHT['reads']
            self.JHT_hits = self.JHT['hits']
            if self.debug >= 1: print('JHT loaded.')


    def G_test_conditionally_independent(self, X, Y, Z):
        result = CITestResult()
        result.start_timing()

        G = self.G_value(X, Y, Z)
        DoF = self.calculate_degrees_of_freedom(None, None, None, None, X, Y, Z)

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
            self.JHT_hits += 1
            if self.debug >= 2: print('\tJMI cache hit: found H={:8.6f} for {}'.format(H, jht_key))
        except KeyError:
            joint_variables = self.datasetmatrix.get_variables('X', jht_key)
            pmf = PMF(joint_variables)
            H = - pmf.expected_value(lambda v, p: math.log(p))
            self.JHT[jht_key] = H
            if self.debug >= 2: print('\tJHT miss and update: store H={:8.6f} for {}'.format(H, jht_key))
            self.cache_pmf_info(jht_key, pmf)

        return H


    def end(self):
        super().end()

        jht_save_path = self.parameters.get('ci_test_jht_path__save', None)
        if jht_save_path is not None:
            self.JHT['reads'] = self.JHT_reads
            self.JHT['hits'] = self.JHT_hits
            with jht_save_path.open('wb') as f:
                pickle.dump(self.JHT, f)
        if self.debug >= 1: print('JHT has been saved to {}'.format(jht_save_path))
