from pprint import pprint
import numpy
import unittest

from pathlib import Path

import mbff.utilities.functions as util

from mbff.dataset.DatasetMatrix import DatasetMatrix
from mbff.math.Variable import Variable, JointVariables, Omega
from mbff.math.PMF import PMF, CPMF
from mbff.dataset.sources.SampledBayesianNetworkDatasetSource import SampledBayesianNetworkDatasetSource
from mbff.math.G_test import *

from mbff_tests.TestBase import TestBase

import scipy.stats

@unittest.skipIf(TestBase.tag_excluded('sampling'), 'Sampling tests excluded')
class TestGStat(TestBase):

    ClassIsSetUp = False
    DatasetMatrix = None
    Omega = None


    def setUp(self):
        if not TestGStat.ClassIsSetUp:
            self.prepare_datasetmatrix()


    def prepare_datasetmatrix(self):
        dataset_folder = Path('testfiles', 'tmp', 'test_gstat_dm')
        configuration = {}
        configuration['sourcepath'] = Path('testfiles', 'bif_files', 'survey.bif')
        configuration['sample_count'] = int(2e4)
        configuration['random_seed'] = 42*42
        configuration['values_as_indices'] = False
        configuration['objectives'] = ['TRN']
        try:
            TestGStat.DatasetMatrix = DatasetMatrix('g_stat_dataset')
            TestGStat.DatasetMatrix.load(dataset_folder)
        except:
            bayesian_network = util.read_bif_file(configuration['sourcepath'])
            bayesian_network.finalize()
            sbnds = SampledBayesianNetworkDatasetSource(configuration)
            sbnds.reset_random_seed = True
            TestGStat.DatasetMatrix = sbnds.create_dataset_matrix('g_stat_dataset')
            TestGStat.DatasetMatrix.finalize()
            TestGStat.DatasetMatrix.save(dataset_folder)
            print('Dataset rebuilt.')
        TestGStat.Omega = Omega(configuration['sample_count'])
        TestGStat.ClassIsSetUp = True


    def test_G_value(self):
        AGE = Variable(self.DatasetMatrix.get_column_by_label('X', 'AGE'), 'AGE')
        SEX = Variable(self.DatasetMatrix.get_column_by_label('X', 'SEX'), 'SEX')
        EDU = Variable(self.DatasetMatrix.get_column_by_label('X', 'EDU'), 'EDU')
        OCC = Variable(self.DatasetMatrix.get_column_by_label('X', 'OCC'), 'OCC')
        R = Variable(self.DatasetMatrix.get_column_by_label('X', 'R'), 'R')
        TRN = Variable(self.DatasetMatrix.get_column_by_label('Y', 'TRN'), 'TRN')


        self.print_pmf(EDU)
        print()
        self.print_pmf(OCC)
        print()
        self.print_pmf(R)
        print()
        self.print_cpmf(OCC, EDU)
        print()
        self.print_cpmf(R, EDU)
        print()
        self.print_pmf(JointVariables(OCC, R))
        print()
        self.print_cpmf(JointVariables(OCC, R), EDU)
        print()

        self.analyze_variable_g_values(OCC, R, self.Omega)
        self.analyze_variable_g_values(OCC, R, EDU)


    def print_pmf(self, var):
        pmf = PMF(var)
        print('PMF of {}'.format(var.name))
        for k in sorted(pmf.probabilities.keys()):
            print('  {}:  {}'.format(k, pmf.probabilities[k]))


    def print_cpmf(self, var, cvar):
        cpmf = CPMF(var, cvar)
        print('PMF of {} given {}'.format(var.name, cvar.name))
        for ck in sorted(cpmf.conditional_probabilities.keys()):
            print('  {}:'.format(ck))
            pmf = cpmf.given(ck)
            for k in sorted(pmf.probabilities.keys()):
                print('    {}:  {}'.format(k, pmf.probabilities[k]))
            



    def analyze_variable_g_values(self, X, Y, Z):
        chi = scipy.stats.chi2
        (Gvalue, cMI) = G_value__unoptimized_with_cMI(X, Y, Z)
        DF = calculate_degrees_of_freedom(X, Y)
        print('-----------------------------')
        print('{} vs {} given {}'.format(X.name, Y.name, Z.name))
        print('CMI\t', cMI) 
        print('DF\t', DF)
        print('G-value\t', Gvalue)
        print('p\t', chi.cdf(Gvalue, DF))
        print()



    @unittest.skip
    def test_cmi(self):
        AGE = Variable(self.DatasetMatrix.get_column_by_label('X', 'AGE'))
        SEX = Variable(self.DatasetMatrix.get_column_by_label('X', 'SEX'))
        EDU = Variable(self.DatasetMatrix.get_column_by_label('X', 'EDU'))
        OCC = Variable(self.DatasetMatrix.get_column_by_label('X', 'OCC'))
        R = Variable(self.DatasetMatrix.get_column_by_label('X', 'R'))
        TRN = Variable(self.DatasetMatrix.get_column_by_label('Y', 'TRN'))

        X = AGE
        Y = SEX
        Z = self.Omega
        (PrXYcZ, PrXcZ, PrYcZ, PrZ) = calculate_pmf_for_cmi(X, Y, Z)
        cMI = conditional_mutual_information(PrXYcZ, PrXcZ, PrYcZ, PrZ, base=2)
        self.assertLess(cMI, 2e-6)

        X = JointVariables(AGE, SEX)
        Y = EDU
        Z = self.Omega
        (PrXYcZ, PrXcZ, PrYcZ, PrZ) = calculate_pmf_for_cmi(X, Y, Z)
        cMI = conditional_mutual_information(PrXYcZ, PrXcZ, PrYcZ, PrZ, base=2)
        self.assertGreater(cMI, 0.025)

        X = AGE
        Y = OCC
        Z = self.Omega
        (PrXYcZ, PrXcZ, PrYcZ, PrZ) = calculate_pmf_for_cmi(X, Y, Z)
        cMI = conditional_mutual_information(PrXYcZ, PrXcZ, PrYcZ, PrZ, base=2)
        self.assertGreater(cMI, 0.0001)

        X = AGE
        Y = OCC
        Z = EDU
        (PrXYcZ, PrXcZ, PrYcZ, PrZ) = calculate_pmf_for_cmi(X, Y, Z)
        cMI = conditional_mutual_information(PrXYcZ, PrXcZ, PrYcZ, PrZ, base=2)
        self.assertLess(cMI, 2e-5)


    def print_var_pmf(self, var, pmf, name):
        try:
            print('{}.variables'.format(name), var.variables)
        except:
            pass
        print('{}.values'.format(name), var.values)
        try:
            print('Pr{}.probabilities'.format(name))
            pprint(pmf.probabilities)
            print('None')
        except:
            pass
        try:
            print('Pr{}.conditional_probabilities'.format(name))
            pprint(pmf.conditional_probabilities)
        except:
            print('None')
            pass

