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
            self.prepare_datasetmatrices()


    @unittest.skip
    def test_G_value__lungcancer(self):
        Omega = TestGStat.Omega['lungcancer']
        lungcancer = TestGStat.DatasetMatrix['lungcancer']

        ASIA    = Variable(lungcancer.get_column_by_label('X', 'ASIA'), 'ASIA')
        TUB     = Variable(lungcancer.get_column_by_label('X', 'TUB'), 'TUB')
        EITHER  = Variable(lungcancer.get_column_by_label('X', 'EITHER'), 'EITHER')
        LUNG    = Variable(lungcancer.get_column_by_label('X', 'LUNG'), 'LUNG')
        SMOKE   = Variable(lungcancer.get_column_by_label('X', 'SMOKE'), 'SMOKE')
        BRONC   = Variable(lungcancer.get_column_by_label('X', 'BRONC'), 'BRONC')
        DYSP    = Variable(lungcancer.get_column_by_label('Y', 'DYSP'), 'DYSP')
        XRAY    = Variable(lungcancer.get_column_by_label('Y', 'XRAY'), 'XRAY')

        self.print_pmf(SMOKE)
        print()
        self.print_pmf(LUNG)
        print()
        self.print_pmf(BRONC)
        print()
        self.print_cpmf(LUNG, SMOKE)
        print()
        self.print_cpmf(BRONC, SMOKE)
        print()

        self.analyze_variable_g_values(LUNG, SMOKE, Omega)
        self.analyze_variable_g_values(BRONC, SMOKE, Omega)
        self.analyze_variable_g_values(LUNG, BRONC, Omega)
        self.analyze_variable_g_values(LUNG, BRONC, SMOKE)


    def test_cmi__survey(self):
        Omega = TestGStat.Omega['survey']
        survey = TestGStat.DatasetMatrix['survey']
        AGE = Variable(survey.get_column_by_label('X', 'AGE'))
        SEX = Variable(survey.get_column_by_label('X', 'SEX'))
        EDU = Variable(survey.get_column_by_label('X', 'EDU'))
        OCC = Variable(survey.get_column_by_label('X', 'OCC'))
        R = Variable(survey.get_column_by_label('X', 'R'))
        TRN = Variable(survey.get_column_by_label('Y', 'TRN'))

        self.print_pmf(SMOKE)
        print()
        self.print_pmf(LUNG)
        print()
        self.print_pmf(BRONC)
        print()
        self.print_cpmf(LUNG, SMOKE)
        print()
        self.print_cpmf(BRONC, SMOKE)
        print()


    def configure_datasetmatrix(self, dm_label):
        configuration = {}
        if dm_label == 'lungcancer':
            configuration['sourcepath'] = Path('testfiles', 'bif_files', 'lungcancer.bif')
            configuration['sample_count'] = int(2e5)
            configuration['random_seed'] = 42*42
            configuration['values_as_indices'] = False
            configuration['objectives'] = ['DYSP', 'XRAY']

        if dm_label == 'survey':
            configuration['sourcepath'] = Path('testfiles', 'bif_files', 'survey.bif')
            configuration['sample_count'] = int(2e5)
            configuration['random_seed'] = 42*42
            configuration['values_as_indices'] = False
            configuration['objectives'] = ['TRN']
        return configuration


    def prepare_datasetmatrices(self):
        TestGStat.DatasetMatrix = {}
        TestGStat.Omega = {}

        dataset_folder = Path('testfiles', 'tmp', 'test_gstat_dm')
        for dm_label in ['survey', 'lungcancer']:
            configuration = self.configure_datasetmatrix(dm_label)
            bayesian_network = util.read_bif_file(configuration['sourcepath'])
            try:
                datasetmatrix = DatasetMatrix(dm_label)
                datasetmatrix.load(dataset_folder)
                TestGStat.DatasetMatrix[dm_label] = datasetmatrix
            except:
                bayesian_network.finalize()
                sbnds = SampledBayesianNetworkDatasetSource(configuration)
                sbnds.reset_random_seed = True
                datasetmatrix = sbnds.create_dataset_matrix(dm_label)
                datasetmatrix.finalize()
                datasetmatrix.save(dataset_folder)
                TestGStat.DatasetMatrix[dm_label] = datasetmatrix
                print('Dataset rebuilt.')
            TestGStat.Omega[dm_label] = Omega(configuration['sample_count'])
        TestGStat.ClassIsSetUp = True


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

