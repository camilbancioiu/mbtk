import unittest
import os
import shutil
from pathlib import Path

from mbff.dataset.DatasetMatrix import DatasetMatrix
from mbff.math.Variable import Omega
import mbff.utilities.functions as util
from mbff.dataset.sources.SampledBayesianNetworkDatasetSource import SampledBayesianNetworkDatasetSource

class TestBase(unittest.TestCase):

    TestFilesRootFolder = 'testfiles/tmp'
    TestsTagsToExclude = []


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initTestResources()


    def tag_excluded(tag):
        return tag in TestBase.TestsTagsToExclude


    def setUp(self):
        if not self.ResourcesSetUp:
            self.prepareTestResources()
            self.ResourcesSetUp = True


    def initTestResources(self):
        self.DatasetsInUse = list()
        self.DatasetMatrixFolder = Path('testfiles', 'tmp', 'test_dm')
        self.DatasetMatrices = None
        self.Omega = None
        self.BayesianNetworks = None
        self.ResourcesSetUp = False


    def prepareTestResources(self):
        self.prepare_Bayesian_networks();
        self.prepare_datasetmatrices()
        self.prepare_omega_variables()


    def tearDown(self):
        import gc; gc.collect()


    def ensure_tmp_subfolder(self, subfolder):
        path = Path(TestBase.TestFilesRootFolder + '/' + subfolder)
        path.mkdir(parents=True, exist_ok=True)
        return path


    def ensure_empty_tmp_subfolder(self, subfolder):
        try:
            shutil.rmtree(TestBase.TestFilesRootFolder + '/' + subfolder)
        except FileNotFoundError:
            pass
        path = Path(TestBase.TestFilesRootFolder + '/' + subfolder)
        path.mkdir(parents=True, exist_ok=True)
        return path


    def configure_dataset(self, dm_label):
        return dict()


    def prepare_Bayesian_networks(self):
        self.BayesianNetworks = dict()
        for dm_label in self.DatasetsInUse:
            configuration = self.configure_dataset(dm_label)
            bayesian_network = util.read_bif_file(configuration['sourcepath'])
            bayesian_network.finalize()
            self.BayesianNetworks[dm_label] = bayesian_network


    def prepare_datasetmatrices(self):
        self.DatasetMatrices = dict()

        for dm_label in self.DatasetsInUse:
            configuration = self.configure_dataset(dm_label)
            try:
                datasetmatrix = DatasetMatrix(dm_label)
                datasetmatrix.load(self.DatasetMatrixFolder)
                self.DatasetMatrices[dm_label] = datasetmatrix
            except:
                sbnds = SampledBayesianNetworkDatasetSource(configuration)
                sbnds.reset_random_seed = True
                datasetmatrix = sbnds.create_dataset_matrix(dm_label)
                datasetmatrix.finalize()
                datasetmatrix.save(self.DatasetMatrixFolder)
                self.DatasetMatrices[dm_label] = datasetmatrix


    def prepare_omega_variables(self):
        self.Omega = dict()
        for dm_label in self.DatasetsInUse:
            configuration = self.configure_dataset(dm_label)
            self.Omega[dm_label] = Omega(configuration['sample_count'])

