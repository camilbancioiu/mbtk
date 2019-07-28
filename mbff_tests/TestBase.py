import unittest
import gc
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


    @classmethod
    def setUpClass(testClass):
        if not testClass.ResourcesSetUp:
            testClass.initTestResources()
            testClass.prepareTestResources()
            testClass.ResourcesSetUp = True


    @classmethod
    def initTestResources(testClass):
        testClass.DatasetsInUse = list()
        testClass.DatasetMatrixFolder = Path('testfiles', 'tmp', 'test_dm')
        testClass.DatasetMatrices = None
        testClass.OmegaVariables = None
        testClass.BayesianNetworks = None
        testClass.ResourcesSetUp = False


    @classmethod
    def prepareTestResources(testClass):
        testClass.prepare_Bayesian_networks()
        testClass.prepare_datasetmatrices()
        testClass.prepare_omega_variables()


    @classmethod
    def configure_dataset(testClass, dm_label):
        return dict()


    @classmethod
    def prepare_Bayesian_networks(testClass):
        testClass.BayesianNetworks = dict()
        for dm_label in testClass.DatasetsInUse:
            configuration = testClass.configure_dataset(dm_label)
            bayesian_network = util.read_bif_file(configuration['sourcepath'])
            bayesian_network.finalize()
            testClass.BayesianNetworks[dm_label] = bayesian_network


    @classmethod
    def prepare_datasetmatrices(testClass):
        testClass.DatasetMatrices = dict()

        for dm_label in testClass.DatasetsInUse:
            testClass.DatasetMatrices[dm_label] = testClass.prepare_datasetmatrix(dm_label)


    @classmethod
    def prepare_datasetmatrix(testClass, label):
        configuration = testClass.configure_dataset(label)
        try:
            datasetmatrix = DatasetMatrix(label)
            datasetmatrix.load(testClass.DatasetMatrixFolder)
        except:
            sbnds = SampledBayesianNetworkDatasetSource(configuration)
            sbnds.reset_random_seed = True
            datasetmatrix = sbnds.create_dataset_matrix(label)
            datasetmatrix.finalize()
            datasetmatrix.save(testClass.DatasetMatrixFolder)
        return datasetmatrix


    @classmethod
    def prepare_omega_variables(testClass):
        testClass.OmegaVariables = dict()
        for dm_label in testClass.DatasetsInUse:
            configuration = testClass.configure_dataset(dm_label)
            testClass.OmegaVariables[dm_label] = Omega(configuration['sample_count'])


    def tearDown(self):
        gc.collect()


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
