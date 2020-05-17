import pytest
from pathlib import Path
import tests.utilities as testutil
import mbff.utilities.functions as util


@pytest.fixture(scope='session')
def bn_lungcancer():
    bn = util.read_bif_file(Path(testutil.bif_folder, 'lungcancer.bif'))
    bn.finalize()
    return bn



@pytest.fixture(scope='session')
def bn_lc_repaired():
    bn = util.read_bif_file(Path(testutil.bif_folder, 'lc_repaired.bif'))
    bn.finalize()
    return bn



@pytest.fixture(scope='session')
def ds_lungcancer_4e4():
    """ The 'lungcancer' dataset with 4e4 samples"""
    configuration = dict()
    configuration['label'] = 'ds_lungcancer_4e4'
    configuration['sourcepath'] = testutil.bif_folder / 'lungcancer.bif'
    configuration['sample_count'] = int(4e4)
    configuration['random_seed'] = 129
    configuration['values_as_indices'] = False
    configuration['objectives'] = []
    return testutil.MockDataset(configuration)



@pytest.fixture(scope='session')
def ds_lc_repaired_4e4():
    configuration = dict()
    configuration['label'] = 'ds_lc_repaired_4e4'
    configuration['sourcepath'] = testutil.bif_folder / 'lc_repaired.bif'
    configuration['sample_count'] = int(4e4)
    configuration['values_as_indices'] = True
    configuration['objectives'] = []
    configuration['method'] = 'exact'
    configuration['random_seed'] = 1984
    return testutil.MockDataset(configuration)



@pytest.fixture(scope='session')
def ds_lc_repaired_8e3():
    configuration = dict()
    configuration['label'] = 'ds_lc_repaired_8e3'
    configuration['sourcepath'] = testutil.bif_folder / 'lc_repaired.bif'
    configuration['sample_count'] = int(8e3)
    configuration['values_as_indices'] = True
    configuration['objectives'] = []
    configuration['method'] = 'exact'
    configuration['random_seed'] = 1984
    return testutil.MockDataset(configuration)



@pytest.fixture(scope='session')
def adtree_lc_repaired_8e3_llta200(ds_lc_repaired_8e3):
    configuration = dict()
    configuration['label'] = 'adtree_lc_repaired_8e3_llta200'
    configuration['leaf_list_threshold'] = 200
    return testutil.prepare_AD_tree(configuration, ds_lc_repaired_8e3.datasetmatrix)
